"""Aşama 1, dondurulmuş omurga: özellikleri bir kez çıkarır, küçük başı birden çok seed ile eğitir.

Örnek:
    python scripts/train_stage1_frozen.py --dataset luad --backbone midnight \\
        --data-root /kaggle/tmp/veri --cache-dir /kaggle/tmp/ozellik --out-dir /kaggle/working/outputs

İki adım vardır:
  1. Omurga her yamanın özellik haritasını hesaplar, diske yazılır (uzun süren kısım, bir kez yapılır).
  2. Küçük baş bu özelliklerle eğitilir (kısa süren kısım, her seed için tekrarlanır).

Bu betik test kümesine HİÇ dokunmaz. Model seçimi yalnızca doğrulama (val) mIoU'su ile yapılır.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pathoqc import backbones
from pathoqc.cache import extract_features, load_features
from pathoqc.cam import CamHead, cams_to_mask, predict_present
from pathoqc.data import DATASETS, MaskDataset, PatchLabelDataset, build_eval_transform
from pathoqc.metrics import SegmentationMeter


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aşama 1: dondurulmuş omurga + CAM başı")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--backbone", choices=sorted(backbones.FROZEN_BACKBONES), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("features"), help="Özellik dosyalarının yazılacağı klasör")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Her kümeden ilk N yamayı kullan (yalnızca hızlı deneme için)")
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(head: nn.Module, features: torch.Tensor, masks: torch.Tensor, device: str, batch_size: int) -> SegmentationMeter:
    """Doğrulama özelliklerinden CAM maskesi üretir ve gerçek maskeyle karşılaştırır."""
    head.eval()
    meter = SegmentationMeter()
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].to(device).float()
        logits, cams = head(batch)
        prediction = cams_to_mask(cams, predict_present(logits), size=masks.shape[-2:])
        meter.update(masks[start : start + batch_size].numpy(), prediction.cpu().numpy())
    return meter


def train_one_seed(seed, train_features, labels, val_features, val_masks, args, device, class_names, run_dir) -> dict:
    """Tek bir seed ile başı eğitir; en iyi doğrulama sonucunu ve bütün gidişatı döndürür."""
    set_seed(seed)
    head = CamHead(in_channels=train_features.shape[1]).to(device)

    # Sınıf dengesizliği: az görülen sınıfın hatası daha pahalı sayılır (en fazla 10 kat).
    positives = labels.sum(dim=0).clamp_min(1)
    pos_weight = ((len(labels) - positives) / positives).clamp(max=10.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    steps_per_epoch = len(train_features) // args.batch_size
    if steps_per_epoch == 0:
        raise ValueError("Eğitim kümesi bir batch'ten küçük; --batch-size değerini düşür")
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.epochs * steps_per_epoch, power=0.9)
    generator = torch.Generator().manual_seed(seed)  # karıştırma sırası da seed'e bağlı

    best, history = None, []
    for epoch in range(args.epochs):
        head.train()
        order = torch.randperm(len(train_features), generator=generator)
        loss_sum = 0.0
        for step in range(steps_per_epoch):
            index = order[step * args.batch_size : (step + 1) * args.batch_size]
            batch = train_features[index].to(device).float()
            target = labels[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = head(batch)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()

        summary = evaluate(head, val_features, val_masks, device, args.batch_size).summary(class_names)
        history.append({"epoch": epoch + 1, "loss": round(loss_sum / steps_per_epoch, 4), "val": summary})
        if best is None or summary["miou"] > best["val"]["miou"]:
            best = {"epoch": epoch + 1, "val": summary}
            torch.save({"head": head.state_dict(), **best}, run_dir / f"best_seed{seed}.pt")

    print(json.dumps({"seed": seed, "best_epoch": best["epoch"], "best_val": best["val"]}, ensure_ascii=False))
    return {"seed": seed, "best_epoch": best["epoch"], "best_val": best["val"], "history": history}


def main(argv=None) -> dict:
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = backbones.FROZEN_BACKBONES[args.backbone]
    info = DATASETS[args.dataset]
    root = args.data_root / info.folder
    run_dir = args.out_dir / f"{args.dataset}_stage1_{args.backbone}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Rastgele çevirme yok: özellikler bir kez hesaplanıp saklandığı için her yamanın tek hali vardır.
    transform = build_eval_transform(spec.mean, spec.std)
    train_set = PatchLabelDataset(root / "training", transform, limit=args.limit)
    val_set = MaskDataset(root / "val", transform, limit=args.limit)

    tag = f"{args.dataset}_{args.backbone}" + (f"_limit{args.limit}" if args.limit else "")
    train_path = args.cache_dir / f"{tag}_training.npy"
    val_path = args.cache_dir / f"{tag}_val.npy"
    extraction = None
    if not (train_path.is_file() and val_path.is_file()):
        backbone = backbones.load_frozen_backbone(args.backbone)
        started = time.time()
        extract_features(backbone, train_set, train_path, device, args.batch_size, args.workers)
        extract_features(backbone, val_set, val_path, device, args.batch_size, args.workers)
        seconds = time.time() - started
        extraction = {"seconds": round(seconds, 1), "images_per_second": round((len(train_set) + len(val_set)) / seconds, 1)}
        print(json.dumps({"ozellik_cikarma": extraction}, ensure_ascii=False))
        del backbone
        if device == "cuda":
            torch.cuda.empty_cache()

    train_features = load_features(train_path)
    val_features = load_features(val_path)
    if len(train_features) != len(train_set) or len(val_features) != len(val_set):
        raise RuntimeError("Önbellekteki özellik sayısı veri kümesiyle uyuşmuyor; önbellek klasörünü silip yeniden çalıştır")
    val_masks = torch.stack([val_set.read_mask(i).to(torch.uint8) for i in range(len(val_set))])
    print(json.dumps({"ozellik_boyutu": list(train_features.shape), "val": list(val_features.shape)}))

    runs = [
        train_one_seed(seed, train_features, train_set.labels, val_features, val_masks, args, device, info.class_names, run_dir)
        for seed in args.seeds
    ]
    scores = [run["best_val"]["miou"] for run in runs]
    result = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "backbone": {"model_id": spec.model_id, "license": spec.license},
        "extraction": extraction,
        "val_miou_per_seed": scores,
        "val_miou_mean": round(float(np.mean(scores)), 2),
        # Sapma, seed'ler arası örneklem sapmasıdır (n-1'e bölünür); tek seed varsa hesaplanamaz.
        "val_miou_std": round(float(np.std(scores, ddof=1)), 2) if len(scores) > 1 else None,
        "runs": runs,
    }
    (run_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("val_miou_per_seed", "val_miou_mean", "val_miou_std")}))
    return result


if __name__ == "__main__":
    main()
