"""Aşama 1, dondurulmuş omurga: özellikleri bir kez çıkarır, küçük başı farklı ayarlarla eğitir.

Örnek:
    python scripts/train_stage1_frozen.py --dataset luad --backbone midnight \\
        --data-root /kaggle/tmp/veri --cache-dir /kaggle/tmp/ozellik --out-dir /kaggle/working/outputs

Üç adım vardır:
  1. Omurga her yamanın özellik haritasını hesaplar, diske yazılır (uzun süren kısım, bir kez yapılır).
  2. Küçük baş her öğrenme hızı ve her seed için ayrı ayrı eğitilir (kısa süren kısım).
     Eğitim sırasında her `--eval-every` adımda doğrulama kümesinde ölçüm yapılır.
  3. Seed'lerin ortalamasına göre en iyi (öğrenme hızı, adım sayısı, maske yöntemi) seçilir.

Neden sık ölçüm: ilk denemede en iyi maskeler eğitimin en başında çıktı, baş eğitildikçe bozuldu.
Epoch sonunda ölçünce en iyi noktayı kaçırıyorduk.

Bu betik test kümesine HİÇ dokunmaz. Bütün seçimler yalnızca doğrulama (val) kümesiyle yapılır.
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

# Maske üretmenin iki yolu (ayrıntısı cam.py içinde): klasik CAM ölçeklemesi ve ham skor.
VARIANTS = {"normalized": True, "raw": False}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aşama 1: dondurulmuş omurga + CAM başı")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--backbone", choices=sorted(backbones.FROZEN_BACKBONES), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("features"), help="Özellik dosyalarının yazılacağı klasör")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--lrs", type=float, nargs="+", default=[3e-5, 1e-4, 3e-4, 1e-3], help="Denenecek öğrenme hızları")
    parser.add_argument("--max-steps", type=int, default=600, help="Her koşuda en fazla kaç adım eğitilecek")
    parser.add_argument("--eval-every", type=int, default=50, help="Kaç adımda bir doğrulama ölçümü yapılacak")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224, help="Yama omurgaya verilmeden önce bu boyuta getirilir (224 = olduğu gibi)")
    parser.add_argument("--extract-batch-size", type=int, default=64, help="Özellik çıkarırken grup boyutu; büyük görüntüde belleğe sığması için küçültülür")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Her kümeden eşit aralıklı N yama kullan (hızlı deneme için)")
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(head: nn.Module, features: torch.Tensor, masks: torch.Tensor, device: str, batch_size: int) -> dict:
    """Doğrulama özelliklerinden iki yöntemle maske üretir; her yöntem için ayrı ölçüm döndürür."""
    head.eval()
    meters = {name: SegmentationMeter() for name in VARIANTS}
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].to(device).float()
        logits, cams = head(batch)
        present = predict_present(logits)
        target = masks[start : start + batch_size].numpy()
        for name, normalize in VARIANTS.items():
            prediction = cams_to_mask(cams, present, size=masks.shape[-2:], normalize=normalize)
            meters[name].update(target, prediction.cpu().numpy())
    return meters


def train_one_run(lr, seed, train_features, labels, val_features, val_masks, args, device, class_names) -> list[dict]:
    """Tek bir (öğrenme hızı, seed) koşusu. Her ölçüm noktasının sonucunu ve o andaki baş ağırlıklarını döndürür."""
    set_seed(seed)
    head = CamHead(in_channels=train_features.shape[1]).to(device)

    # Sınıf dengesizliği: az görülen sınıfın hatası daha pahalı sayılır (en fazla 10 kat).
    positives = labels.sum(dim=0).clamp_min(1)
    pos_weight = ((len(labels) - positives) / positives).clamp(max=10.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    # Öğrenme hızı sabit tutulur. Böylece "k adımda dur" kararı, toplam adım sayısından bağımsız olur.
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)  # karıştırma sırası da seed'e bağlı

    points, loss_sum, loss_count, order, cursor = [], 0.0, 0, None, 0
    for step in range(1, args.max_steps + 1):
        if order is None or cursor + args.batch_size > len(order):  # veri bitti: yeniden karıştır
            order, cursor = torch.randperm(len(train_features), generator=generator), 0
        index = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size

        head.train()
        optimizer.zero_grad(set_to_none=True)
        logits, _ = head(train_features[index].to(device).float())
        loss = criterion(logits, labels[index].to(device))
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        loss_count += 1

        if step % args.eval_every == 0 or step == args.max_steps:
            meters = evaluate(head, val_features, val_masks, device, args.batch_size)
            points.append({
                "step": step,
                "loss": round(loss_sum / loss_count, 4),  # son ölçümden beri ortalama kayıp
                "val": {name: meter.summary(class_names) for name, meter in meters.items()},
                "state": {key: value.detach().cpu().clone() for key, value in head.state_dict().items()},
            })
            loss_sum, loss_count = 0.0, 0
    return points


def select_best(curves: dict, seeds: list[int]) -> dict:
    """Seed ortalaması en yüksek olan (öğrenme hızı, yöntem, adım) üçlüsünü bulur.

    Her seed için ayrı en iyi adım SEÇİLMEZ: bütün seed'ler aynı ayarla değerlendirilir.
    Aksi halde her seed kendi şanslı noktasını seçer ve ortalama olduğundan iyi görünür.
    """
    best = None
    for lr, by_seed in curves.items():
        steps = [point["step"] for point in by_seed[seeds[0]]]
        for position, step in enumerate(steps):
            for variant in VARIANTS:
                scores = [by_seed[seed][position]["val"][variant]["miou"] for seed in seeds]
                mean = float(np.mean(scores))
                if best is None or mean > best["val_miou_mean"]:
                    best = {"lr": lr, "variant": variant, "step": step, "position": position,
                            "val_miou_per_seed": scores, "val_miou_mean": mean}
    return best


def main(argv=None) -> dict:
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = backbones.FROZEN_BACKBONES[args.backbone]
    info = DATASETS[args.dataset]
    root = args.data_root / info.folder
    run_dir = args.out_dir / f"{args.dataset}_stage1_{args.backbone}_{args.image_size}px"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Rastgele çevirme yok: özellikler bir kez hesaplanıp saklandığı için her yamanın tek hali vardır.
    transform = build_eval_transform(spec.mean, spec.std, None if args.image_size == 224 else args.image_size)
    train_set = PatchLabelDataset(root / "training", transform, limit=args.limit)
    val_set = MaskDataset(root / "val", transform, limit=args.limit)

    tag = f"{args.dataset}_{args.backbone}_{args.image_size}px" + (f"_limit{args.limit}" if args.limit else "")
    train_path = args.cache_dir / f"{tag}_training.npy"
    val_path = args.cache_dir / f"{tag}_val.npy"
    extraction = None
    if not (train_path.is_file() and val_path.is_file()):
        backbone = backbones.load_frozen_backbone(args.backbone)
        started = time.time()
        extract_features(backbone, train_set, train_path, device, args.extract_batch_size, args.workers)
        extract_features(backbone, val_set, val_path, device, args.extract_batch_size, args.workers)
        seconds = time.time() - started
        extraction = {"seconds": round(seconds, 1), "images_per_second": round((len(train_set) + len(val_set)) / seconds, 1)}
        print(json.dumps({"ozellik_cikarma": extraction}, ensure_ascii=False), flush=True)
        del backbone
        if device == "cuda":
            torch.cuda.empty_cache()

    train_features = load_features(train_path)
    val_features = load_features(val_path)
    if len(train_features) != len(train_set) or len(val_features) != len(val_set):
        raise RuntimeError("Önbellekteki özellik sayısı veri kümesiyle uyuşmuyor; önbellek klasörünü silip yeniden çalıştır")
    val_masks = torch.stack([val_set.read_mask(i).to(torch.uint8) for i in range(len(val_set))])
    print(json.dumps({"ozellik_boyutu": list(train_features.shape), "val": list(val_features.shape)}), flush=True)

    # curves[öğrenme hızı][seed] = o koşunun ölçüm noktaları
    curves = {}
    for lr in args.lrs:
        curves[lr] = {}
        for seed in args.seeds:
            started = time.time()
            curves[lr][seed] = train_one_run(
                lr, seed, train_features, train_set.labels, val_features, val_masks, args, device, info.class_names
            )
            trace = {name: [point["val"][name]["miou"] for point in curves[lr][seed]] for name in VARIANTS}
            print(json.dumps({"lr": lr, "seed": seed, "saniye": round(time.time() - started, 1), **trace}), flush=True)

    best = select_best(curves, args.seeds)
    chosen = {seed: curves[best["lr"]][seed][best["position"]] for seed in args.seeds}
    for seed, point in chosen.items():
        torch.save({"head": point["state"], "lr": best["lr"], "step": best["step"], "variant": best["variant"]},
                   run_dir / f"best_seed{seed}.pt")

    scores = best["val_miou_per_seed"]
    per_class = {
        name: round(float(np.mean([chosen[seed]["val"][best["variant"]]["iou"][name] for seed in args.seeds])), 2)
        for name in info.class_names
    }
    result = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "backbone": {"model_id": spec.model_id, "license": spec.license},
        "extraction": extraction,
        "selected": {
            "lr": best["lr"], "variant": best["variant"], "step": best["step"],
            "val_miou_per_seed": scores,
            "val_miou_mean": round(best["val_miou_mean"], 2),
            # Sapma, seed'ler arası örneklem sapmasıdır (n-1'e bölünür); tek seed varsa hesaplanamaz.
            "val_miou_std": round(float(np.std(scores, ddof=1)), 2) if len(scores) > 1 else None,
            "val_iou_mean_per_class": per_class,
            "val_per_seed": {str(seed): chosen[seed]["val"][best["variant"]] for seed in args.seeds},
        },
        # Bütün eğriler (ağırlıklar hariç): her öğrenme hızı ve seed için adım adım doğrulama mIoU'su.
        "curves": {
            str(lr): {
                str(seed): [{"step": p["step"], "loss": p["loss"], **{n: p["val"][n]["miou"] for n in VARIANTS}} for p in points]
                for seed, points in by_seed.items()
            }
            for lr, by_seed in curves.items()
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"secilen": {k: result["selected"][k] for k in
                                  ("lr", "variant", "step", "val_miou_per_seed", "val_miou_mean", "val_miou_std", "val_iou_mean_per_class")}},
                     ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    main()
