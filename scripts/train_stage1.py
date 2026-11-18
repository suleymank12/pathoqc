"""Aşama 1: sınıflandırıcıyı yama etiketleriyle eğitir, CAM maskelerini doğrulama kümesinde ölçer.

Örnek:
    python scripts/train_stage1.py --dataset luad --data-root /kaggle/tmp/veri --out-dir /kaggle/working/outputs

Bu betik test kümesine HİÇ dokunmaz. Model seçimi yalnızca doğrulama (val) mIoU'su ile yapılır.
Test, bütün deneyler bittikten sonra ayrı bir betikle bir kez ölçülür.
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
from torch.utils.data import DataLoader

from pathoqc.cam import CamClassifier, cams_to_mask, predict_present
from pathoqc.data import DATASETS, MaskDataset, PatchLabelDataset, build_eval_transform, build_train_transform
from pathoqc.metrics import SegmentationMeter


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aşama 1: CAM sınıflandırıcı eğitimi")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="LUAD-HistoSeg ve BCSS-WSSS klasörlerinin bulunduğu yer")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true", help="ImageNet ağırlıklarını indirme (yalnızca test için)")
    parser.add_argument("--limit", type=int, default=None, help="Her kümeden ilk N yamayı kullan (yalnızca hızlı deneme için)")
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    """Rastgeleliği sabitler: aynı seed ile aynı sonuç çıksın diye."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """Veri yükleyen her yardımcı sürecin rastgeleliğini de sabitler."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> SegmentationMeter:
    """Doğrulama yamalarında CAM maskesi üretir ve gerçek maskeyle karşılaştırır."""
    model.eval()
    meter = SegmentationMeter()
    for images, masks, _ in loader:
        logits, cams = model(images.to(device))
        prediction = cams_to_mask(cams, predict_present(logits), size=masks.shape[-2:])
        meter.update(masks.numpy(), prediction.cpu().numpy())
    return meter


def main(argv=None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"  # yarım hassasiyet: GPU'da daha hızlı ve daha az bellek

    info = DATASETS[args.dataset]
    root = args.data_root / info.folder
    run_dir = args.out_dir / f"{args.dataset}_stage1_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_set = PatchLabelDataset(root / "training", build_train_transform(), limit=args.limit)
    val_set = MaskDataset(root / "val", build_eval_transform(), limit=args.limit)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, worker_init_fn=seed_worker, generator=generator, pin_memory=use_amp,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if len(train_loader) == 0:
        raise ValueError("Eğitim kümesi bir batch'ten küçük; --batch-size değerini düşür")

    model = CamClassifier(pretrained=not args.no_pretrained).to(device)

    # Sınıf dengesizliği: az görülen sınıfın hatası daha pahalı sayılır.
    # Ağırlık = (o sınıfın olmadığı yama sayısı) / (olduğu yama sayısı), en fazla 10.
    positives = train_set.labels.sum(dim=0).clamp_min(1)
    pos_weight = ((len(train_set) - positives) / positives).clamp(max=10.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    # Öğrenme hızı her adımda yavaşça sıfıra iner ("poly" azalma; bu alandaki makalelerin kullandığı).
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_steps, power=0.9)
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    # Kaldığı yerden devam: Kaggle oturumu 12 saatte kesilir, eğitim baştan başlamasın.
    start_epoch, best_miou, history = 0, -1.0, []
    last_path = run_dir / "last.pt"
    if last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_miou, history = state["epoch"], state["best_miou"], state["history"]
        print(f"Kaldığı yerden devam: {start_epoch}. epoch bitmişti")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        started, loss_sum = time.time(), 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                logits, _ = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item()

        summary = evaluate(model, val_loader, device).summary(info.class_names)
        record = {"epoch": epoch + 1, "loss": round(loss_sum / len(train_loader), 4),
                  "seconds": round(time.time() - started, 1), "val": summary}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if summary["miou"] > best_miou:
            best_miou = summary["miou"]
            torch.save({"model": model.state_dict(), "epoch": epoch + 1, "val": summary}, run_dir / "best.pt")
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict(), "epoch": epoch + 1, "best_miou": best_miou, "history": history},
            last_path,
        )

    result = {"config": {k: str(v) for k, v in vars(args).items()}, "best_val_miou": best_miou, "history": history}
    (run_dir / "history.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
