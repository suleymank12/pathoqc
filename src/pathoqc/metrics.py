"""Segmentasyon ölçümleri: IoU, mIoU, FwIoU, Dice.

Hesap, veri setini yayınlayan ekibin koduyla aynı kurala uyar: maskede değeri 4 olan
(arka plan) pikseller hiç sayılmaz. Böylece sayılarımız makalelerdeki sayılarla karşılaştırılabilir.
"""

from __future__ import annotations

import numpy as np


class SegmentationMeter:
    """Tüm görüntüler boyunca bir karışıklık matrisi biriktirir, sonunda ölçümleri verir.

    Karışıklık matrisi: satır = gerçek sınıf, sütun = tahmin edilen sınıf.
    matrix[1, 3] = "gerçekte 1 olan kaç piksele 3 dedik".
    """

    def __init__(self, num_classes: int = 4, ignore_index: int = 4):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, target: np.ndarray, prediction: np.ndarray) -> None:
        """Bir grup maske ve tahmini ekler. İkisi aynı boyutta tam sayı dizisi olmalı."""
        target = np.asarray(target)
        prediction = np.asarray(prediction)
        if target.shape != prediction.shape:
            raise ValueError(f"Boyutlar farklı: maske {target.shape}, tahmin {prediction.shape}")
        valid = (target >= 0) & (target < self.num_classes)  # arka planı (4) dışarıda bırakır
        target = target[valid].astype(np.int64)
        prediction = prediction[valid].astype(np.int64)
        if prediction.size and (prediction.min() < 0 or prediction.max() >= self.num_classes):
            raise ValueError("Tahminde sınıf aralığı dışında değer var")
        # (gerçek, tahmin) çiftini tek sayıya çevirip sayıyoruz: hızlı ve döngüsüz.
        counts = np.bincount(target * self.num_classes + prediction, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def iou_per_class(self) -> np.ndarray:
        """Sınıf başına IoU = kesişim / birleşim. Sınıf hiç yoksa nan döner."""
        hit = np.diag(self.matrix).astype(np.float64)
        union = self.matrix.sum(axis=1) + self.matrix.sum(axis=0) - hit
        with np.errstate(divide="ignore", invalid="ignore"):
            return hit / union

    def dice_per_class(self) -> np.ndarray:
        """Sınıf başına Dice = 2 * kesişim / (gerçek + tahmin)."""
        hit = np.diag(self.matrix).astype(np.float64)
        total = self.matrix.sum(axis=1) + self.matrix.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2 * hit / total

    def miou(self) -> float:
        """Sınıfların IoU ortalaması. Veride hiç bulunmayan sınıf ortalamaya girmez."""
        return float(np.nanmean(self.iou_per_class()))

    def mdice(self) -> float:
        return float(np.nanmean(self.dice_per_class()))

    def fwiou(self) -> float:
        """Sıklık ağırlıklı IoU: çok pikseli olan sınıfın sözü daha çok geçer."""
        frequency = self.matrix.sum(axis=1) / self.matrix.sum()
        iou = self.iou_per_class()
        present = frequency > 0
        return float((frequency[present] * iou[present]).sum())

    def pixel_accuracy(self) -> float:
        return float(np.diag(self.matrix).sum() / self.matrix.sum())

    def summary(self, class_names: tuple[str, ...] | None = None) -> dict:
        """Hepsini tek sözlükte, yüzde olarak verir. Dosyaya yazmak ve ekrana basmak için."""
        names = class_names or tuple(str(i) for i in range(self.num_classes))
        return {
            "miou": round(100 * self.miou(), 2),
            "fwiou": round(100 * self.fwiou(), 2),
            "mdice": round(100 * self.mdice(), 2),
            "pixel_accuracy": round(100 * self.pixel_accuracy(), 2),
            "iou": {n: round(100 * float(v), 2) for n, v in zip(names, self.iou_per_class())},
        }
