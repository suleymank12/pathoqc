"""Veri setlerini diskten okuyan kod.

İki veri seti aynı klasör düzenini kullanır:

    <veri seti>/training/        yamalar; etiket dosya adının içindedir
    <veri seti>/val/img, mask    yama + piksel piksel boyanmış maske
    <veri seti>/test/img, mask   aynı düzen

Eğitimde yalnızca "bu yamada hangi dokular var" bilgisi (yama etiketi) kullanılır.
Maskeler yalnızca ölçüm içindir.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

NUM_CLASSES = 4

# Maskelerde 0-3 doku sınıflarıdır. 4 arka plandır (boş/beyaz alan) ve ölçüme katılmaz.
# Veri setini yayınlayan ekibin kodu da 4'ü dışarıda bırakır; aynısını yapmazsak sonuçlar karşılaştırılamaz.
IGNORE_INDEX = 4

# ImageNet ile ön eğitilmiş omurgaların beklediği renk ortalaması ve sapması.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetInfo:
    """Bir veri setinin değişmeyen bilgileri."""

    folder: str  # zip açılınca çıkan klasörün adı
    class_names: tuple[str, ...]  # etiketteki ve maskedeki sıra (0, 1, 2, 3)


DATASETS = {
    # TE: tümör epiteli, NEC: nekroz, LYM: lenfosit, TAS: tümörle ilişkili stroma
    "luad": DatasetInfo(folder="LUAD-HistoSeg", class_names=("TE", "NEC", "LYM", "TAS")),
    # TUM: tümör, STR: stroma, LYM: lenfosit, NEC: nekroz
    "bcss": DatasetInfo(folder="BCSS-WSSS", class_names=("TUM", "STR", "LYM", "NEC")),
}

# Köşeli parantez içindeki 0, 1 ve boşlukları yakalar: "[1 0 0 1]" (LUAD) ve "[1001]" (BCSS).
_LABEL_PATTERN = re.compile(r"\[([01 ]+)\]")


def parse_label(filename: str) -> tuple[int, ...]:
    """Dosya adından yama etiketini çıkarır.

    "1031280-10364-10000-[1 0 0 1].png" -> (1, 0, 0, 1)
    "TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500+0[1101].png" -> (1, 1, 0, 1)
    """
    match = _LABEL_PATTERN.search(filename)
    if match is None:
        raise ValueError(f"Dosya adında etiket yok: {filename}")
    label = tuple(int(ch) for ch in match.group(1) if ch in "01")
    if len(label) != NUM_CLASSES:
        raise ValueError(f"Etiket {NUM_CLASSES} haneli olmalı, {len(label)} hane bulundu: {filename}")
    return label


def is_real_image(path: Path) -> bool:
    """Gerçek bir yama mı?

    LUAD zip'i Mac'te hazırlandığı için içinden "._ad.png" adlı çöp dosyalar ve "__MACOSX"
    klasörü çıkıyor. Bunlar resim değildir; açmaya çalışınca hata verirler.
    """
    return (
        path.is_file()
        and path.suffix.lower() == ".png"
        and not path.name.startswith("._")
        and "__MACOSX" not in path.parts
    )


def list_images(folder: Path) -> list[Path]:
    """Klasördeki yamaları ada göre sıralı döndürür. Sıralama her çalıştırmada aynı sonucu verir."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Klasör yok: {folder}")
    images = sorted(p for p in folder.iterdir() if is_real_image(p))
    if not images:
        raise FileNotFoundError(f"Klasörde yama yok: {folder}")
    return images


def build_train_transform() -> transforms.Compose:
    """Eğitim dönüşümü: rastgele çevirme + tensöre çevirme + renk normalizasyonu.

    Doku yamalarında yön bilgisi yoktur (üstü altı belli değildir), bu yüzden çevirmek etiketi bozmaz.
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """Ölçüm dönüşümü: rastgelelik yok, yalnızca tensöre çevirme + normalizasyon."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class PatchLabelDataset(Dataset):
    """Eğitim kümesi: yama + yama etiketi (hangi dokular var).

    Her öğe: (görüntü tensörü [3, Y, G], etiket tensörü [4], dosya adı)
    """

    def __init__(self, folder: Path, transform=None, limit: int | None = None):
        self.paths = list_images(folder)[:limit]
        self.transform = transform or build_eval_transform()
        # Etiketleri baştan okuyup saklıyoruz: sınıf dengesizliğini hesaplamak için lazım.
        self.labels = torch.tensor([parse_label(p.name) for p in self.paths], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return image, self.labels[index], path.stem


class MaskDataset(Dataset):
    """Doğrulama / test kümesi: yama + piksel maskesi.

    Her öğe: (görüntü tensörü [3, Y, G], maske tensörü [Y, G] değerleri 0-4, dosya adı)
    """

    def __init__(self, folder: Path, transform=None, limit: int | None = None):
        folder = Path(folder)
        self.image_paths = list_images(folder / "img")[:limit]
        self.mask_dir = folder / "mask"
        self.transform = transform or build_eval_transform()
        missing = [p.name for p in self.image_paths if not (self.mask_dir / p.name).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} yamanın maskesi yok, ilki: {missing[0]}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        # Maskeler "paletli" PNG'dir: her piksel bir renk değil, sınıf numarası (0-4) tutar.
        with Image.open(self.mask_dir / path.name) as mask_image:
            mask = np.array(mask_image, dtype=np.int64)
        if mask.max() > IGNORE_INDEX:
            raise ValueError(f"Maskede beklenmeyen değer {mask.max()}: {path.name}")
        return image, torch.from_numpy(mask), path.stem
