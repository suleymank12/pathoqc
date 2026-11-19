"""Özellik önbelleği.

Dondurulmuş omurga aynı görüntüye her seferinde aynı özelliği verir. O yüzden her yama için
özelliği BİR KEZ hesaplayıp diske yazarız. Sonraki bütün eğitimler (farklı seed'ler dahil)
dev modeli tekrar çalıştırmadan bu dosyadan okur: dakikalar yerine saniyeler sürer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def extract_features(
    backbone: torch.nn.Module, dataset: Dataset, path: Path, device: str, batch_size: int = 64, workers: int = 4
) -> Path:
    """Veri kümesindeki her yamanın özellik haritasını hesaplar ve `path` dosyasına yazar.

    Dosya: [N, C, h, w] boyutlu, yarım hassasiyetli (float16) bir dizi. Sıra, veri kümesinin sırasıdır.
    Önce geçici dosyaya yazılır, bitince asıl ada taşınır: yarım kalmış bir dosya asla kullanılmaz.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npy")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    backbone = backbone.to(device).eval()

    array, start = None, 0
    for batch in loader:
        images = batch[0].to(device)
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            features = backbone(images)
        features = features.to(torch.float16).cpu().numpy()
        if array is None:  # boyutu ilk gruptan öğrenip dosyayı o zaman açarız
            array = np.lib.format.open_memmap(
                temporary, mode="w+", dtype=np.float16, shape=(len(dataset), *features.shape[1:])
            )
        array[start : start + len(features)] = features
        start += len(features)
    if array is None or start != len(dataset):
        raise RuntimeError(f"Özellik çıkarma eksik kaldı: {start}/{len(dataset)}")
    array.flush()
    del array
    temporary.replace(path)
    return path


def load_features(path: Path) -> torch.Tensor:
    """Önbellek dosyasının tamamını belleğe alır: [N, C, h, w], float16."""
    return torch.from_numpy(np.load(Path(path)))
