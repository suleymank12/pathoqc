"""Testlerin ortak yardımcıları: diskte minik, sahte bir veri seti kurar."""

import numpy as np
import pytest
from PIL import Image


def _save_image(path, rng, size):
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(path)


@pytest.fixture
def tiny_dataset(tmp_path):
    """LUAD-HistoSeg ile aynı klasör düzeninde 8 eğitim, 4 doğrulama yaması üretir."""
    rng = np.random.default_rng(0)
    size = 64
    root = tmp_path / "LUAD-HistoSeg"
    (root / "training").mkdir(parents=True)
    labels = ["1 0 0 1", "1 0 0 0", "0 1 0 1", "0 0 1 1", "1 1 0 0", "0 0 0 1", "1 0 1 0", "1 0 0 1"]
    for i, label in enumerate(labels):
        _save_image(root / "training" / f"p{i}-[{label}].png", rng, size)
    # Mac çöpü: okuyucu bunu görmezden gelmeli.
    (root / "training" / "._p0-[1 0 0 1].png").write_bytes(b"resim degil")

    for split in ("val", "test"):
        (root / split / "img").mkdir(parents=True)
        (root / split / "mask").mkdir(parents=True)
        for i in range(4):
            _save_image(root / split / "img" / f"v{i}.png", rng, size)
            mask = Image.fromarray(rng.integers(0, 5, (size, size), dtype=np.uint8), mode="P")
            mask.save(root / split / "mask" / f"v{i}.png")
    return tmp_path
