import numpy as np
import torch
from torch import nn

from pathoqc.cache import extract_features, load_features
from pathoqc.data import PatchLabelDataset


class FakeBackbone(nn.Module):
    """Gerçek omurganın yerine geçen minik model: her 16x16 bölgenin ortalama rengini verir."""

    def forward(self, images):
        return nn.functional.avg_pool2d(images, kernel_size=16)


def test_extract_features_writes_all_patches_in_order(tiny_dataset, tmp_path):
    dataset = PatchLabelDataset(tiny_dataset / "LUAD-HistoSeg" / "training")
    path = extract_features(FakeBackbone(), dataset, tmp_path / "cache" / "ozellik.npy", "cpu", batch_size=3, workers=0)

    features = load_features(path)
    assert features.shape == (8, 3, 4, 4) and features.dtype == torch.float16
    assert not (tmp_path / "cache" / "ozellik.tmp.npy").exists()

    # Dosyadaki 5. satır, veri kümesinin 5. yamasının özelliği olmalı.
    expected = FakeBackbone()(dataset[5][0][None]).to(torch.float16)
    assert np.allclose(features[5].numpy(), expected[0].numpy())
