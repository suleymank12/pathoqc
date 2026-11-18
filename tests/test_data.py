import pytest
import torch

from pathoqc.data import MaskDataset, PatchLabelDataset, parse_label


def test_parse_label_luad_format():
    assert parse_label("1031280-10364-10000-[1 0 0 1].png") == (1, 0, 0, 1)


def test_parse_label_bcss_format():
    assert parse_label("TCGA-A1-A0SK-DX1_xmin45749_ymin25055_MPP-0.2500+0[1101].png") == (1, 1, 0, 1)


@pytest.mark.parametrize("name", ["etiketsiz.png", "eksik-[1 0 1].png", "fazla-[10101].png"])
def test_parse_label_rejects_bad_names(name):
    with pytest.raises(ValueError):
        parse_label(name)


def test_patch_label_dataset_skips_mac_junk(tiny_dataset):
    dataset = PatchLabelDataset(tiny_dataset / "LUAD-HistoSeg" / "training")
    assert len(dataset) == 8
    image, label, name = dataset[0]
    assert image.shape == (3, 64, 64)
    assert label.tolist() == [1.0, 0.0, 0.0, 1.0]
    assert name == "p0-[1 0 0 1]"
    assert dataset.labels.sum(dim=0).tolist() == [5.0, 2.0, 2.0, 5.0]


def test_mask_dataset_returns_class_indices(tiny_dataset):
    dataset = MaskDataset(tiny_dataset / "LUAD-HistoSeg" / "val")
    assert len(dataset) == 4
    image, mask, _ = dataset[0]
    assert image.shape == (3, 64, 64)
    assert mask.shape == (64, 64) and mask.dtype == torch.int64
    assert 0 <= mask.min() and mask.max() <= 4


def test_mask_dataset_fails_when_mask_missing(tiny_dataset):
    (tiny_dataset / "LUAD-HistoSeg" / "val" / "mask" / "v0.png").rename(tiny_dataset / "v0.png")
    with pytest.raises(FileNotFoundError):
        MaskDataset(tiny_dataset / "LUAD-HistoSeg" / "val")
