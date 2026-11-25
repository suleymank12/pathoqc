import pytest
import torch

from pathoqc.data import MaskDataset, PatchLabelDataset, build_eval_transform, evenly_spaced, parse_label


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


def test_evenly_spaced_covers_whole_list():
    assert evenly_spaced(list(range(10)), 5) == [0, 2, 4, 6, 8]
    assert evenly_spaced(list(range(10)), 3) == [0, 3, 6]
    assert evenly_spaced(list(range(4)), None) == [0, 1, 2, 3]
    assert evenly_spaced(list(range(4)), 9) == [0, 1, 2, 3]


def test_limit_takes_patches_from_across_the_folder(tiny_dataset):
    dataset = PatchLabelDataset(tiny_dataset / "LUAD-HistoSeg" / "training", limit=4)
    assert [path.stem.split("-")[0] for path in dataset.paths] == ["p0", "p2", "p4", "p6"]
    assert dataset.labels.shape == (4, 4)


def test_eval_transform_can_enlarge_the_patch(tiny_dataset):
    transform = build_eval_transform((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), image_size=128)
    dataset = PatchLabelDataset(tiny_dataset / "LUAD-HistoSeg" / "training", transform)
    assert dataset[0][0].shape == (3, 128, 128)
    # Maske büyütülmez: ölçüm her zaman özgün boyutta yapılır.
    masks = MaskDataset(tiny_dataset / "LUAD-HistoSeg" / "val", transform)
    image, mask, _ = masks[0]
    assert image.shape == (3, 128, 128) and mask.shape == (64, 64)
