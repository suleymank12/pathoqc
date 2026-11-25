"""Uçtan uca duman testi: minik bir ViT ile özellik çıkarma + ayar taraması + önbelleğin yeniden kullanımı."""

import json

import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

import train_stage1_frozen
from pathoqc import backbones


@pytest.fixture
def tiny_backbone(monkeypatch):
    spec = backbones.FrozenBackbone(model_id="test/tiny", mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), license="test")
    monkeypatch.setitem(backbones.FROZEN_BACKBONES, "tiny", spec)
    calls = []

    def fake_loader(name):
        calls.append(name)
        torch.manual_seed(0)  # minik omurga her yüklemede aynı ağırlıklarla kurulsun
        config = Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, patch_size=14, image_size=56)
        return backbones.ViTFeatures(Dinov2Model(config))

    monkeypatch.setattr(backbones, "load_frozen_backbone", fake_loader)
    return calls


def test_sweep_selects_one_setting_for_all_seeds(tiny_dataset, tmp_path, tiny_backbone):
    arguments = ["--dataset", "luad", "--backbone", "tiny", "--data-root", str(tiny_dataset),
                 "--cache-dir", str(tmp_path / "cache"), "--out-dir", str(tmp_path / "out"),
                 "--lrs", "0.001", "0.01", "--max-steps", "5", "--eval-every", "2",
                 "--batch-size", "4", "--workers", "0", "--seeds", "0", "1"]

    first = train_stage1_frozen.main(arguments)
    assert tiny_backbone == ["tiny"]  # omurga bir kez yüklendi

    # 2 öğrenme hızı x 2 seed; ölçüm noktaları 2., 4. ve son (5.) adım.
    assert set(first["curves"]) == {"0.001", "0.01"}
    assert [p["step"] for p in first["curves"]["0.001"]["0"]] == [2, 4, 5]

    selected = first["selected"]
    assert selected["lr"] in (0.001, 0.01) and selected["variant"] in ("normalized", "raw")
    assert selected["step"] in (2, 4, 5)
    assert len(selected["val_miou_per_seed"]) == 2 and selected["val_miou_std"] is not None

    # Seçilen sayı, eğrilerdeki ilgili noktaların ortalamasıyla aynı olmalı ve hiçbir ayar onu geçmemeli.
    def mean_at(lr, variant, step):
        values = [next(p[variant] for p in first["curves"][lr][s] if p["step"] == step) for s in ("0", "1")]
        return sum(values) / 2

    assert selected["val_miou_mean"] == pytest.approx(mean_at(str(selected["lr"]), selected["variant"], selected["step"]), abs=0.01)
    for lr in first["curves"]:
        for variant in ("normalized", "raw"):
            for step in (2, 4, 5):
                assert mean_at(lr, variant, step) <= selected["val_miou_mean"] + 0.01

    run_dir = tmp_path / "out" / "luad_stage1_tiny_224px"
    saved = torch.load(run_dir / "best_seed0.pt", weights_only=False)
    assert saved["step"] == selected["step"] and "classifier.weight" in saved["head"]

    # İkinci çalıştırma önbelleği kullanmalı (omurga yeniden yüklenmez) ve aynı sonucu vermeli.
    second = train_stage1_frozen.main(arguments)
    assert tiny_backbone == ["tiny"]
    assert second["extraction"] is None
    assert second["curves"] == first["curves"]
    assert json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["selected"] == second["selected"]


def test_larger_image_size_gives_finer_feature_map(tiny_dataset, tmp_path, tiny_backbone):
    common = ["--dataset", "luad", "--backbone", "tiny", "--data-root", str(tiny_dataset),
              "--cache-dir", str(tmp_path / "cache"), "--out-dir", str(tmp_path / "out"),
              "--lrs", "0.001", "--max-steps", "2", "--eval-every", "1",
              "--batch-size", "4", "--extract-batch-size", "2", "--workers", "0", "--seeds", "0"]

    train_stage1_frozen.main(common + ["--image-size", "56"])
    train_stage1_frozen.main(common + ["--image-size", "112"])

    # Kare boyu 14 piksel: 56 piksel -> 4x4 kare, 112 piksel -> 8x8 kare. İki önbellek ayrı dosyalarda durur.
    import numpy as np
    small = np.load(tmp_path / "cache" / "luad_tiny_56px_training.npy")
    large = np.load(tmp_path / "cache" / "luad_tiny_112px_training.npy")
    assert small.shape == (8, 32, 4, 4) and large.shape == (8, 32, 8, 8)
    assert (tmp_path / "out" / "luad_stage1_tiny_112px" / "summary.json").is_file()
