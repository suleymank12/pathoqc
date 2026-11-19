"""Uçtan uca duman testi: minik bir ViT ile özellik çıkarma + 2 seed eğitim + önbelleğin yeniden kullanımı."""

import json

import pytest
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
        config = Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, patch_size=14, image_size=56)
        return backbones.ViTFeatures(Dinov2Model(config))

    monkeypatch.setattr(backbones, "load_frozen_backbone", fake_loader)
    return calls


def test_extract_once_then_train_two_seeds(tiny_dataset, tmp_path, tiny_backbone):
    arguments = ["--dataset", "luad", "--backbone", "tiny", "--data-root", str(tiny_dataset),
                 "--cache-dir", str(tmp_path / "cache"), "--out-dir", str(tmp_path / "out"),
                 "--epochs", "2", "--batch-size", "4", "--workers", "0", "--seeds", "0", "1"]

    first = train_stage1_frozen.main(arguments)
    assert tiny_backbone == ["tiny"]  # omurga bir kez yüklendi
    assert len(first["runs"]) == 2 and len(first["val_miou_per_seed"]) == 2
    assert first["val_miou_std"] is not None
    run_dir = tmp_path / "out" / "luad_stage1_tiny"
    assert (run_dir / "best_seed0.pt").is_file() and (run_dir / "best_seed1.pt").is_file()

    # İkinci çalıştırma önbelleği kullanmalı (omurga yeniden yüklenmez) ve aynı seed aynı sonucu vermeli.
    second = train_stage1_frozen.main(arguments)
    assert tiny_backbone == ["tiny"]
    assert second["extraction"] is None
    assert second["val_miou_per_seed"] == first["val_miou_per_seed"]
    saved = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert saved["val_miou_mean"] == second["val_miou_mean"]
