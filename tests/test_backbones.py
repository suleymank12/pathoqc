import pytest
import torch
from transformers import Dinov2Config, Dinov2Model, Dinov2WithRegistersConfig, Dinov2WithRegistersModel

from pathoqc.backbones import FROZEN_BACKBONES, ViTFeatures

TINY = dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, patch_size=14, image_size=56)


def test_vit_features_shape_and_content():
    model = Dinov2Model(Dinov2Config(**TINY))
    features = ViTFeatures(model)
    images = torch.randn(2, 3, 56, 56)
    grid = features(images)
    assert grid.shape == (2, 32, 4, 4)  # 56 / 14 = 4

    # Izgaranın (satır 1, sütun 2) hücresi, dizideki ilgili kare vektörüyle aynı olmalı.
    tokens = model(pixel_values=images).last_hidden_state  # [CLS, 16 kare]
    assert torch.allclose(grid[:, :, 1, 2], tokens[:, 1 + 1 * 4 + 2, :], atol=1e-6)


def test_vit_features_skips_register_tokens():
    model = Dinov2WithRegistersModel(Dinov2WithRegistersConfig(num_register_tokens=4, **TINY))
    features = ViTFeatures(model)
    images = torch.randn(1, 3, 56, 56)
    grid = features(images)
    tokens = model(pixel_values=images).last_hidden_state  # [CLS, 4 register, 16 kare]
    assert tokens.shape[1] == 1 + 4 + 16
    assert torch.allclose(grid[:, :, 0, 0], tokens[:, 5, :], atol=1e-6)


def test_backbone_stays_frozen_and_in_eval_mode():
    features = ViTFeatures(Dinov2Model(Dinov2Config(**TINY)))
    features.train()
    assert not features.training and not features.model.training
    assert all(not p.requires_grad for p in features.parameters())


@pytest.mark.parametrize("name", sorted(FROZEN_BACKBONES))
def test_registry_entries_are_complete(name):
    spec = FROZEN_BACKBONES[name]
    assert "/" in spec.model_id and len(spec.mean) == 3 and len(spec.std) == 3 and spec.license
