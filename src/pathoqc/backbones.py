"""Dondurulmuş patoloji omurgaları (temel modeller).

"Omurga", görüntüden özellik çıkaran büyük modeldir. Buradaki omurgalar milyonlarca patoloji
görüntüsüyle önceden eğitilmiştir. Biz onları EĞİTMEYİZ (dondurulmuş): yalnızca özellik
çıkarmak için kullanırız, üstlerine küçük bir sınıflandırma başı eğitiriz.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FrozenBackbone:
    """Bir omurganın değişmeyen bilgileri."""

    model_id: str  # Hugging Face'teki adı
    mean: tuple[float, float, float]  # modelin beklediği renk ortalaması (model kartından)
    std: tuple[float, float, float]  # modelin beklediği renk sapması (model kartından)
    license: str


FROZEN_BACKBONES = {
    # Kaiko Midnight-12k: ViT-g/14, herkese açık, MIT lisanslı. Yalnızca TCGA ile ön eğitildi;
    # BCSS-WSSS de TCGA kaynaklı olduğu için o veri setindeki sonuçlar bu notla raporlanır.
    "midnight": FrozenBackbone(
        model_id="kaiko-ai/midnight", mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), license="MIT"
    ),
}


class ViTFeatures(nn.Module):
    """Bir ViT modelini "görüntü -> özellik haritası" haline getirir.

    ViT görüntüyü küçük karelere (patch) böler ve her kare için bir vektör üretir.
    224 piksellik yamada kare boyu 14 ise 16x16 = 256 kare çıkar.
    Çıktı: [B, C, 16, 16]; C her karenin vektör uzunluğudur.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)  # dondur: bu ağırlıklar hiç güncellenmez
        self.patch_size = int(model.config.patch_size)
        self.out_channels = int(model.config.hidden_size)

    def train(self, mode: bool = True):
        """Omurga her zaman ölçüm kipinde kalır; üst kod .train() çağırsa bile."""
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        rows = images.shape[-2] // self.patch_size
        cols = images.shape[-1] // self.patch_size
        tokens = self.model(pixel_values=images).last_hidden_state
        # Dizinin başında özet vektörleri (CLS, varsa register) durur; kare vektörleri hep sondadır.
        patches = tokens[:, -rows * cols :, :]
        return patches.transpose(1, 2).reshape(len(images), self.out_channels, rows, cols)


def load_frozen_backbone(name: str) -> ViTFeatures:
    """Omurgayı Hugging Face'ten indirir ve özellik çıkarıcıya sarar."""
    from transformers import AutoModel  # yalnızca gerektiğinde yüklenir; testlerde indirme yapılmaz

    spec = FROZEN_BACKBONES[name]
    return ViTFeatures(AutoModel.from_pretrained(spec.model_id))
