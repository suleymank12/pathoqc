"""Temel model: sınıflandırıcı + sınıf aktivasyon haritası (CAM).

Fikir: Modele yalnızca "bu yamada hangi dokular var" sorusunu öğretiriz. Model bu soruyu
cevaplarken yamanın hangi bölgesine baktığını bir ısı haritası olarak verir. O haritayı
kaba bir maskeye çeviririz. Piksel etiketi hiç kullanılmaz; "zayıf denetimli" budur.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class CamClassifier(nn.Module):
    """ResNet-50 omurga + 1x1 konvolüsyon baş.

    Çıktı iki parçadır:
      cams   [B, 4, h, w]  her sınıf için ısı haritası
      logits [B, 4]        haritanın ortalaması; "bu sınıf yamada var mı" skoru
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        # Son iki aşamada küçültme kapatılır (dilation). 224 piksellik yamada harita 7x7 yerine
        # 28x28 çıkar; doku sınırları daha keskin görünür.
        net = models.resnet50(weights=weights, replace_stride_with_dilation=[False, True, True])
        self.features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3, net.layer4
        )
        # 1x1 konvolüsyon: her konumdaki 2048 özelliği 4 sınıf skoruna çevirir.
        self.classifier = nn.Conv2d(2048, num_classes, kernel_size=1, bias=False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cams = self.classifier(self.features(images))
        logits = cams.mean(dim=(2, 3))
        return logits, cams


def predict_present(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Skorlardan "yamada hangi sınıflar var" kararını üretir: [B, 4], değerler 0 ya da 1.

    Doğrulama ve testte yama etiketini bilmiyormuş gibi davranırız (maskeden okumak kopya olur),
    bu yüzden modelin kendi tahminini kullanırız. Hiçbiri eşiği geçmezse en yüksek skorlu sınıf seçilir.
    """
    probabilities = torch.sigmoid(logits)
    present = (probabilities > threshold).float()
    top = probabilities.argmax(dim=1)
    present[torch.arange(len(top), device=present.device), top] = 1.0
    return present


def cams_to_mask(cams: torch.Tensor, present: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Isı haritalarını maskeye çevirir: [B, Y, G], her piksel 0-3 arası bir sınıf.

    1. Negatif değerler sıfırlanır (o sınıfa karşı kanıt, işimize yaramaz).
    2. Harita yama boyutuna büyütülür.
    3. Her sınıfın haritası kendi en büyük değerine bölünür ki sınıflar aynı ölçekte yarışsın.
    4. Yamada olmayan sınıfların haritası sıfırlanır.
    5. Her pikselde en yüksek haritaya sahip sınıf seçilir.
    """
    cams = F.relu(cams)
    cams = F.interpolate(cams, size=size, mode="bilinear", align_corners=False)
    peak = cams.amax(dim=(2, 3), keepdim=True).clamp_min(1e-5)
    cams = cams / peak
    cams = cams * present[:, :, None, None]
    return cams.argmax(dim=1)
