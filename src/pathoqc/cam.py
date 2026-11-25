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


class CamHead(nn.Module):
    """Dondurulmuş omurganın özellik haritası üstüne eğitilen küçük baş.

    Girdi  [B, C, h, w]  önbellekten okunan özellikler
    Çıktı  logits [B, 4] ve cams [B, 4, h, w]; CamClassifier ile aynı düzen.
    """

    def __init__(self, in_channels: int, num_classes: int = 4):
        super().__init__()
        # Her omurganın özellik ölçeği farklıdır. BatchNorm hepsini aynı ölçeğe getirir;
        # böylece aynı öğrenme hızı bütün omurgalarda çalışır ve karşılaştırma adil olur.
        self.norm = nn.BatchNorm2d(in_channels)
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cams = self.classifier(self.norm(features))
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


def cams_to_mask(
    cams: torch.Tensor, present: torch.Tensor, size: tuple[int, int], normalize: bool = True
) -> torch.Tensor:
    """Isı haritalarını maskeye çevirir: [B, Y, G], her piksel 0-3 arası bir sınıf.

    Harita önce yama boyutuna büyütülür, sonra her pikselde en yüksek skorlu sınıf seçilir.
    Yamada olmayan sınıflar yarışa hiç giremez. Skorun nasıl hazırlandığı `normalize` ile seçilir:

    normalize=True  (klasik CAM): negatifler sıfırlanır, her sınıfın haritası kendi en büyük
                    değerine bölünür. Sınıflar aynı ölçekte yarışır; ama bir sınıfın haritası
                    tek bir noktada sivrilmişse geri kalan her yerde zayıf görünür.
    normalize=False (ham skor): başın verdiği skorlar olduğu gibi karşılaştırılır.
    """
    cams = F.interpolate(cams, size=size, mode="bilinear", align_corners=False)
    absent = present[:, :, None, None] == 0
    if normalize:
        cams = F.relu(cams)
        peak = cams.amax(dim=(2, 3), keepdim=True).clamp_min(1e-5)
        cams = (cams / peak).masked_fill(absent, 0.0)
    else:
        cams = cams.masked_fill(absent, float("-inf"))
    return cams.argmax(dim=1)
