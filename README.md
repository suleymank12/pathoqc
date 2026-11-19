# pathoqc

Histopatoloji (doku) görüntülerinde zayıf denetimli doku segmentasyonu.

"Zayıf denetimli" şu demek: model eğitilirken piksel piksel boyanmış maske kullanılmaz.
Yalnızca "bu yamada hangi dokular var" bilgisi verilir. Maskeler sadece sonucu ölçmek içindir.

Bu yazılım araştırma amaçlıdır. **Tanı koymaz**, klinik kararda kullanılamaz.

## Durum

- [x] Veri setleri indirildi ve doğrulandı (LUAD-HistoSeg, BCSS-WSSS)
- [x] Veri okuma, ölçüm ve temel model kodu, testleriyle birlikte
- [x] Temel modelin (ResNet-50) iki veri setinde ilk eğitimi
- [x] Dondurulmuş patoloji omurgası ve özellik önbelleği kodu
- [ ] Patoloji temel modelleriyle (Midnight, Hibou-L, H-optimus-0) karşılaştırma
- [ ] Aşama 2: kaba maskelerle segmentasyon modeli
- [ ] Kalite kontrol modülü, servis ve arayüz

## Sonuçlar (aşama 1, doğrulama kümesi, kaba maske kalitesi)

| Omurga | LUAD-HistoSeg mIoU | BCSS-WSSS mIoU | Not |
|---|---|---|---|
| ResNet-50 (ImageNet, ince ayarlı) | 69,36 | 60,43 | tek seed (0), 10 epoch |

Test kümelerine henüz dokunulmadı.

## Hedef

Asgari çizgi: LUAD-HistoSeg'de %77,9, BCSS-WSSS'te %71,4 (test, iki aşamanın sonunda).
Asıl hedef, yayımlanmış en yüksek sonuçların üstü: LUAD-HistoSeg'de %79,6, BCSS-WSSS'te %73,2.

## Klasörler

| Yer | Ne | Neden burada |
|---|---|---|
| `src/pathoqc/data.py` | Veri setlerini diskten okur | Etiket biçimi ve klasör düzeni tek yerde dursun |
| `src/pathoqc/metrics.py` | IoU, mIoU, FwIoU, Dice hesaplar | Her deney aynı hesapla ölçülsün |
| `src/pathoqc/cam.py` | Temel model, küçük baş ve ısı haritasını maskeye çeviren kod | Model tanımı eğitim betiğinden ayrı dursun |
| `src/pathoqc/backbones.py` | Dondurulmuş patoloji omurgalarının listesi ve sarmalayıcısı | Yeni omurga eklemek tek satır olsun |
| `src/pathoqc/cache.py` | Özellikleri bir kez hesaplayıp diske yazan kod | Dev model her eğitimde yeniden çalışmasın |
| `scripts/train_stage1.py` | Temel modeli (ResNet-50) eğiten betik | Kaggle ve Colab'da komutla çalıştırılır |
| `scripts/train_stage1_frozen.py` | Dondurulmuş omurga ile eğiten betik (3 seed) | Omurga karşılaştırması bununla yapılır |
| `tests/` | Otomatik testler | Bir şeyi değiştirince bozulanı hemen görmek için |
| `notebooks/` | Veriyi indiren iki not defteri | Colab ve Kaggle'da bir kez çalıştırıldı |
| `docs/veri-setleri.md` | Veri setlerinin sayıları, sınıfları, kaynağı | Rakamlar kodun dışında da yazılı dursun |

## Kurulum ve test

```
pip install -e ".[dev]"
pytest
```

## Deney kuralları

1. Model seçimi yalnızca doğrulama (val) kümesiyle yapılır.
2. Test kümesi bütün deneyler bittikten sonra bir kez ölçülür. `train_stage1.py` test klasörünü hiç açmaz.
3. Her deney 3 farklı seed ile tekrarlanır; ortalama ve sapma raporlanır.
4. Tek model sonucu ile birleştirilmiş (ensemble) sonuç ayrı tablolarda verilir.
5. Maskede değeri 4 olan arka plan pikselleri ölçüme katılmaz (yayındaki protokol).
