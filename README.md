# pathoqc

Histopatoloji (doku) görüntülerinde zayıf denetimli doku segmentasyonu.

"Zayıf denetimli" şu demek: model eğitilirken piksel piksel boyanmış maske kullanılmaz.
Yalnızca "bu yamada hangi dokular var" bilgisi verilir. Maskeler sadece sonucu ölçmek içindir.

Bu yazılım araştırma amaçlıdır. **Tanı koymaz**, klinik kararda kullanılamaz.

## Durum

- [x] Veri setleri indirildi ve doğrulandı (LUAD-HistoSeg, BCSS-WSSS)
- [x] Veri okuma, ölçüm ve temel model kodu, testleriyle birlikte
- [ ] Temel modelin Kaggle'da ilk eğitimi
- [ ] Patoloji temel modelleriyle (Hibou-L, H-optimus-0, Midnight) karşılaştırma
- [ ] Kalite kontrol modülü, servis ve arayüz

## Klasörler

| Yer | Ne | Neden burada |
|---|---|---|
| `src/pathoqc/data.py` | Veri setlerini diskten okur | Etiket biçimi ve klasör düzeni tek yerde dursun |
| `src/pathoqc/metrics.py` | IoU, mIoU, FwIoU, Dice hesaplar | Her deney aynı hesapla ölçülsün |
| `src/pathoqc/cam.py` | Temel model ve ısı haritasını maskeye çeviren kod | Model tanımı eğitim betiğinden ayrı dursun |
| `scripts/train_stage1.py` | Eğitimi başlatan betik | Kaggle ve Colab'da komutla çalıştırılır |
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
