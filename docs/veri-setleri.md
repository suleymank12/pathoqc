# Veri setleri

Kaynak: Han ve ark., "Multi-Layer Pseudo-Supervision for Histopathology Tissue Semantic Segmentation
using Patch-level Classification Labels" (Medical Image Analysis, 2022).
Resmî depo ve indirme bağlantıları: https://github.com/ChuHan89/WSSS-Tissue

19 Kasım 2026'da indirildi ve sayıldı.

## Sayılar

| Veri seti | eğitim | doğrulama | test | zip boyutu |
|---|---|---|---|---|
| LUAD-HistoSeg (akciğer) | 16.678 | 300 | 307 | 2.101.270.056 bayt |
| BCSS-WSSS (meme) | 23.422 | 3.418 | 4.986 | 3.097.415.608 bayt |

Bütün yamalar 224x224 piksel, RGB.

SHA-256 (dosyanın parmak izi; dosya değişirse bu da değişir):

- `LUAD-HistoSeg.zip`: `9c54b691f24dfe266268d03685476daefddecd57b468ce15161c0bbb38f86984`
- `BCSS-WSSS.zip`: `8cb54da91d60585c724ea765bfdd3ac9e70f4fdf96f5daaea865b2060ebed5d9`

## Sınıflar

Etiketteki ve maskedeki sıra aynıdır (0, 1, 2, 3).

| Sıra | LUAD-HistoSeg | Eğitimde kaç yamada var | BCSS-WSSS | Eğitimde kaç yamada var |
|---|---|---|---|---|
| 0 | TE (tümör epiteli) | 13.151 | TUM (tümör) | 15.232 |
| 1 | NEC (nekroz) | 2.628 | STR (stroma) | 16.254 |
| 2 | LYM (lenfosit) | 697 | LYM (lenfosit) | 5.598 |
| 3 | TAS (tümörle ilişkili stroma) | 13.443 | NEC (nekroz) | 2.884 |

Sınıflar dengesizdir: LUAD'da lenfosit yamaların yaklaşık %4'ünde, BCSS'te nekroz yaklaşık %12'sinde bulunur.

## Etiket ve maske biçimi

- Eğitim yamalarında etiket dosya adındadır. LUAD: `ad-[1 0 0 1].png`. BCSS: `ad[1101].png`.
- Doğrulama ve test maskeleri paletli PNG'dir; her piksel sınıf numarası tutar.
- Maskede 4 değeri arka plandır ve ölçüme katılmaz.
- LUAD zip'inden Mac'e ait çöp dosyalar (`._ad.png`, `__MACOSX/`) çıkar; kod bunları atlar.

## Veri nerede duruyor

Veri bu depoda yoktur. İki zip, Kaggle'daki özel `pathoqc-veri` not defterinin çıktısında durur:
https://www.kaggle.com/code/suleymankaraman/pathoqc-veri

## Bilinen sınırlamalar

- Bu iki türetilmiş veri seti için açık bir lisans beyanı bulunamadı. Kaynak BCSS verisi CC0 1.0'dır.
  Veri ve bu veriyle eğitilmiş ağırlıklar yeniden dağıtılmaz.
- Resmî eğitim/doğrulama/test ayrımının hasta düzeyinde ayrık olduğu birincil kaynaktan doğrulanamadı.
- LUAD doğrulama kümesi bazı makalelerde 306 yama olarak geçer; indirilen sürümde 300 yamadır.
