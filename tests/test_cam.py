import torch

from pathoqc.cam import CamClassifier, cams_to_mask, predict_present


def test_classifier_output_shapes():
    model = CamClassifier(pretrained=False).eval()
    with torch.no_grad():
        logits, cams = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 4)
    assert cams.shape == (2, 4, 28, 28)  # 224 / 8
    assert torch.allclose(logits, cams.mean(dim=(2, 3)), atol=1e-5)


def test_predict_present_always_picks_at_least_one_class():
    logits = torch.tensor([[-5.0, -4.0, -6.0, -7.0], [3.0, -3.0, 2.0, -3.0]])
    assert predict_present(logits).tolist() == [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]]


def test_cams_to_mask_never_predicts_absent_class():
    cams = torch.zeros(1, 4, 2, 2)
    cams[0, 0, 0, :] = 1.0  # sınıf 0 üst satırda güçlü
    cams[0, 2, 1, :] = 3.0  # sınıf 2 alt satırda güçlü
    cams[0, 3] = 9.0        # sınıf 3 her yerde çok güçlü ama yamada yok
    present = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    mask = cams_to_mask(cams, present, size=(2, 2))
    assert mask.tolist() == [[[0, 0], [2, 2]]]


def test_peaky_map_loses_under_normalization_but_not_with_raw_scores():
    # Sınıf 2 her yerde sınıf 0'dan güçlü (3 > 2), üstelik bir pikselde sivrilmiş (30).
    cams = torch.zeros(1, 4, 2, 2)
    cams[0, 0] = 2.0
    cams[0, 0, 1, 1] = 1.0
    cams[0, 2] = 3.0
    cams[0, 2, 1, 1] = 30.0
    present = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

    # Ham skor: sınıf 2 her pikselde kazanır.
    assert cams_to_mask(cams, present, size=(2, 2), normalize=False).tolist() == [[[2, 2], [2, 2]]]
    # Klasik yöntem: sınıf 2 kendi sivri değerine (30) bölününce her yerde 0.1'e düşer ve
    # sivri piksel dışında sınıf 0'a kaybeder. Eğitim uzadıkça maskelerin bozulmasının nedeni budur.
    assert cams_to_mask(cams, present, size=(2, 2), normalize=True).tolist() == [[[0, 0], [0, 2]]]


def test_cams_to_mask_raw_never_predicts_absent_class():
    cams = torch.full((1, 4, 2, 2), -3.0)
    cams[0, 3] = 50.0  # yamada yok
    present = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    cams[0, 1, 0, 0] = -1.0
    assert cams_to_mask(cams, present, size=(2, 2), normalize=False).tolist() == [[[1, 0], [0, 0]]]
