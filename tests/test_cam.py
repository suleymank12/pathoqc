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
