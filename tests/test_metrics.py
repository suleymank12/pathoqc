import numpy as np
import pytest

from pathoqc.metrics import SegmentationMeter


def test_hand_computed_example():
    # 4 piksel. Sağ alttaki arka plan (4) olduğu için sayılmaz; geriye 3 piksel kalır.
    target = np.array([[0, 0], [1, 4]])
    prediction = np.array([[0, 1], [1, 2]])
    meter = SegmentationMeter()
    meter.update(target, prediction)

    # Sınıf 0: 1 doğru; gerçekte 2, tahminde 1 piksel -> IoU = 1 / (2 + 1 - 1) = 0.5
    # Sınıf 1: 1 doğru; gerçekte 1, tahminde 2 piksel -> IoU = 1 / (1 + 2 - 1) = 0.5
    iou = meter.iou_per_class()
    assert iou[0] == pytest.approx(0.5) and iou[1] == pytest.approx(0.5)
    assert np.isnan(iou[2]) and np.isnan(iou[3])  # veride hiç yoklar
    assert meter.miou() == pytest.approx(0.5)
    assert meter.fwiou() == pytest.approx(0.5)
    assert meter.mdice() == pytest.approx(2 / 3)
    assert meter.pixel_accuracy() == pytest.approx(2 / 3)


def test_perfect_prediction_scores_one():
    target = np.array([[0, 1, 2, 3, 4]])
    meter = SegmentationMeter()
    meter.update(target, np.array([[0, 1, 2, 3, 0]]))
    assert meter.miou() == pytest.approx(1.0)
    assert meter.summary()["miou"] == 100.0


def test_updates_accumulate():
    one = SegmentationMeter()
    one.update(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    two = SegmentationMeter()
    two.update(np.array([0, 0]), np.array([0, 1]))
    two.update(np.array([1, 1]), np.array([1, 1]))
    assert (one.matrix == two.matrix).all()


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError):
        SegmentationMeter().update(np.zeros((2, 2)), np.zeros((2, 3)))
