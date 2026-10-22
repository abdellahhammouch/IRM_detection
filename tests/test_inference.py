"""Tests de l'inférence utilisée par l'application (Jour 8)."""
import numpy as np
import pytest
from PIL import Image

from src.inference import DET_NAMES, classify, detect, draw_boxes
from src.preprocessing import load_gray, preprocess_array, preprocess_image

CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


def test_upload_and_file_preprocessing_are_identical(tmp_path):
    g = np.random.RandomState(0).randint(0, 255, (90, 120), dtype=np.uint8)
    p = tmp_path / "x.png"
    Image.fromarray(g).save(p)
    assert np.array_equal(preprocess_image(p), preprocess_array(load_gray(p)))


class FakeClassifier:
    def predict(self, x, verbose=0):
        assert x.shape == (1, 224, 224, 3) and x.dtype == np.uint8
        return np.array([[0.1, 0.6, 0.05, 0.25]])


def test_classify_returns_label_and_probabilities_summing_to_one():
    label, probs = classify(FakeClassifier(), Image.new("L", (100, 80), 50), CLASSES)
    assert label == "Meningioma" and list(probs) == CLASSES and sum(probs.values()) == pytest.approx(1.0)


class T:
    def __init__(self, a): self.a = np.array(a, dtype=float)
    def cpu(self): return self
    def numpy(self): return self.a


class FakeDetector:
    def predict(self, source, conf, imgsz, verbose):
        class B: cls, conf, xyxy = T([0, 2]), T([0.6, 0.9]), T([[1, 2, 30, 40], [5, 5, 20, 20]])
        class R: boxes = B()
        return [R()]


def test_detect_names_and_sorts_boxes_by_confidence():
    boxes = detect(FakeDetector(), Image.new("L", (64, 64)), conf=0.5)
    assert [b[0] for b in boxes] == ["Pituitary", "Glioma"] and boxes[0][1] == pytest.approx(0.9)
    assert set(DET_NAMES.values()) == {"Glioma", "Meningioma", "Pituitary"}


def test_draw_boxes_keeps_size_and_does_not_modify_input():
    img = Image.new("L", (80, 60), 0)
    out = draw_boxes(img, [("Pituitary", 0.9, (5, 5, 30, 30))])
    assert out.size == img.size and img.getextrema() == (0, 0) and out.getextrema() != ((0, 0),) * 3