"""Tests des transformations géométriques appliquées à l'image et à ses boîtes (étape 2.6)."""
import numpy as np
import pytest

from src.augmentation import augment_with_boxes, sample_affine, transform_boxes, warp_image

IDENT = {"angle": 0.0, "tx": 0.0, "ty": 0.0, "scale": 1.0}


def make_image(w=341, h=395, bg=10, box=(0.6, 0.35, 0.10, 0.08)):
    """Image de fond `bg` avec un carré clair ; retourne (image, boîte YOLO du carré)."""
    img = np.full((h, w), bg, dtype=np.uint8)
    cx, cy, bw, bh = box
    x0, x1 = int((cx - bw / 2) * w), int((cx + bw / 2) * w)
    y0, y1 = int((cy - bh / 2) * h), int((cy + bh / 2) * h)
    img[y0:y1, x0:x1] = 220
    return img, [(1, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)]


def bright_bbox(img, level=120):
    ys, xs = np.nonzero(img > level)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def test_identity_keeps_everything():
    img, boxes = make_image()
    out = transform_boxes(boxes, IDENT, 341, 395)
    assert np.allclose(out[0][1:], boxes[0][1:])
    assert np.array_equal(warp_image(img, IDENT), img)


def test_translation_only_shifts_box_exactly():
    img, boxes = make_image()
    p = {"angle": 0.0, "tx": 0.05, "ty": -0.03, "scale": 1.0}
    out = transform_boxes(boxes, p, 341, 395)[0]
    assert out[1] == pytest.approx(boxes[0][1] + 0.05) and out[2] == pytest.approx(boxes[0][2] - 0.03)
    assert out[3] == pytest.approx(boxes[0][3]) and out[4] == pytest.approx(boxes[0][4])


@pytest.mark.parametrize("seed", range(40))
def test_box_still_contains_the_lesion(seed):
    """Sur une image non carrée : la tache claire reste à l'intérieur de la boîte transformée."""
    img, boxes = make_image()
    out_img, out_boxes, _ = augment_with_boxes(img, boxes, np.random.RandomState(seed))
    assert len(out_boxes) == 1                           # aucune boîte perdue avec les réglages de base
    h, w = out_img.shape
    _, cx, cy, bw, bh = out_boxes[0]
    bx0, by0, bx1, by1 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
    x0, y0, x1, y1 = bright_bbox(out_img)
    tol = 2.0                                           # un peu de flou sur les bords (interpolation)
    assert bx0 - tol <= x0 and by0 - tol <= y0 and x1 <= bx1 + tol and y1 <= by1 + tol


def test_class_id_and_range_preserved():
    img, boxes = make_image()
    for seed in range(20):
        _, out, _ = augment_with_boxes(img, boxes, np.random.RandomState(seed))
        for c, cx, cy, w, h in out:
            assert c == 1 and 0 <= cx - w / 2 and cx + w / 2 <= 1 + 1e-9 and 0 <= cy - h / 2 and cy + h / 2 <= 1 + 1e-9


def test_box_leaving_the_image_is_dropped():
    boxes = [(0, 0.95, 0.5, 0.08, 0.08)]
    p = {"angle": 0.0, "tx": 0.10, "ty": 0.0, "scale": 1.0}   # décalage de 10 % vers la droite
    assert transform_boxes(boxes, p, 400, 400) == []


def test_half_visible_threshold_and_clipping():
    boxes = [(2, 0.9, 0.5, 0.2, 0.2)]                          # va de 0,8 à 1,0
    p = {"angle": 0.0, "tx": 0.06, "ty": 0.0, "scale": 1.0}    # 0,86 à 1,06 : 70 % visible
    out = transform_boxes(boxes, p, 400, 400)
    assert len(out) == 1 and out[0][1] + out[0][3] / 2 == pytest.approx(1.0)
    p2 = {"angle": 0.0, "tx": 0.12, "ty": 0.0, "scale": 1.0}   # 0,92 à 1,12 : 40 % visible
    assert transform_boxes(boxes, p2, 400, 400) == []
    assert len(transform_boxes(boxes, p2, 400, 400, min_visible=0.3)) == 1


def test_corners_get_background_color():
    img, _ = make_image(bg=37)
    out = warp_image(img, {"angle": 15.0, "tx": 0.1, "ty": 0.1, "scale": 0.85})
    assert out[0, 0] == 37 and out[-1, -1] == 37 and out[0, -1] == 37


def test_rgb_image_supported_and_reproducible():
    img, boxes = make_image()
    rgb = np.repeat(img[:, :, None], 3, axis=2)
    a = augment_with_boxes(rgb, boxes, np.random.RandomState(7))
    b = augment_with_boxes(rgb, boxes, np.random.RandomState(7))
    assert a[0].shape == rgb.shape and np.array_equal(a[0], b[0]) and a[1] == b[1]
    assert (a[0][..., 0] == a[0][..., 1]).all()