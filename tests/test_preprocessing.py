"""Tests du prétraitement (étape 2.8)."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.preprocessing import (estimate_background, load_gray, pad_to_square, preprocess_image,
                               select_images)


def save(tmp_path, name, arr, mode=None):
    p = tmp_path / name
    Image.fromarray(arr, mode).save(p)
    return p


def test_gray_and_rgb_images_load_as_2d_uint8(tmp_path):
    g = np.random.RandomState(0).randint(0, 255, (40, 30), dtype=np.uint8)
    rgb = np.repeat(g[:, :, None], 3, axis=2)
    for name, arr in (("g.png", g), ("c.png", rgb)):
        out = load_gray(save(tmp_path, name, arr))
        assert out.ndim == 2 and out.dtype == np.uint8 and out.shape == (40, 30)


def test_transparent_png_is_placed_on_black(tmp_path):
    rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    rgba[..., :3] = 200                      # gris clair
    rgba[:10, :, 3] = 255                    # moitié haute opaque, moitié basse transparente
    out = load_gray(save(tmp_path, "t.png", rgba))
    assert out[:10].mean() == pytest.approx(200, abs=1) and out[10:].max() == 0


def test_estimate_background_ignores_a_little_tissue_on_the_border():
    img = np.full((200, 200), 12, dtype=np.uint8)
    img[0, 50:70] = 255                      # un peu de tissu sur le bord
    img[80:120, 80:120] = 180                # centre clair
    assert estimate_background(img) == 12


def test_pad_to_square_is_square_and_keeps_pixels():
    img = np.full((100, 60), 12, dtype=np.uint8)
    img[40:60, 20:40] = 200
    out = pad_to_square(img, "background")
    assert out.shape == (100, 100)
    assert out[0, 0] == 12 and out[0, -1] == 12           # bandes de la couleur du fond
    assert np.array_equal(out[:, 20:80], img)             # l'image est intacte, centrée
    assert pad_to_square(img, "black")[0, 0] == 0
    sq = np.zeros((50, 50), dtype=np.uint8)
    assert pad_to_square(sq) is sq                        # déjà carrée : rien ne change
    with pytest.raises(ValueError):
        pad_to_square(img, "inconnu")


@pytest.mark.parametrize("shape", [(341, 395), (512, 512), (201, 207), (550, 664)])
def test_preprocess_image_output(tmp_path, shape):
    g = np.random.RandomState(1).randint(0, 255, shape, dtype=np.uint8)
    out = preprocess_image(save(tmp_path, "x.jpg", g), size=224)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    assert (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all()


def make_manifest():
    rows = []
    for split in ("train", "val", "test", "excluded"):
        for k in range(4):
            rows.append(dict(image_path=f"{split}{k}.jpg", split=split, is_representative=(k < 2),
                             use_for_classification=(split != "excluded"),
                             use_for_detection=(split != "excluded" and k != 3)))
    return pd.DataFrame(rows)


def test_select_images_rules():
    m = make_manifest()
    assert len(select_images(m, "val")) == 2 and select_images(m, "val")["is_representative"].all()
    assert len(select_images(m, "train", train_dedup="representative")) == 2
    assert len(select_images(m, "train", train_dedup="all")) == 4          # toutes les copies, train seulement
    assert len(select_images(m, "test", train_dedup="all")) == 2           # val/test : toujours un représentant
    assert len(select_images(m, "train", "detection", "all")) == 3
    assert len(select_images(m, "excluded")) == 0
    with pytest.raises(ValueError):
        select_images(m, "train", train_dedup="autre")
    with pytest.raises(ValueError):
        select_images(m, "train", task="autre")