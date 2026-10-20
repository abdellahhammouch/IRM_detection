from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.utils import ROOT


def select_images(manifest: pd.DataFrame, split: str, task: str = "classification",
                  train_dedup: str = "representative") -> pd.DataFrame:
    """Liste des images d'un split pour une tâche.

    - validation et test : toujours un représentant par groupe ;
    - entraînement : un représentant par groupe ('representative') ou toutes les copies ('all').
    """
    if train_dedup not in ("representative", "all"):
        raise ValueError("train_dedup doit valoir 'representative' ou 'all'")
    if task not in ("classification", "detection"):
        raise ValueError("task doit valoir 'classification' ou 'detection'")
    flag = "use_for_classification" if task == "classification" else "use_for_detection"
    df = manifest[manifest[flag] & (manifest["split"] == split)]
    if split != "train" or train_dedup == "representative":
        df = df[df["is_representative"]]
    return df.reset_index(drop=True)


def pil_to_gray(im) -> np.ndarray:
    """Image PIL -> niveaux de gris uint8 (H x W). Les PNG transparents sont posés sur fond noir."""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        black = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        im = Image.alpha_composite(black, rgba)
    return np.asarray(im.convert("L"), dtype=np.uint8)


def load_gray(path) -> np.ndarray:
    """Charge une image en niveaux de gris (tableau uint8, hauteur x largeur).

    Les PNG transparents (RGBA, ou palette avec transparence) sont posés sur fond noir."""
    with Image.open(path) as im:
        return pil_to_gray(im)


def border_median(gray: np.ndarray) -> int:
    """Ancienne estimation du fond : médiane des pixels du bord (1 pixel). Gardée pour comparaison."""
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    return int(np.median(border))


def estimate_background(gray: np.ndarray) -> int:
    """Niveau de gris du fond : 25e percentile d'un liseré de 3 % (au moins 2 pixels) le long des 4 bords.

    Le fond est la zone la plus sombre ; un peu de tissu qui touche un bord ne fausse pas le résultat."""
    h, w = gray.shape
    k = max(2, int(0.03 * min(h, w)))
    strips = np.concatenate([gray[:k, :].ravel(), gray[-k:, :].ravel(),
                             gray[:, :k].ravel(), gray[:, -k:].ravel()])
    return int(np.percentile(strips, 25))


def pad_to_square(gray: np.ndarray, pad_mode: str = "background") -> np.ndarray:
    """Ajoute des bandes pour rendre l'image carrée, sans déformer le cerveau.

    'background' : bandes de la couleur du fond (voir estimate_background) ; 'black' : noires."""
    h, w = gray.shape
    if h == w:
        return gray
    if pad_mode == "background":
        fill = estimate_background(gray)
    elif pad_mode == "black":
        fill = 0
    else:
        raise ValueError("pad_mode doit valoir 'background' ou 'black'")
    side = max(h, w)
    canvas = np.full((side, side), fill, dtype=np.uint8)
    top, left = (side - h) // 2, (side - w) // 2
    canvas[top:top + h, left:left + w] = gray
    return canvas


def preprocess_array(gray: np.ndarray, size: int = 224, pad_mode: str = "background") -> np.ndarray:
    """Tableau gris (H x W) -> tableau (size, size, 3) uint8 : carré, redimensionné, gris copié sur 3 canaux.

    Même traitement pour l'entraînement et pour l'application (images téléversées)."""
    gray = pad_to_square(gray, pad_mode)
    small = Image.fromarray(gray).resize((size, size), Image.BILINEAR)
    return np.repeat(np.asarray(small, dtype=np.uint8)[:, :, None], 3, axis=2)


def preprocess_image(rel_path, size: int = 224, pad_mode: str = "background") -> np.ndarray:
    """Chemin relatif -> tableau (size, size, 3) uint8 : gris copié sur 3 canaux, carré, redimensionné.

    Les valeurs restent entre 0 et 255 : la normalisation sera faite dans le modèle."""
    return preprocess_array(load_gray(ROOT / rel_path), size, pad_mode)


def plot_before_after(df: pd.DataFrame, size: int = 224, pad_mode: str = "background", out_path=None):
    """Ligne du haut : image d'origine ; ligne du bas : image prétraitée."""
    n = len(df)
    fig, axes = plt.subplots(2, n, figsize=(2.8 * n, 6))
    axes = np.array(axes).reshape(2, n)
    for k, (_, r) in enumerate(df.iterrows()):
        gray = load_gray(ROOT / r["image_path"])
        axes[0, k].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[0, k].set_title(f"{r['class']} | {Path(r['image_path']).stem}\n"
                             f"{gray.shape[1]}x{gray.shape[0]}, mode {r['mode']}", fontsize=8)
        axes[1, k].imshow(preprocess_image(r["image_path"], size, pad_mode))
        axes[1, k].set_title(f"{size}x{size}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
    plt.show()



def band_step(gray: np.ndarray, lo: float = 0.08, hi: float = 0.20):
    """Pour une image qui va recevoir des bandes : (couleur des bandes, fond proche du bord, écart).

    Le fond proche = 25e percentile de deux zones situées de 8 % à 20 % de la largeur (ou de la hauteur)
    sur les côtés où les bandes seront ajoutées. Un grand écart = bandes visibles à côté de l'image."""
    h, w = gray.shape
    if h == w:
        return None
    fill = estimate_background(gray)
    if w < h:
        a = int(lo * w); b = max(int(hi * w), a + 1)
        zone = np.concatenate([gray[:, a:b].ravel(), gray[:, w - b:w - a].ravel()])
    else:
        a = int(lo * h); b = max(int(hi * h), a + 1)
        zone = np.concatenate([gray[a:b, :].ravel(), gray[h - b:h - a, :].ravel()])
    near = int(np.percentile(zone, 25))
    return fill, near, abs(fill - near)