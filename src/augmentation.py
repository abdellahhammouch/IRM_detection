"""Pipeline d'entraînement : images en mémoire, augmentations (train seulement), tf.data."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from PIL import Image

from src.preprocessing import estimate_background, preprocess_image, select_images

DEFAULT_AUG = {
    "rotation_deg": 15, "translation": 0.10, "zoom": 0.15,
    "contrast": 0.20, "brightness": 0.15,
    "degrade_prob": 0.5, "min_scale": 0.5,
    "box_min_visible": 0.5,   # une boîte est écartée si moins de 50 % de sa surface reste dans l'image
}


def load_split(manifest, split, classes, size=224, pad_mode="background",
               train_dedup="representative"):
    """Prétraite toutes les images d'un split une seule fois.

    Retourne (X, y, df) : X (n, size, size, 3) uint8 ; y = indice de classe (ordre de `classes`) ;
    df = lignes du manifeste, dans le même ordre que X."""
    df = select_images(manifest, split, "classification", train_dedup)
    X = np.empty((len(df), size, size, 3), dtype=np.uint8)
    for k, p in enumerate(df["image_path"]):
        X[k] = preprocess_image(p, size, pad_mode)
    index = {c: i for i, c in enumerate(classes)}
    y = df["class"].map(index).to_numpy(dtype=np.int32)
    return X, y, df


def _blur(x):
    """Flou léger : moyenne pondérée 3x3 (poids 1-2-1). x : float32 (H, W, 3)."""
    k1 = tf.constant([1.0, 2.0, 1.0])
    k = k1[:, None] * k1[None, :]
    k = k / tf.reduce_sum(k)
    k = tf.tile(k[:, :, None, None], [1, 1, 3, 1])
    x4 = tf.pad(x[None], [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
    return tf.nn.depthwise_conv2d(x4, k, strides=[1, 1, 1, 1], padding="VALID")[0]


def _lower_resolution(x, size, min_scale, seed):
    """Réduit l'image à une taille tirée au hasard (de min_scale à 100 %), puis la remonte."""
    s = tf.random.uniform([], min_scale, 1.0, seed=seed)
    small = tf.cast(tf.round(size * s), tf.int32)
    y = tf.image.resize(x, [small, small], method="bilinear", antialias=True)
    return tf.image.resize(y, [size, size], method="bilinear")


def _background(x, size):
    """Niveau de gris du fond d'une image (float32, H x W x 3) : 25e percentile d'un liseré de 3 %
    (au moins 2 pixels) le long des 4 bords. Même idée que `estimate_background` du prétraitement."""
    k = max(2, int(0.03 * size))
    g = x[..., 0]
    strips = tf.concat([tf.reshape(g[:k, :], [-1]), tf.reshape(g[-k:, :], [-1]),
                        tf.reshape(g[:, :k], [-1]), tf.reshape(g[:, -k:], [-1])], axis=0)
    srt = tf.sort(strips)
    return srt[tf.shape(srt)[0] // 4]


def make_augmenter(aug_cfg=None, size=224, seed=42):
    """Retourne une fonction image uint8 -> image uint8 qui applique les augmentations.

    À utiliser sur le train seulement. Les 3 canaux restent identiques.
    Les coins vides laissés par rotation / décalage / zoom reçoivent la couleur du fond de l'image."""
    c = {**DEFAULT_AUG, **(aug_cfg or {})}
    geometry = keras.Sequential([
        keras.layers.RandomRotation(c["rotation_deg"] / 360.0, fill_mode="constant", seed=seed),
        keras.layers.RandomTranslation(c["translation"], c["translation"],
                                       fill_mode="constant", seed=seed + 1),
        keras.layers.RandomZoom((-c["zoom"], c["zoom"]), fill_mode="constant", seed=seed + 2),
    ])
    light = keras.Sequential([
        keras.layers.RandomContrast(c["contrast"], value_range=(0, 255), seed=seed + 3),
        keras.layers.RandomBrightness(c["brightness"], value_range=(0, 255), seed=seed + 4),
    ])

    @tf.autograph.experimental.do_not_convert
    def augment(img):
        x = tf.cast(img, tf.float32)
        bg = _background(x, size)
        x = geometry(x - bg, training=True) + bg   # coins vides = couleur du fond de l'image
        x = light(x, training=True)
        if c["degrade_prob"] > 0:
            go = tf.random.uniform([], seed=seed + 5) < c["degrade_prob"]
            x = tf.cond(go,
                        lambda: _lower_resolution(_blur(x), size, c["min_scale"], seed + 6),
                        lambda: x)
        x = tf.clip_by_value(x, 0.0, 255.0)
        x = tf.cast(tf.round(x), tf.uint8)
        x.set_shape([size, size, 3])
        return x

    return augment


# ---------------------------------------------------------------------------
# Étape 2.6 : transformations géométriques appliquées à l'image ET à ses boîtes
# ---------------------------------------------------------------------------
# Les couches Keras tirent leurs valeurs au hasard à l'intérieur de la couche : on ne peut pas
# les relire pour déplacer les boîtes. Ici on tire explicitement (angle, décalage, zoom) et on
# applique les MÊMES valeurs à l'image et aux boîtes. Une boîte est un tuple YOLO
# (classe, cx, cy, w, h), coordonnées normalisées (0 à 1) par la taille de l'image d'origine.

def sample_affine(rng, rotation_deg=15, translation=0.10, zoom=0.15):
    """Tire une transformation : angle (degrés), décalage (fraction de la taille), échelle."""
    return {
        "angle": float(rng.uniform(-rotation_deg, rotation_deg)),
        "tx": float(rng.uniform(-translation, translation)),
        "ty": float(rng.uniform(-translation, translation)),
        "scale": float(1.0 + rng.uniform(-zoom, zoom)),
    }


def _affine(params, w, h):
    """Retourne (A, centre, décalage en pixels) : point_sortie = A (point - centre) + centre + décalage."""
    a = np.deg2rad(params["angle"])
    A = params["scale"] * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return A, np.array([w / 2.0, h / 2.0]), np.array([params["tx"] * w, params["ty"] * h])


def warp_image(img, params, fill=None):
    """Applique la transformation à une image uint8 (H, W) ou (H, W, 3).

    Les coins vides prennent la couleur du fond de l'image (`fill`, estimée si absente)."""
    h, w = img.shape[:2]
    pil = Image.fromarray(img)
    if fill is None:
        fill = estimate_background(np.asarray(pil.convert("L")))
    A, c, t = _affine(params, w, h)
    inv = np.linalg.inv(A)                      # PIL veut la transformation sortie -> entrée
    off = c - inv @ (c + t)
    data = (inv[0, 0], inv[0, 1], off[0], inv[1, 0], inv[1, 1], off[1])
    color = int(fill) if img.ndim == 2 else (int(fill),) * img.shape[2]
    out = pil.transform((w, h), Image.AFFINE, data, resample=Image.BILINEAR, fillcolor=color)
    return np.asarray(out)


def transform_boxes(boxes, params, w, h, min_visible=0.5):
    """Déplace les boîtes avec la même transformation que l'image.

    Les 4 coins de chaque boîte sont transformés, puis on prend le plus petit rectangle droit
    qui les contient (après rotation, la boîte est donc un peu plus grande que la lésion).
    La boîte est ensuite recadrée au cadre de l'image ; elle est écartée si moins de
    `min_visible` de sa surface y reste."""
    A, c, t = _affine(params, w, h)
    out = []
    for cls, cx, cy, bw, bh in boxes:
        x0, x1 = (cx - bw / 2) * w, (cx + bw / 2) * w
        y0, y1 = (cy - bh / 2) * h, (cy + bh / 2) * h
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        q = (corners - c) @ A.T + c + t
        nx0, ny0 = q.min(axis=0)
        nx1, ny1 = q.max(axis=0)
        area = (nx1 - nx0) * (ny1 - ny0)
        kx0, kx1 = np.clip([nx0, nx1], 0, w)
        ky0, ky1 = np.clip([ny0, ny1], 0, h)
        visible = max(0.0, kx1 - kx0) * max(0.0, ky1 - ky0)
        if area <= 0 or visible / area < min_visible:
            continue
        out.append((cls, (kx0 + kx1) / 2 / w, (ky0 + ky1) / 2 / h, (kx1 - kx0) / w, (ky1 - ky0) / h))
    return out


def augment_with_boxes(img, boxes, rng, aug_cfg=None):
    """Tire une transformation et l'applique à l'image et à ses boîtes.

    Retourne (image, boîtes, paramètres). `rng` = np.random.RandomState (graine = résultat reproductible)."""
    c = {**DEFAULT_AUG, **(aug_cfg or {})}
    params = sample_affine(rng, c["rotation_deg"], c["translation"], c["zoom"])
    h, w = img.shape[:2]
    return (warp_image(img, params), transform_boxes(boxes, params, w, h, c["box_min_visible"]), params)


def plot_box_augmentation(df, n_versions=2, aug_cfg=None, seed=0, out_path=None):
    """Une colonne par image de `df` (colonnes image_path, label_path, class).

    Ligne 1 : image d'origine avec ses boîtes ; lignes suivantes : versions transformées, avec les
    boîtes déplacées par `transform_boxes`. Le titre indique le nombre de boîtes (avant -> après)."""
    from matplotlib.patches import Rectangle
    from src.data import CLASS_COLORS, parse_label_file
    from src.preprocessing import load_gray
    from src.utils import ROOT

    rng = np.random.RandomState(seed)
    n = len(df)
    fig, axes = plt.subplots(1 + n_versions, n, figsize=(2.9 * n, 3.0 * (1 + n_versions)), squeeze=False)
    for col, (_, r) in enumerate(df.iterrows()):
        gray = load_gray(ROOT / r["image_path"])
        boxes, _ = parse_label_file(ROOT / r["label_path"])
        versions = [(gray, boxes, f"{r['class']} | {Path(r['image_path']).stem}\n{len(boxes)} boîte(s)")]
        for _ in range(n_versions):
            im, bx, _p = augment_with_boxes(gray, boxes, rng, aug_cfg)
            versions.append((im, bx, f"{len(boxes)} -> {len(bx)} boîte(s)"))
        for row, (im, bx, title) in enumerate(versions):
            ax = axes[row, col]
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            H, W = im.shape[:2]
            for c, cx, cy, w, h in bx:
                ax.add_patch(Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H, fill=False,
                                       edgecolor=CLASS_COLORS[int(c)], linewidth=2))
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
    plt.show()


def box_transform_stats(df, n_draws=20, aug_cfg=None, seed=0):
    """Combien de boîtes sont écartées, et de combien la boîte grandit-elle ?

    Pour chaque image de `df` (colonnes label_path, width, height) on tire `n_draws` transformations.
    Résultat par type de boîte : nombre de boîtes testées, % écartées, rapport de surface
    (nouvelle surface / surface zoomée attendue : 1 = même taille, plus = la rotation agrandit la boîte)."""
    from src.data import CLASS_NAMES, parse_label_file
    from src.utils import ROOT

    c = {**DEFAULT_AUG, **(aug_cfg or {})}
    rng = np.random.RandomState(seed)
    rows = []
    for _, r in df.iterrows():
        boxes, _ = parse_label_file(ROOT / r["label_path"])
        for _ in range(n_draws):
            params = sample_affine(rng, c["rotation_deg"], c["translation"], c["zoom"])
            for b in boxes:
                out = transform_boxes([b], params, r["width"], r["height"], c["box_min_visible"])
                ratio = out[0][3] * out[0][4] / (params["scale"] ** 2 * b[3] * b[4]) if out else np.nan
                rows.append((CLASS_NAMES[int(b[0])], len(out) == 0, ratio))
    d = pd.DataFrame(rows, columns=["type", "ecartee", "rapport"])
    g = d.groupby("type")
    return pd.DataFrame({
        "boites_testees": g.size(),
        "ecartees_pct": (g["ecartee"].mean() * 100).round(2),
        "rapport_surface_median": g["rapport"].median().round(2),
        "rapport_surface_p95": g["rapport"].quantile(0.95).round(2),
    })


def build_dataset(X, y, training, batch_size=32, augment_fn=None, seed=42):
    """tf.data : train = mélange (graine) + augmentations ; val/test = aucun hasard, ordre fixe."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
        if augment_fn is not None:
            ds = ds.map(lambda a, b: (augment_fn(a), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def plot_augmentation(X, y, classes, augment_fn, n=6, n_versions=2, seed=0, out_path=None):
    """Ligne 1 : images prétraitées (sans augmentation) ; lignes suivantes : versions augmentées."""
    idx = np.random.RandomState(seed).choice(len(X), n, replace=False)
    fig, axes = plt.subplots(1 + n_versions, n, figsize=(2.4 * n, 2.5 * (1 + n_versions)))
    for col, i in enumerate(idx):
        axes[0, col].imshow(X[i]); axes[0, col].set_title(f"{classes[y[i]]}", fontsize=8)
        for r in range(n_versions):
            axes[r + 1, col].imshow(augment_fn(X[i]).numpy())
    axes[0, 0].set_ylabel("original")
    for r in range(n_versions):
        axes[r + 1, 0].set_ylabel(f"augmenté {r + 1}")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
    plt.show()