"""Grad-CAM : carte de chaleur de ce que regarde le classifieur (dernière couche convolutive)."""
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def gradcam(model, X, class_idx=None):
    """X : images uint8 (n, 224, 224, 3). Retourne (cartes (n, 7, 7) entre 0 et 1, classes utilisées).
    Le score est le logit (avant softmax) de la classe prédite, ou de `class_idx` si donné."""
    pre = model.layers[1]
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    head = model.layers[model.layers.index(base) + 1:]
    x = tf.cast(X, tf.float32)
    with tf.GradientTape() as tape:
        fmap = base(pre(x), training=False)
        tape.watch(fmap)
        y = fmap
        for l in head[:-1]:
            y = l(y, training=False)
        logits = tf.matmul(y, head[-1].kernel) + head[-1].bias
        idx = tf.argmax(logits, axis=1) if class_idx is None else tf.fill([len(X)], class_idx)
        score = tf.gather(logits, idx, batch_dims=1)
    grads = tape.gradient(score, fmap)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * fmap, axis=-1))
    cam = tf.math.divide_no_nan(cam, tf.reduce_max(cam, axis=(1, 2), keepdims=True))
    return cam.numpy(), idx.numpy()


def plot_gradcam(model, X, y, classes, proba=None, out_path=None, alpha=0.45):
    """Ligne 1 : image ; ligne 2 : image + carte (rouge = zone qui décide). Titre : vrai -> prédit."""
    cam, idx = gradcam(model, X)
    n = len(X)
    fig, ax = plt.subplots(2, n, figsize=(2.6 * n, 5.4), squeeze=False)
    for k in range(n):
        up = tf.image.resize(cam[k][None, :, :, None], X.shape[1:3], method="bilinear").numpy()[0, :, :, 0]
        ax[0, k].imshow(X[k]); ax[1, k].imshow(X[k]); ax[1, k].imshow(up, cmap="jet", alpha=alpha, vmin=0, vmax=1)
        ok = "✓" if idx[k] == y[k] else "✗"
        conf = f" ({proba[k].max():.0%})" if proba is not None else ""
        ax[0, k].set_title(f"{classes[y[k]]} -> {classes[idx[k]]} {ok}{conf}", fontsize=8,
                           color="green" if idx[k] == y[k] else "red")
    for a in ax.flat:
        a.axis("off")
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
    plt.show()