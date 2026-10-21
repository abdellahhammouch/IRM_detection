"""Classifieur de base (Jour 3) : MobileNetV2 préentraîné gelé + petite tête, évalué sur la validation."""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src.utils import ROOT

PLAUSIBILITY_F1 = 0.65   # score du thermomètre sur les pixels (étape 2.8)


def build_classifier(n_classes=4, size=224, dropout=0.3, l2=1e-4, weights="imagenet"):
    """Entrée : images uint8 0-255. Le prétraitement MobileNetV2 (x/127,5 - 1) est DANS le modèle,
    donc Streamlit enverra des images brutes. Le corps préentraîné est gelé ; seule la tête apprend."""
    inputs = keras.Input((size, size, 3), dtype="uint8", name="image")
    x = keras.layers.Rescaling(1 / 127.5, offset=-1.0, name="preprocess")(inputs)   # = preprocess_input
    base = keras.applications.MobileNetV2(include_top=False, weights=weights, input_shape=(size, size, 3))
    base.trainable = False
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(dropout)(x)
    out = keras.layers.Dense(n_classes, activation="softmax", dtype="float32", name="classe",
                             kernel_regularizer=keras.regularizers.l2(l2))(x)
    return keras.Model(inputs, out, name="mobilenetv2_base")


def unfreeze_top(model, n_layers=30):
    """Fine-tuning : dégèle les `n_layers` dernières couches de MobileNetV2 (BatchNorm laissées gelées).
    Il faut recompiler ensuite (train_classifier le fait). Retourne le nombre de paramètres entraînables."""
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    base.trainable = True
    for l in base.layers[:-n_layers]:
        l.trainable = False
    for l in base.layers[-n_layers:]:
        l.trainable = not isinstance(l, keras.layers.BatchNormalization)
    return sum(int(np.prod(w.shape)) for w in model.trainable_weights)


def class_weights(y, n_classes=4):
    """Poids 'balanced' : une classe rare pèse plus dans la perte (D16 A)."""
    w = compute_class_weight("balanced", classes=np.arange(n_classes), y=np.asarray(y))
    return {i: float(v) for i, v in enumerate(w)}


def train_classifier(model, train_ds, val_ds, cfg, ytr, ckpt_name="classifier_base.keras", lr=None, epochs=None):
    """Entraîne la tête. EarlyStopping sur la perte de validation (D17), meilleur modèle restauré.
    Retourne (historique, durée en s, chemin du meilleur modèle)."""
    c = cfg["classifier"]
    lr = lr or c["lr"]
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    ckpt = ROOT / cfg["paths"]["models"] / ckpt_name
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    cbs = [keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True),
           keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True),
           keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=2, min_lr=lr / 100)]
    t0 = time.time()
    h = model.fit(train_ds, validation_data=val_ds, epochs=epochs or c["epochs"], callbacks=cbs, verbose=2, shuffle=False,
                  class_weight=class_weights(ytr, len(cfg["data"]["classes"])))
    model.load_weights(ckpt)                       # on repart toujours du MEILLEUR modèle (perte de validation minimale)
    return h.history, time.time() - t0, ckpt


def evaluate_classifier(model, val_ds, yva, classes, cfg, history=None, name="classifier_base", plot=True):
    """Scores sur la VALIDATION (jamais le test) : F1 macro, rappel/précision par classe, matrice de
    confusion, courbes. Enregistre reports/metrics/<name>.json et reports/figures/<name>.png."""
    pred = model.predict(val_ds, verbose=0).argmax(axis=1)     # val_ds : ordre fixe, aligné sur yva
    rep = classification_report(yva, pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(yva, pred, labels=range(len(classes)))
    f1m = float(f1_score(yva, pred, average="macro"))
    res = {"val_accuracy": float((pred == yva).mean()), "val_f1_macro": f1m,
           "recall": {c: rep[c]["recall"] for c in classes},
           "precision": {c: rep[c]["precision"] for c in classes},
           "f1": {c: rep[c]["f1-score"] for c in classes},
           "confusion_matrix": cm.tolist(), "plausibility_threshold": PLAUSIBILITY_F1,
           "above_threshold": f1m > PLAUSIBILITY_F1}
    if history:
        best = int(np.argmin(history["val_loss"]))
        res.update(best_epoch=best + 1, epochs_run=len(history["loss"]), best_val_loss=float(history["val_loss"][best]))
    mdir, fdir = ROOT / cfg["paths"]["metrics"], ROOT / cfg["paths"]["figures"]
    mdir.mkdir(parents=True, exist_ok=True); fdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"{name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    if not plot:
        return res
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if history:
        for k, lab in (("loss", "train"), ("val_loss", "val")):
            ax[0].plot(history[k], label=lab)
        for k, lab in (("accuracy", "train"), ("val_accuracy", "val")):
            ax[1].plot(history[k], label=lab)
    ax[0].set_title("Perte"); ax[1].set_title("Accuracy"); ax[0].legend(); ax[1].legend()
    ax[0].set_xlabel("époque"); ax[1].set_xlabel("époque")
    pct = cm / cm.sum(axis=1, keepdims=True)
    ax[2].imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax[2].text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=8,
                       color="white" if pct[i, j] > 0.5 else "black")
    ax[2].set_xticks(range(len(classes)), classes, rotation=30); ax[2].set_yticks(range(len(classes)), classes)
    ax[2].set_xlabel("prédit"); ax[2].set_ylabel("réel"); ax[2].set_title("Matrice de confusion (validation)")
    plt.tight_layout(); fig.savefig(fdir / f"{name}.png", dpi=110); plt.show()
    return res