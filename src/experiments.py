"""Jour 4 : exécuter une configuration (tête, puis fine-tuning), tableau d'erreurs."""
import time

import keras
import numpy as np
import pandas as pd

from src.augmentation import build_dataset, make_augmenter
from src.classifier import build_classifier, evaluate_classifier, train_classifier, unfreeze_top


def run_config(name, spec, cfg, Xtr, ytr, Xva, yva, classes, seed=42, plot=False):
    """Une configuration, une graine. Phase 1 : tête seule (corps gelé). Phase 2 (si
    spec['finetune_layers'] > 0) : dégel des dernières couches, taux très bas.
    La graine change l'initialisation, le mélange et les augmentations. Validation seulement.
    Retourne (modèle, résultat, historique fusionné)."""
    keras.utils.set_random_seed(seed)
    size, batch = cfg["data"]["image_size"], cfg["classifier"]["batch_size"]
    train_ds = build_dataset(Xtr, ytr, True, batch, make_augmenter(cfg["augmentation"], size, seed), seed)
    val_ds = build_dataset(Xva, yva, False, batch)
    model = build_classifier(len(classes), size, spec["dropout"], spec["l2"])
    tag = f"{name}_s{seed}"
    hist, secs, ckpt = train_classifier(model, train_ds, val_ds, cfg, ytr, f"{tag}.keras")
    n_head = len(hist["loss"])
    if spec["finetune_layers"] > 0:
        unfreeze_top(model, spec["finetune_layers"])
        h2, s2, ckpt = train_classifier(model, train_ds, val_ds, cfg, ytr, f"{tag}.keras",
                                        lr=cfg["finetune"]["lr"], epochs=cfg["finetune"]["epochs"])
        hist = {k: hist[k] + h2[k] for k in hist}
        secs += s2
    res = evaluate_classifier(model, val_ds, yva, classes, cfg, hist, name=tag, plot=plot)
    res.update(config=name, seed=seed, phase1_epochs=n_head, minutes=round(secs / 60, 1),
               gap_train_val_acc=float(hist["accuracy"][res["best_epoch"] - 1] - hist["val_accuracy"][res["best_epoch"] - 1]),
               model_path=str(ckpt))
    return model, res, hist


def summary_row(res, classes):
    """Une ligne de tableau de comparaison."""
    return {"config": res["config"], "graine": res["seed"], "F1 macro": round(res["val_f1_macro"], 3),
            "accuracy": round(res["val_accuracy"], 3),
            **{f"rappel {c}": round(res["recall"][c], 2) for c in classes},
            "meilleure époque": res["best_epoch"], "époques": res["epochs_run"],
            "val_loss": round(res["best_val_loss"], 3), "écart acc. train-val": round(res["gap_train_val_acc"], 3),
            "minutes": res["minutes"]}


def error_table(dva, yva, proba, classes):
    """Une ligne par image de validation : vraie classe, prédite, confiance, taille, famille de nom."""
    pred = proba.argmax(axis=1)
    t = dva[["image_path", "class", "width", "height", "name_pattern"]].copy().reset_index(drop=True)
    t["pred"] = [classes[i] for i in pred]
    t["confiance"] = proba.max(axis=1).round(3)
    t["correct"] = (pred == np.asarray(yva))
    t["taille"] = np.where((t.width == 512) & (t.height == 512), "512x512", "autres tailles")
    return t


def final_test_classifier(model_path, manifest, cfg, classes):
    """ÉVALUATION FINALE sur le TEST (545 images), UNE SEULE FOIS. Refuse de relancer si le résultat existe.
    Retourne (résultat, tableau d'erreurs). Écrit reports/metrics/classifier_test.json et la figure."""
    from src.augmentation import load_split
    from src.utils import ROOT
    out = ROOT / cfg["paths"]["metrics"] / "classifier_test.json"
    if out.exists():
        raise RuntimeError(f"REFUS : {out} existe déjà. Le test ne s'évalue qu'une fois (règle du projet).")
    Xte, yte, dte = load_split(manifest, "test", classes, cfg["data"]["image_size"],
                               cfg["preprocessing"]["pad_mode"])
    model = keras.models.load_model(model_path)
    ds = build_dataset(Xte, yte, False, cfg["classifier"]["batch_size"])
    res = evaluate_classifier(model, ds, yte, classes, cfg, None, name="classifier_test")
    return res, error_table(dte, yte, model.predict(ds, verbose=0), classes)