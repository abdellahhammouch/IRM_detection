"""Pipeline en ligne de commande (Jour 6) : python -m src.pipeline <étape>

Étapes, dans l'ordre : validate -> preprocess -> train -> evaluate -> save.
Chaque étape lit config.yaml, s'arrête avec un code d'erreur (1) en cas de problème (c'est ce qu'Airflow
détecte) et n'utilise JAMAIS le jeu de test (règle D24/D27). Entraînement « léger » : tête seule, peu d'époques."""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

from src.utils import ROOT, load_config

MODEL_NAME = "airflow_light"


def _dirs(cfg):
    proc, met = ROOT / cfg["paths"]["processed"], ROOT / cfg["paths"]["metrics"]
    proc.mkdir(parents=True, exist_ok=True); met.mkdir(parents=True, exist_ok=True)
    return proc, met


def validate(cfg):
    """Le manifeste est-il valide, et tous les fichiers existent-ils ?"""
    from src.manifest import load_manifest, validate_manifest
    m = load_manifest()
    problems = validate_manifest(m, check_files=True)
    if os.environ.get("PIPELINE_SIMULATE_FAILURE") == "1":
        problems.append("ÉCHEC SIMULÉ (démonstration de la gestion d'erreurs)")
    if problems:
        print("PROBLÈMES DÉTECTÉS :", *problems, sep="\n - ")
        sys.exit(1)
    print(f"Manifeste valide : {len(m)} images")


def preprocess(cfg):
    """Prétraite train et validation (une fois) et les enregistre dans data/processed/*.npz."""
    from src.augmentation import load_split
    from src.manifest import load_manifest
    proc, _ = _dirs(cfg)
    m, d = load_manifest(), cfg["data"]
    for split in ("train", "val"):
        X, y, _ = load_split(m, split, d["classes"], d["image_size"], cfg["preprocessing"]["pad_mode"],
                             cfg["preprocessing"]["train_dedup"])
        np.savez_compressed(proc / f"{split}.npz", X=X, y=y)
        print(f"{split} : {X.shape}, classes {np.bincount(y).tolist()}")


def train(cfg):
    """Entraîne la tête (corps gelé) quelques époques ; enregistre models/airflow_light.keras."""
    import keras
    from src.augmentation import build_dataset, make_augmenter
    from src.classifier import build_classifier, train_classifier
    proc, met = _dirs(cfg)
    tr, va = np.load(proc / "train.npz"), np.load(proc / "val.npz")
    seed, size, batch = cfg["seed"], cfg["data"]["image_size"], cfg["classifier"]["batch_size"]
    keras.utils.set_random_seed(seed)
    train_ds = build_dataset(tr["X"], tr["y"], True, batch, make_augmenter(cfg["augmentation"], size, seed), seed)
    val_ds = build_dataset(va["X"], va["y"], False, batch)
    model = build_classifier(len(cfg["data"]["classes"]), size, cfg["classifier"]["dropout"], cfg["classifier"]["l2"])
    light = {**cfg, "classifier": {**cfg["classifier"], "epochs": cfg["airflow"]["epochs"]}}
    hist, secs, ckpt = train_classifier(model, train_ds, val_ds, light, tr["y"], f"{MODEL_NAME}.keras")
    (met / "airflow_train.json").write_text(json.dumps({
        "epochs_run": len(hist["loss"]), "best_val_loss": min(hist["val_loss"]), "seconds": round(secs, 1),
        "model": str(ckpt)}, indent=2), encoding="utf-8")
    print(f"{len(hist['loss'])} époques en {secs:.0f} s ; modèle : {ckpt}")


def evaluate(cfg):
    """Évalue sur la VALIDATION seulement. Échoue si le F1 macro est sous le seuil de qualité (config)."""
    import keras
    from src.augmentation import build_dataset
    from src.classifier import evaluate_classifier
    proc, _ = _dirs(cfg)
    va = np.load(proc / "val.npz")
    model = keras.models.load_model(ROOT / cfg["paths"]["models"] / f"{MODEL_NAME}.keras")
    ds = build_dataset(va["X"], va["y"], False, cfg["classifier"]["batch_size"])
    res = evaluate_classifier(model, ds, va["y"], cfg["data"]["classes"], cfg, None, name="airflow_val", plot=False)
    print(f"Validation : F1 macro {res['val_f1_macro']:.3f}, accuracy {res['val_accuracy']:.3f}")
    if res["val_f1_macro"] < cfg["airflow"]["min_f1"]:
        print(f"ÉCHEC : F1 macro sous le seuil de qualité {cfg['airflow']['min_f1']}")
        sys.exit(1)


def save(cfg):
    """Ajoute une ligne de résumé à reports/metrics/airflow_history.csv."""
    _, met = _dirs(cfg)
    t, v = (json.loads((met / f).read_text(encoding="utf-8")) for f in ("airflow_train.json", "airflow_val.json"))
    row = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "val_f1_macro": round(v["val_f1_macro"], 4),
           "val_accuracy": round(v["val_accuracy"], 4), "epochs": t["epochs_run"], "seconds": t["seconds"], "model": t["model"]}
    path = met / "airflow_history.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print("Métriques enregistrées :", path)


STEPS = {"validate": validate, "preprocess": preprocess, "train": train, "evaluate": evaluate, "save": save}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=STEPS)
    t0 = time.time()
    STEPS[ap.parse_args().step](load_config())
    print(f"(étape terminée en {time.time() - t0:.0f} s)")