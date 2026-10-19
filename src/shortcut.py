"""Thermomètre du raccourci de source (étape 2.8) : devine-t-on la classe sans regarder la tumeur ?

Un petit arbre de décision reçoit seulement des chiffres globaux (luminosité, netteté, métadonnées du
fichier...) et essaie de deviner la classe. Si le score est bien au-dessus de « toujours la classe
majoritaire », la classe est trahie par ces chiffres. Ce n'est qu'un INDICE : une vraie différence
d'anatomie peut aussi les faire varier."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import estimate_background, load_gray, preprocess_image
from src.utils import ROOT

PIXEL_FEATURES = ["mean", "std", "sharpness", "dark_share", "bright_share", "bg_level"]
LABELS = {"mean": "luminosité moyenne", "std": "contraste", "sharpness": "netteté",
          "dark_share": "part de noir (≤ 15)", "bright_share": "part de clair (≥ 200)",
          "bg_level": "niveau du fond"}


def pixel_features(gray: np.ndarray) -> dict:
    """Six chiffres globaux d'une image en niveaux de gris (tableau uint8, H x W)."""
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return {
        "mean": float(g.mean()),                        # luminosité moyenne
        "std": float(g.std()),                          # contraste
        "sharpness": float(lap.var()),                  # netteté : variance du laplacien (grande = net)
        "dark_share": float((gray <= 15).mean()),       # part de pixels presque noirs
        "bright_share": float((gray >= 200).mean()),    # part de pixels très clairs
        "bg_level": float(estimate_background(gray)),   # niveau de gris du fond
    }


def compute_pixel_table(df: pd.DataFrame, size: int = 224, pad_mode: str = "background") -> pd.DataFrame:
    """Une ligne par image de `df` : les 6 chiffres sur l'image d'origine (avant_*) et sur l'image
    prétraitée 224x224 (apres_*). Les images sont lues une fois chacune ; rien n'est modifié."""
    rows = []
    for rel in df["image_path"]:
        before = pixel_features(load_gray(ROOT / rel))
        after = pixel_features(preprocess_image(rel, size, pad_mode)[:, :, 0])
        rows.append({"image_path": rel,
                     **{f"avant_{k}": v for k, v in before.items()},
                     **{f"apres_{k}": v for k, v in after.items()}})
    return pd.DataFrame(rows)


def metadata_features(m: pd.DataFrame) -> pd.DataFrame:
    """Les métadonnées de fichier utilisées au Jour 1 : taille, forme, poids, mode couleur."""
    return pd.DataFrame({
        "width": m["width"].values, "height": m["height"].values,
        "aspect": (m["width"] / m["height"]).values, "size_kb": m["size_kb"].values,
        "mode": m["mode"].astype("category").cat.codes.values,
    })


def tree_baseline(X: pd.DataFrame, y, seed: int = 42, max_depth: int = 4) -> dict:
    """Arbre de profondeur 4, 5 validations croisées (même méthode qu'au Jour 1)."""
    y = np.asarray(y)
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)
    pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    rep = classification_report(y, pred, zero_division=0, output_dict=True)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            **{f"rappel {c}": rep[c]["recall"] for c in sorted(set(y))}}


def majority_baseline(y) -> dict:
    """Score d'un « modèle » qui répond toujours la classe la plus fréquente."""
    y = np.asarray(y)
    top = pd.Series(y).value_counts().idxmax()
    pred = np.full(len(y), top, dtype=y.dtype)
    rep = classification_report(y, pred, zero_division=0, output_dict=True)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            **{f"rappel {c}": rep[c]["recall"] for c in sorted(set(y))}}


def thermometer_table(m: pd.DataFrame, pixels: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Compare 4 lignes : classe majoritaire, métadonnées, pixels d'origine, pixels prétraités.

    `m` = lignes du manifeste (colonnes class, width, height, size_kb, mode) et `pixels` = résultat de
    compute_pixel_table, dans le même ordre."""
    y = m["class"].values
    X_before = pixels[[f"avant_{k}" for k in PIXEL_FEATURES]]
    X_after = pixels[[f"apres_{k}" for k in PIXEL_FEATURES]]
    rows = {
        "Toujours la classe majoritaire": majority_baseline(y),
        "Métadonnées du fichier (taille, poids, mode)": tree_baseline(metadata_features(m), y, seed),
        "Pixels : images d'origine": tree_baseline(X_before, y, seed),
        "Pixels : après prétraitement": tree_baseline(X_after, y, seed),
    }
    return pd.DataFrame(rows).T.round(3)


def tree_importances(pixels: pd.DataFrame, y, prefix: str, seed: int = 42, max_depth: int = 4) -> pd.Series:
    """Quel chiffre l'arbre utilise-t-il le plus ? (arbre entraîné sur toutes les images, à titre descriptif)"""
    X = pixels[[f"{prefix}_{k}" for k in PIXEL_FEATURES]]
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed).fit(X, np.asarray(y))
    imp = pd.Series(clf.feature_importances_, index=[LABELS[k] for k in PIXEL_FEATURES])
    return imp.sort_values(ascending=False).round(3)