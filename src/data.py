import warnings
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.patches as patches
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from src.utils import ROOT

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
LABEL_EXT = ".txt"
EXT_FORMAT = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}
CLASS_COLORS = {0: "red", 1: "yellow", 2: "cyan", 3: "lime"}


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def find_unexpected_files(raw_dir) -> list[str]:
    """Fichiers qui ne sont ni une image ni une annotation .txt."""
    ok = IMAGE_EXTS | {LABEL_EXT}
    return sorted(
        _rel(p) for p in Path(raw_dir).rglob("*")
        if p.is_file() and p.suffix.lower() not in ok
    )


def build_inventory(raw_dir) -> pd.DataFrame:
    """Une ligne par (split, classe, nom de fichier sans extension).

    Structure attendue : raw_dir/<Split>/<Classe>/{images,labels}/
    """
    rows = []
    for split_dir in sorted(p for p in Path(raw_dir).iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            img_dir, lbl_dir = class_dir / "images", class_dir / "labels"

            img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
            lbl_files = [p for p in lbl_dir.iterdir() if p.suffix.lower() == LABEL_EXT]
            images = {p.stem: p for p in img_files}
            labels = {p.stem: p for p in lbl_files}

            # Deux images de même nom (ex. a.jpg et a.png) s'écraseraient dans le dict
            if len(images) != len(img_files) or len(labels) != len(lbl_files):
                warnings.warn(f"Noms en collision dans {class_dir}")

            for stem in sorted(images.keys() | labels.keys()):
                img, lbl = images.get(stem), labels.get(stem)
                rows.append({
                    "split_origin": split_dir.name,
                    "class": class_dir.name,
                    "stem": stem,
                    "image_path": _rel(img) if img else None,
                    "label_path": _rel(lbl) if lbl else None,
                    "has_image": img is not None,
                    "has_label": lbl is not None,
                })
    return pd.DataFrame(rows)


def summarize_inventory(df: pd.DataFrame):
    """Tableau classe x split + liste des orphelins (image sans label ou l'inverse)."""
    table = df.pivot_table(
        index="class", columns="split_origin", values="stem",
        aggfunc="count", margins=True, margins_name="Total", fill_value=0,
    )
    orphans = df[~(df.has_image & df.has_label)]
    return table, orphans


def add_status(df: pd.DataFrame) -> pd.DataFrame:
    """ok = image + label ; sinon orphan_image / orphan_label."""
    df = df.copy()
    df["status"] = "ok"
    df.loc[df.has_image & ~df.has_label, "status"] = "orphan_image"
    df.loc[~df.has_image & df.has_label, "status"] = "orphan_label"
    return df


def class_distribution(df: pd.DataFrame):
    """Nombre et pourcentage d'images par classe et par split (images seulement)."""
    imgs = df[df.has_image]
    counts = imgs.pivot_table(
        index="class", columns="split_origin", values="stem",
        aggfunc="count", fill_value=0,
    )
    pct = (counts / counts.sum() * 100).round(1)
    return counts, pct


def plot_class_distribution(counts, pct, out_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts.plot(kind="bar", ax=axes[0], rot=0, title="Nombre d'images")
    pct.plot(kind="bar", ax=axes[1], rot=0, title="Proportion au sein de chaque split (%)")
    for ax in axes:
        ax.set_xlabel("")
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
    plt.show()


def check_images(df: pd.DataFrame) -> pd.DataFrame:
    """Ouvre et décode chaque image ; relève format, mode, dimensions, poids."""
    rows = []
    for rel in df.loc[df.has_image, "image_path"]:
        p = ROOT / rel
        rec = {
            "image_path": rel, "readable": True, "error": None,
            "format": None, "mode": None, "width": None, "height": None,
            "ext_ok": False, "size_kb": round(p.stat().st_size / 1024, 1),
        }
        try:
            with Image.open(p) as im:
                rec["format"], rec["mode"] = im.format, im.mode
                rec["width"], rec["height"] = im.size
                im.load()  # décode tous les pixels : détecte les fichiers tronqués
            rec["ext_ok"] = EXT_FORMAT.get(p.suffix.lower()) == rec["format"]
        except Exception as e:
            rec["readable"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
        rows.append(rec)
    return pd.DataFrame(rows)

def name_pattern(stem: str) -> str:
    """'Tr-gl_0495' -> 'Tr-gl_#' : regroupe les noms par famille (indice de la source)."""
    import re
    return re.sub(r"\d+", "#", stem)


def metadata_baseline(m: pd.DataFrame, features=("width", "height", "aspect", "size_kb", "mode"),
                      seed: int = 42) -> dict:
    """Devine la classe avec les seules métadonnées de fichier (aucun pixel).

    Sert de thermomètre : si ça marche bien, la classe est trahie par la source.
    """
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.tree import DecisionTreeClassifier

    full = pd.DataFrame({
        "width": m["width"].values,
        "height": m["height"].values,
        "aspect": (m["width"] / m["height"]).values,
        "size_kb": m["size_kb"].values,
        "mode": m["mode"].astype("category").cat.codes.values,
    })
    X = full[list(features)]
    y = m["class"].values
    clf = DecisionTreeClassifier(max_depth=4, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, X, y, cv=cv)
    rep = classification_report(y, pred, zero_division=0, output_dict=True)
    return {
        "accuracy": round(accuracy_score(y, pred), 3),
        "majority_baseline": round(pd.Series(y).value_counts(normalize=True).max(), 3),
        "recall": {c: round(rep[c]["recall"], 2) for c in sorted(set(y))},
        "report": classification_report(y, pred, zero_division=0),
    }


def parse_label_file(path):
    """Lit un .txt YOLO. Retourne (boîtes, problèmes de format).

    Format attendu : 'classe cx cy w h' (5 valeurs, coordonnées normalisées).
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    boxes, format_issues = [], []
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            format_issues.append("polygon" if len(parts) > 5 else "malformed")
            continue
        try:
            c, cx, cy, w, h = (float(x) for x in parts)
        except ValueError:
            format_issues.append("non_numeric")
            continue
        boxes.append((c, cx, cy, w, h))
    return boxes, format_issues


def build_annotation_tables(inv_ok: pd.DataFrame, tol: float = 1e-6):
    """Retourne (boîtes, fichiers) : un tableau par boîte, un tableau par fichier."""
    box_rows, file_rows = [], []
    cols = ["label_path", "class", "split_origin"]
    for rec in inv_ok.loc[inv_ok.has_label, cols].to_dict("records"):
        boxes, fmt = parse_label_file(ROOT / rec["label_path"])
        file_rows.append({
            **rec,
            "n_boxes": len(boxes),
            "format_issues": ",".join(sorted(set(fmt))),
            "duplicate_boxes": len(set(boxes)) < len(boxes),
        })
        for c, cx, cy, w, h in boxes:
            flags = []
            if c != int(c) or c < 0:
                flags.append("bad_class_id")
            if not all(-tol <= v <= 1 + tol for v in (cx, cy, w, h)):
                flags.append("out_of_range")
            if w <= 0 or h <= 0:
                flags.append("zero_size")
            if (cx - w / 2 < -tol or cx + w / 2 > 1 + tol
                    or cy - h / 2 < -tol or cy + h / 2 > 1 + tol):
                flags.append("outside_image")
            box_rows.append({
                **rec, "class_id": int(c), "cx": cx, "cy": cy, "w": w, "h": h,
                "area": w * h, "flags": ",".join(flags),
            })
    return pd.DataFrame(box_rows), pd.DataFrame(file_rows)


def parse_label_file(path):
    """Lit un .txt YOLO. Retourne (boîtes, problèmes de format).

    Format attendu : 'classe cx cy w h' (5 valeurs, coordonnées normalisées).
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    boxes, format_issues = [], []
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            format_issues.append("polygon" if len(parts) > 5 else "malformed")
            continue
        try:
            c, cx, cy, w, h = (float(x) for x in parts)
        except ValueError:
            format_issues.append("non_numeric")
            continue
        boxes.append((c, cx, cy, w, h))
    return boxes, format_issues


def build_annotation_tables(inv_ok: pd.DataFrame, tol: float = 1e-6):
    """Retourne (boîtes, fichiers) : un tableau par boîte, un tableau par fichier."""
    box_rows, file_rows = [], []
    cols = ["label_path", "class", "split_origin"]
    for rec in inv_ok.loc[inv_ok.has_label, cols].to_dict("records"):
        boxes, fmt = parse_label_file(ROOT / rec["label_path"])
        file_rows.append({
            **rec,
            "n_boxes": len(boxes),
            "format_issues": ",".join(sorted(set(fmt))),
            "duplicate_boxes": len(set(boxes)) < len(boxes),
        })
        for c, cx, cy, w, h in boxes:
            flags = []
            if c != int(c) or c < 0:
                flags.append("bad_class_id")
            if not all(-tol <= v <= 1 + tol for v in (cx, cy, w, h)):
                flags.append("out_of_range")
            if w <= 0 or h <= 0:
                flags.append("zero_size")
            if (cx - w / 2 < -tol or cx + w / 2 > 1 + tol
                    or cy - h / 2 < -tol or cy + h / 2 > 1 + tol):
                flags.append("outside_image")
            box_rows.append({
                **rec, "class_id": int(c), "cx": cx, "cy": cy, "w": w, "h": h,
                "area": w * h, "flags": ",".join(flags),
            })
    return pd.DataFrame(box_rows), pd.DataFrame(file_rows)


FOLDER_TO_ID = {"Glioma": 0, "Meningioma": 1, "No Tumor": 2, "Pituitary": 3}


def label_consistency(boxes: pd.DataFrame, files: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque fichier d'annotation : les boîtes ont-elles la classe du dossier ?"""
    b = boxes.copy()
    b["match"] = b["class_id"] == b["class"].map(FOLDER_TO_ID)
    g = b.groupby("label_path")["match"].agg(["sum", "count"])
    g["consistency"] = "mixed"
    g.loc[g["sum"] == g["count"], "consistency"] = "all_match"
    g.loc[g["sum"] == 0, "consistency"] = "none_match"
    out = files.merge(g[["consistency"]], left_on="label_path", right_index=True, how="left")
    out["consistency"] = out["consistency"].fillna("no_box")
    return out


def draw_image_with_boxes(ax, image_path, label_path, title=None):
    """Affiche une image et dessine ses rectangles (une couleur par type)."""
    img = Image.open(ROOT / image_path).convert("RGB")
    W, H = img.size
    ax.imshow(img)
    boxes, _ = parse_label_file(ROOT / label_path)
    for c, cx, cy, w, h in boxes:
        x0, y0 = (cx - w / 2) * W, (cy - h / 2) * H
        ax.add_patch(patches.Rectangle(
            (x0, y0), w * W, h * H, fill=False,
            edgecolor=CLASS_COLORS[int(c)], linewidth=2))
    ax.set_title(title or "", fontsize=8)
    ax.axis("off")


def plot_gallery(df, ncols=4, size=3.2, out_path=None):
    """Galerie d'images avec leurs rectangles. df doit avoir image_path, label_path, class."""
    n = len(df)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size))
    axes = np.array(axes).reshape(-1)
    for ax, (_, r) in zip(axes, df.iterrows()):
        title = f"{r['class']} | {Path(r['image_path']).stem}"
        draw_image_with_boxes(ax, r["image_path"], r["label_path"], title)
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
    plt.show()


def apply_conflict_status(inv: pd.DataFrame, cons: pd.DataFrame) -> pd.DataFrame:
    """Marque 'label_conflict' les fichiers dont aucune boîte ne correspond au dossier."""
    inv = inv.copy()
    conflict = cons.loc[cons.consistency == "none_match", "label_path"]
    inv.loc[inv.label_path.isin(conflict), "status"] = "label_conflict"
    return inv

