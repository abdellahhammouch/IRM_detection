import json

import pandas as pd

from src.data import name_pattern
from src.utils import ROOT

MANIFEST_COLUMNS = [
    "image_path", "label_path", "class", "split", "is_representative",
    "group_id", "group_size", "status", "consistency", "n_boxes",
    "use_for_classification", "use_for_detection", "split_origin",
    "name_pattern", "sha256", "width", "height", "mode", "format", "size_kb",
]
SPLITS = {"train", "val", "test", "excluded"}


def build_manifest(m: pd.DataFrame, props: pd.DataFrame = None) -> pd.DataFrame:
    """Assemble le manifeste : une ligne par image, colonnes dans un ordre fixe."""
    out = m.copy()
    out["group_size"] = out.groupby("group_id")["image_path"].transform("size")
    out["name_pattern"] = out["stem"].map(name_pattern)
    if props is not None:
        cols = ["image_path", "width", "height", "mode", "format", "size_kb"]
        out = out.merge(props[cols], on="image_path", how="left")
    return out[MANIFEST_COLUMNS].sort_values("image_path").reset_index(drop=True)


def validate_manifest(df: pd.DataFrame, check_files: bool = True) -> list:
    """Retourne la liste des problèmes trouvés (liste vide = manifeste valide)."""
    problems = []
    if df["image_path"].duplicated().any():
        problems.append("chemins d'image en double")
    unknown = set(df["split"].unique()) - SPLITS
    if unknown:
        problems.append(f"valeurs de split inconnues : {unknown}")

    el = df[df["use_for_classification"]]
    if (df.loc[~df["use_for_classification"], "split"] != "excluded").any():
        problems.append("une image non utilisable n'est pas 'excluded'")
    if (el["split"] == "excluded").any():
        problems.append("une image utilisable est 'excluded'")
    if (el.groupby("group_id")["split"].nunique() > 1).any():
        problems.append("un groupe est coupé entre plusieurs splits")

    reps = df[df["is_representative"]]
    if reps["group_id"].duplicated().any():
        problems.append("un groupe a plusieurs représentants")
    if not reps["use_for_classification"].all():
        problems.append("un représentant n'est pas utilisable en classification")
    if set(el["group_id"]) - set(reps["group_id"]):
        problems.append("un groupe utilisable n'a pas de représentant")

    det = df[df["use_for_detection"]]
    if (det["class"] == "No Tumor").any():
        problems.append("une image No Tumor est utilisée en détection")
    if (det["status"] != "ok").any() or (det["n_boxes"] < 1).any():
        problems.append("une image de détection n'a pas d'annotation valide")

    if check_files:
        paths = list(df["image_path"]) + list(df.loc[df["label_path"].notna(), "label_path"])
        missing = [p for p in paths if not (ROOT / p).exists()]
        if missing:
            problems.append(f"{len(missing)} fichiers introuvables (ex. {missing[0]})")
    return problems


def build_report(df: pd.DataFrame, inv: pd.DataFrame, cfg: dict) -> dict:
    """Chiffres clés du découpage, pour le suivi dans Git et le README."""
    def table(frame):
        ct = pd.crosstab(frame["class"], frame["split"])
        return {c: {s: int(n) for s, n in row.items()} for c, row in ct.iterrows()}

    reps = df[df["is_representative"]]
    return {
        "n_images": int(len(df)),
        "status": {k: int(v) for k, v in df["status"].value_counts().items()},
        "use_for_classification": int(df["use_for_classification"].sum()),
        "use_for_detection": int(df["use_for_detection"].sum()),
        "n_groups": int(df["group_id"].nunique()),
        "distinct_images_per_split": table(reps),
        "files_per_split": table(df),
        "seed": cfg["seed"],
        "split_ratios": cfg["data"]["split"],
        "duplicates_thresholds": cfg["duplicates"],
        "label_conflict_files": sorted(df.loc[df["status"] == "label_conflict", "image_path"]),
        "orphan_image_files": sorted(df.loc[df["status"] == "orphan_image", "image_path"]),
        "orphan_label_files": sorted(inv.loc[inv["status"] == "orphan_label", "label_path"]),
    }


def save_manifest(df: pd.DataFrame, report: dict, out_dir: str = "manifests"):
    out = ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "manifest.csv", index=False, lineterminator="\n")
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    (out / "data_report.json").write_text(text, encoding="utf-8")
    return out


def load_manifest(path: str = "manifests/manifest.csv") -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


