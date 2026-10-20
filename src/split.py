import numpy as np
import pandas as pd


def add_use_flags(meta: pd.DataFrame, cons: pd.DataFrame) -> pd.DataFrame:
    """Ajoute n_boxes, consistency et les deux indicateurs d'utilisation."""
    m = meta.merge(cons[["label_path", "n_boxes", "consistency"]],
                   on="label_path", how="left")
    m["n_boxes"] = m["n_boxes"].fillna(0).astype(int)
    m["use_for_classification"] = m["status"].isin(["ok", "orphan_image"])
    m["use_for_detection"] = (
        (m["status"] == "ok") & (m["class"] != "No Tumor") & (m["n_boxes"] > 0)
    )
    return m


def _iou(a, b):
    """a, b = (cx, cy, w, h) normalisés. Part de recouvrement entre deux boîtes (0 à 1)."""
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _compare(ref, other, iou_min=0.5):
    """Compare deux listes de boîtes (classe, cx, cy, w, h)."""
    if len(ref) != len(other):
        return "nombre de boîtes différent"
    if sorted(b[0] for b in ref) != sorted(b[0] for b in other):
        return "classes différentes"
    free = list(other)
    for b in ref:
        cands = [(i, _iou(b[1:], o[1:])) for i, o in enumerate(free) if o[0] == b[0]]
        i, best = max(cands, key=lambda t: t[1])
        if best < iou_min:
            return "position différente"
        free.pop(i)
    return "identiques"


VERDICT_ORDER = ["nombre de boîtes différent", "classes différentes",
                 "position différente", "identiques"]


def compare_twin_boxes(m: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque groupe dont au moins 2 images ont une annotation valide,
    compare les boîtes de chaque image à celles de la première (ordre alphabétique)."""
    by_label = {lp: list(zip(g.class_id, g.cx, g.cy, g.w, g.h))
                for lp, g in boxes.groupby("label_path")}
    rows = []
    lab = m[m["status"] == "ok"].sort_values("image_path")
    for gid, g in lab.groupby("group_id"):
        if len(g) < 2:
            continue
        paths = g["label_path"].tolist()
        ref = by_label.get(paths[0], [])
        verdicts = [_compare(ref, by_label.get(p, [])) for p in paths[1:]]
        worst = min(verdicts, key=VERDICT_ORDER.index)
        rows.append({"group_id": gid, "class": g["class"].iloc[0],
                     "n_members": len(g), "verdict": worst})
    return pd.DataFrame(rows)


def choose_representatives(m: pd.DataFrame) -> pd.DataFrame:
    """Un représentant par groupe. Ordre de préférence :
    1) image utilisable en classification ; 2) utilisable en détection ;
    3) le plus de boîtes ; 4) ordre alphabétique du chemin (toujours le même)."""
    m = m.copy()
    ordered = m.sort_values(
        ["group_id", "use_for_classification", "use_for_detection", "n_boxes", "image_path"],
        ascending=[True, False, False, False, True],
    )
    rep_idx = ordered.drop_duplicates("group_id").index
    m["is_representative"] = m.index.isin(rep_idx) & m["use_for_classification"]
    return m


def split_groups(m: pd.DataFrame, ratios, seed: int = 42) -> pd.DataFrame:
    """Découpe par groupes, classe par classe. Un groupe n'est jamais coupé."""
    el = m[m["use_for_classification"]]
    per_group = el.groupby("group_id")["class"].nunique()
    assert (per_group == 1).all(), "un groupe mélange plusieurs classes"
    group_class = el.groupby("group_id")["class"].first()

    rng = np.random.RandomState(seed)
    split_of = {}
    for cls in sorted(group_class.unique()):
        gids = np.sort(group_class[group_class == cls].index.values)
        rng.shuffle(gids)
        n = len(gids)
        n_val = int(round(n * ratios["val"]))
        n_test = int(round(n * ratios["test"]))
        for g in gids[:n_test]:
            split_of[g] = "test"
        for g in gids[n_test:n_test + n_val]:
            split_of[g] = "val"
        for g in gids[n_test + n_val:]:
            split_of[g] = "train"

    out = m.copy()
    out["split"] = out["group_id"].map(split_of)
    out.loc[~out["use_for_classification"], "split"] = "excluded"
    out["split"] = out["split"].fillna("excluded")
    return out


def extra_box_summary(m: pd.DataFrame, boxes: pd.DataFrame, cmp: pd.DataFrame) -> pd.DataFrame:
    """Pour les groupes au 'nombre de boîtes différent' : quelles boîtes la version
    la plus riche a-t-elle en plus, et de quelles familles de noms viennent les deux versions ?"""
    from collections import Counter
    from src.data import CLASS_NAMES, name_pattern

    by_label = {lp: Counter(g.class_id) for lp, g in boxes.groupby("label_path")}
    ids = cmp.loc[cmp.verdict == "nombre de boîtes différent", "group_id"]
    rows = []
    for gid in ids:
        g = m[(m.group_id == gid) & (m.status == "ok")].copy()
        g["cnt"] = g["label_path"].map(lambda p: sum(by_label.get(p, Counter()).values()))
        g = g.sort_values(["cnt", "image_path"], ascending=[False, True])
        rich, poor = g.iloc[0], g.iloc[-1]
        extra = by_label.get(rich.label_path, Counter()) - by_label.get(poor.label_path, Counter())
        rows.append({
            "group_id": gid,
            "class": rich["class"],
            "boîtes en plus": ", ".join(f"{CLASS_NAMES[int(c)]} x{n}" for c, n in sorted(extra.items())) or "aucune",
            "famille riche": name_pattern(rich.stem),
            "famille pauvre": name_pattern(poor.stem),
        })
    return pd.DataFrame(rows)


