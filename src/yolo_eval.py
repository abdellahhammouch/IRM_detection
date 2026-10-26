"""Évaluation de YOLO (Jour 5) : métriques par classe, faux positifs / objets manqués, fausses alarmes No Tumor.
N'importe pas TensorFlow : à utiliser dans un processus séparé (scripts/eval_yolo.py)."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

from src.split import _iou
from src.utils import ROOT

NAMES = {0: "Glioma", 1: "Meningioma", 2: "Pituitary"}
COLORS = {0: "red", 1: "yellow", 2: "lime"}


def write_subset_yaml(manifest: pd.DataFrame, index: pd.DataFrame, out_dir="data/yolo") -> Path:
    """data_allmatch.yaml : val et test restreints aux images dont TOUTES les boîtes ont le type du dossier
    (consistency == all_match). Listes d'images en chemins absolus (fichiers générés, hors Git)."""
    out = ROOT / out_dir
    m = index.merge(manifest[["image_path", "consistency"]], on="image_path", how="left")
    cfg = {"nc": 3, "names": NAMES}
    for split in ("val", "test"):
        sel = m[(m["split"] == split) & (m["consistency"] == "all_match")]
        lst = out / f"{split}_allmatch.txt"
        lst.write_text("".join(str((ROOT / p).resolve()) + "\n" for p in sel["yolo_image"]), encoding="utf-8")
        cfg[split] = str(lst.resolve())
    cfg["train"] = cfg["val"]            # clé obligatoire pour Ultralytics, non utilisée ici
    path = out / "data_allmatch.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def metrics_table(m) -> pd.DataFrame:
    """Tableau par classe + ligne 'toutes' depuis le résultat de model.val()."""
    b = m.box
    rows = [{"classe": NAMES[int(c)], "précision": b.p[i], "rappel": b.r[i], "mAP50": b.ap50[i], "mAP50-95": b.ap[i]}
            for i, c in enumerate(b.ap_class_index)]
    rows.append({"classe": "toutes", "précision": b.mp, "rappel": b.mr, "mAP50": b.map50, "mAP50-95": b.map})
    return pd.DataFrame(rows).set_index("classe").round(3)


def read_gt(label_path):
    """Boîtes de vérité terrain d'un fichier YOLO : liste de (classe, cx, cy, w, h)."""
    p = Path(label_path)
    if not p.exists():
        return []
    return [(int(float(l.split()[0])), *map(float, l.split()[1:5])) for l in p.read_text().splitlines() if l.strip()]


def match_counts(preds, gts, iou_min=0.5):
    """Compare prédictions (classe, conf, cx, cy, w, h) et vérité (classe, cx, cy, w, h) d'UNE image.
    Retourne {classe: [TP, FP, FN, FP_mauvaise_classe]}. Une prédiction est un TP si elle recouvre (IoU >= iou_min)
    une boîte de même classe non déjà prise (les plus confiantes d'abord)."""
    out = {c: [0, 0, 0, 0] for c in NAMES}
    free = list(range(len(gts)))
    for cls, conf, *box in sorted(preds, key=lambda p: -p[1]):
        cands = [(_iou(box, gts[i][1:]), i) for i in free if gts[i][0] == cls]
        best = max(cands, default=(0.0, None))
        if best[0] >= iou_min:
            out[cls][0] += 1
            free.remove(best[1])
        else:
            out[cls][1] += 1
            if any(g[0] != cls and _iou(box, g[1:]) >= iou_min for g in gts):   # bonne place, mauvais type
                out[cls][3] += 1
    for i in free:
        out[gts[i][0]][2] += 1
    return out


def detection_counts(model, image_paths, label_dir_of, conf=0.25, imgsz=640):
    """Compte TP / FP / objets manqués par classe sur une liste d'images (seuil de confiance `conf`)."""
    tot = {c: np.zeros(4, dtype=int) for c in NAMES}
    for r, path in zip(model.predict(source=[str(p) for p in image_paths], conf=conf, imgsz=imgsz,
                                     stream=True, verbose=False), image_paths):
        b = r.boxes
        preds = [(int(c), float(s), *xy) for c, s, xy in zip(b.cls.cpu().numpy(), b.conf.cpu().numpy(), b.xywhn.cpu().numpy())]
        for c, v in match_counts(preds, read_gt(label_dir_of(path))).items():
            tot[c] += v
    df = pd.DataFrame({NAMES[c]: v for c, v in tot.items()}, index=["TP", "FP", "manqués", "dont FP bon endroit mauvais type"]).T
    df["précision"] = (df.TP / (df.TP + df.FP).clip(lower=1)).round(3)
    df["rappel"] = (df.TP / (df.TP + df["manqués"]).clip(lower=1)).round(3)
    return df


def sweep_table(model, paths, manifest, split, confs=(0.25, 0.3, 0.4, 0.5, 0.6), imgsz=640):
    """Pour chaque seuil de confiance : précision, rappel, F1 (toutes classes, IoU >= 0,5) et
    part d'images No Tumor avec fausse alarme. Sert à CHOISIR le seuil sur la validation."""
    rows = []
    for conf in confs:
        c = detection_counts(model, paths, label_of, conf, imgsz)
        tp, fp, fn = int(c.TP.sum()), int(c.FP.sum()), int(c["manqués"].sum())
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        nt = no_tumor_false_alarms(model, manifest, split, conf, imgsz)
        rows.append({"seuil": conf, "TP": tp, "FP": fp, "manqués": fn, "précision": round(p, 3), "rappel": round(r, 3),
                     "F1": round(2 * p * r / max(p + r, 1e-9), 3), "No Tumor avec alarme": f"{nt['avec_fausse_alarme']}/{nt['images_no_tumor']}"})
    return pd.DataFrame(rows).set_index("seuil")


def label_of(img_path):
    p = Path(img_path)
    return p.parents[2] / "labels" / p.parent.name / (p.stem + ".txt")


def no_tumor_false_alarms(model, manifest, split, conf=0.25, imgsz=640):
    """Images No Tumor (représentants) du split : combien reçoivent au moins une boîte ? (aucune attendue)"""
    m = manifest[(manifest["class"] == "No Tumor") & (manifest["split"] == split)
                 & manifest["is_representative"] & manifest["use_for_classification"]]
    paths = [ROOT / p for p in m["image_path"]]
    n, confs = 0, []
    for r in model.predict(source=[str(p) for p in paths], conf=conf, imgsz=imgsz, stream=True, verbose=False):
        if len(r.boxes):
            n += 1
            confs.append(float(r.boxes.conf.max()))
    return {"images_no_tumor": len(paths), "avec_fausse_alarme": n, "part": round(n / max(len(paths), 1), 3),
            "confiance_mediane_des_alarmes": round(float(np.median(confs)), 3) if confs else None}


def plot_predictions(model, index, split, n=8, conf=0.25, imgsz=640, seed=0, out_path=None):
    """n images au hasard : vérité terrain en blanc (pointillés), prédictions en couleur avec leur confiance."""
    import matplotlib.pyplot as plt
    rows = index[index["split"] == split].sample(n, random_state=seed)
    fig, ax = plt.subplots(2, n // 2, figsize=(3.4 * n // 2, 8))
    for a, (_, r) in zip(ax.flat, rows.iterrows()):
        path = ROOT / r["yolo_image"]
        res = next(iter(model.predict(source=str(path), conf=conf, imgsz=imgsz, stream=True, verbose=False)))
        img = res.orig_img[..., ::-1]
        H, W = img.shape[:2]
        a.imshow(img)
        for c, cx, cy, w, h in read_gt(label_of(path)):
            a.add_patch(plt.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H, fill=False, ec="white", ls="--", lw=1.5))
        for c, s, (cx, cy, w, h) in zip(res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.xywhn.cpu().numpy()):
            a.add_patch(plt.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H, fill=False, ec=COLORS[int(c)], lw=2))
            a.text((cx - w / 2) * W, (cy - h / 2) * H - 3, f"{NAMES[int(c)][:4]} {s:.2f}", color=COLORS[int(c)], fontsize=7)
        a.set_title(f"{r['class']} | {Path(r['yolo_image']).stem}", fontsize=8); a.axis("off")
    fig.suptitle("pointillés blancs = vérité terrain | couleur = prédiction (rouge Glioma, jaune Meningioma, vert Pituitary)", fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
    plt.close(fig)