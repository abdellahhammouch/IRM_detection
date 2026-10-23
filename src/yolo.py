"""Préparation du dossier YOLO (étape 2.7) : copies des images, étiquettes en 3 classes, data.yaml."""
import re
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.data import parse_label_file
from src.preprocessing import select_images
from src.utils import ROOT

SPLITS = ("train", "val", "test")
# identifiants d'origine -> identifiants YOLO (No Tumor, 2, n'existe pas en détection)
YOLO_IDS = {0: 0, 1: 1, 3: 2}
YOLO_NAMES = {0: "Glioma", 1: "Meningioma", 2: "Pituitary"}
YOLO_COLORS = {0: "red", 1: "yellow", 2: "lime"}   # mêmes couleurs que les figures précédentes
EXPECTED_IDS = set(YOLO_NAMES)


def clean_name(name: str) -> str:
    """'m (49).jpg' -> 'm_49.jpg' : espaces et parenthèses remplacés par '_' (D13 B)."""
    p = Path(name)
    return re.sub(r"[ ()]+", "_", p.stem).strip("_") + p.suffix.lower()


def build_yolo_dataset(manifest: pd.DataFrame, out_dir: str = "data/yolo",
                       train_dedup: str = "representative"):
    """Crée data/yolo/{images,labels}/{train,val,test}, data.yaml et index.csv depuis le manifeste.

    - images COPIÉES (espaces et parenthèses du nom remplacés par '_') ; étiquettes réécrites avec les identifiants 0, 1, 2 ;
    - une boîte d'un type absent de la détection (ex. No Tumor) est écartée et comptée ;
    - une image qui n'a plus aucune boîte est écartée et comptée (jamais copiée en « fond »).
    Le dossier de sortie est effacé puis refait : il est entièrement généré. data/raw n'est jamais touché.
    Retourne (index, rapport)."""
    out = ROOT / out_dir
    data_dir = (ROOT / "data").resolve()
    protected = {ROOT.resolve(), data_dir, (data_dir / "raw").resolve()}
    if out.resolve() in protected or data_dir not in out.resolve().parents:
        raise ValueError(f"Dossier de sortie refusé (doit être un sous-dossier de data/, pas data/raw) : {out}")
    if out.exists():
        shutil.rmtree(out)

    rows, dropped, skipped, box_counts = [], Counter(), [], Counter()
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for r in select_images(manifest, split, "detection", train_dedup).to_dict("records"):
            boxes, issues = parse_label_file(ROOT / r["label_path"])
            kept = []
            for c, cx, cy, w, h in boxes:
                new = YOLO_IDS.get(int(c))
                if new is None:
                    dropped[int(c)] += 1
                else:
                    kept.append((new, cx, cy, w, h))
            if issues or not kept:
                skipped.append((r["image_path"], "format" if issues else "aucune boîte utilisable"))
                continue
            name = clean_name(Path(r["image_path"]).name)
            dst_img = out / "images" / split / name
            dst_lbl = out / "labels" / split / (Path(name).stem + ".txt")
            if dst_img.exists() or dst_lbl.exists():
                raise FileExistsError(f"Collision de nom dans {split} : {name}")
            shutil.copy2(ROOT / r["image_path"], dst_img)
            dst_lbl.write_text("".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                                       for c, cx, cy, w, h in kept), encoding="utf-8")
            for c, *_ in kept:
                box_counts[(split, c)] += 1
            rows.append({"split": split, "class": r["class"], "image_path": r["image_path"],
                         "label_path": r["label_path"], "yolo_image": dst_img.relative_to(ROOT).as_posix(),
                         "yolo_label": dst_lbl.relative_to(ROOT).as_posix(), "n_boxes": len(kept),
                         "n_types": len({c for c, *_ in kept})})
    index = pd.DataFrame(rows)
    index.to_csv(out / "index.csv", index=False, lineterminator="\n")
    # Pas de chemin absolu : les dossiers sont relatifs à l'emplacement de data.yaml (utile dans Docker)
    cfg = {"train": "images/train", "val": "images/val", "test": "images/test",
           "nc": len(YOLO_NAMES), "names": YOLO_NAMES}
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    report = {
        "images": {s: int((index["split"] == s).sum()) for s in SPLITS},
        "boites": {s: {YOLO_NAMES[c]: box_counts[(s, c)] for c in YOLO_NAMES} for s in SPLITS},
        "boites_ecartees_par_identifiant_origine": dict(dropped),
        "images_ecartees": skipped,
    }
    return index, report


def check_yolo_dataset(index: pd.DataFrame, out_dir: str = "data/yolo", tol: float = 2e-6) -> list:
    """Retourne la liste des problèmes trouvés (liste vide = dossier valide).

    Contrôle : data.yaml, une étiquette par image (et inversement), identifiants 0 à 2, coordonnées
    entre 0 et 1, et que chaque étiquette est bien celle d'origine avec les identifiants convertis."""
    out, problems = ROOT / out_dir, []
    cfg = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    if cfg.get("names") != YOLO_NAMES or cfg.get("nc") != 3:
        problems.append("data.yaml : classes inattendues")
    for split in SPLITS:
        imgs = {p.stem for p in (out / "images" / split).iterdir()}
        lbls = {p.stem for p in (out / "labels" / split).iterdir()}
        if imgs != lbls:
            problems.append(f"{split} : {len(imgs - lbls)} image(s) sans étiquette, {len(lbls - imgs)} étiquette(s) sans image")
        if len(imgs) != int((index["split"] == split).sum()):
            problems.append(f"{split} : le dossier et l'index n'ont pas le même nombre d'images")
    for r in index.to_dict("records"):
        new, fmt = parse_label_file(ROOT / r["yolo_label"])
        if fmt or not new:
            problems.append(f"{r['yolo_label']} : format invalide ou vide")
            continue
        if any(int(c) not in EXPECTED_IDS or c != int(c) for c, *_ in new):
            problems.append(f"{r['yolo_label']} : identifiant hors 0-2")
        if any(not (-tol <= v <= 1 + tol) or w <= 0 or h <= 0 for _, cx, cy, w, h in new for v in (cx, cy, w, h)):
            problems.append(f"{r['yolo_label']} : coordonnées invalides")
        old, _ = parse_label_file(ROOT / r["label_path"])
        want = sorted((YOLO_IDS[int(c)], cx, cy, w, h) for c, cx, cy, w, h in old if int(c) in YOLO_IDS)
        got = sorted(new)
        if len(want) != len(got) or not np.allclose(np.array(want), np.array(got), atol=tol):
            problems.append(f"{r['yolo_label']} : différente de l'étiquette d'origine convertie")
        if not (ROOT / r["yolo_image"]).exists():
            problems.append(f"{r['yolo_image']} : introuvable")
    return problems


def plot_yolo_samples(index: pd.DataFrame, seed: int = 0, out_path=None):
    """6 images du dossier YOLO avec leurs boîtes lues dans les NOUVELLES étiquettes :
    une par type (Glioma, Meningioma, Pituitary) puis jusqu'à 3 images qui ont deux types de boîtes."""
    rng = np.random.RandomState(seed)
    tr = index[index["split"] == "train"]
    picks = [tr[tr["class"] == c].sample(1, random_state=rng).iloc[0] for c in YOLO_NAMES.values()]
    multi = tr[(tr["n_types"] >= 2) & ~tr["yolo_image"].isin([r["yolo_image"] for r in picks])]
    if len(multi):
        picks += [r for _, r in multi.sample(min(3, len(multi)), random_state=rng).iterrows()]
    fig, axes = plt.subplots(1, len(picks), figsize=(3.0 * len(picks), 3.4), squeeze=False)
    for ax, r in zip(axes[0], picks):
        im = Image.open(ROOT / r["yolo_image"]).convert("L")
        W, H = im.size
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        boxes, _ = parse_label_file(ROOT / r["yolo_label"])
        for c, cx, cy, w, h in boxes:
            ax.add_patch(plt.Rectangle(((cx - w / 2) * W, (cy - h / 2) * H), w * W, h * H, fill=False,
                                       edgecolor=YOLO_COLORS[int(c)], linewidth=2))
        ids = sorted({int(c) for c, *_ in boxes})
        ax.set_title(f"dossier {r['class']}\n{Path(r['yolo_image']).stem}\nid YOLO : {ids}", fontsize=8)
        ax.axis("off")
    fig.suptitle("rouge = Glioma (0) | jaune = Meningioma (1) | vert = Pituitary (2)", fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
    plt.show()