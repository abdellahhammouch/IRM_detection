"""Évalue le détecteur sur la validation OU sur le test. Terminal, racine du projet :
    python scripts/eval_yolo.py --split val
    python scripts/eval_yolo.py --split test     (UNE SEULE FOIS : refuse de relancer si le résultat existe)"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from ultralytics import YOLO

from src.manifest import load_manifest
from src.utils import ROOT, load_config
from src.yolo_eval import (detection_counts, label_of, metrics_table, no_tumor_false_alarms, plot_predictions,
                           sweep_table, write_subset_yaml)

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["val", "test"], required=True)
ap.add_argument("--weights", default="models/yolo_best.pt")
ap.add_argument("--conf", type=float, default=0.25)
ap.add_argument("--sweep", action="store_true", help="validation seulement : compare plusieurs seuils de confiance, puis s'arrête")
a = ap.parse_args()

cfg = load_config()
mdir, fdir = ROOT / cfg["paths"]["metrics"], ROOT / cfg["paths"]["figures"]
mdir.mkdir(parents=True, exist_ok=True); fdir.mkdir(parents=True, exist_ok=True)
out_json = mdir / f"yolo_{a.split}.json"
if a.split == "test" and out_json.exists():
    sys.exit(f"REFUS : {out_json} existe déjà. Le test ne s'évalue qu'une fois (règle du projet).")

manifest, yolo_dir, imgsz = load_manifest(), ROOT / cfg["paths"]["yolo"], cfg["yolo"]["imgsz"]
index = pd.read_csv(yolo_dir / "index.csv")
sub_yaml = write_subset_yaml(manifest, index, cfg["paths"]["yolo"])
model = YOLO(str(ROOT / a.weights))
if a.sweep:
    if a.split != "val":
        sys.exit("--sweep n'est autorisé que sur la validation (le seuil se choisit avant le test).")
    rows_ = index[index["split"] == "val"]
    tab = sweep_table(model, [ROOT / p for p in rows_["yolo_image"]], manifest, "val", imgsz=imgsz)
    print("\n=== choix du seuil de confiance (validation, IoU >= 0,5) ===\n", tab)
    tab.to_csv(mdir / "yolo_val_sweep.csv")
    sys.exit(0)
result = {"weights": a.weights, "split": a.split, "conf_seuil_comptage": a.conf}

for tag, yml in (("tout", yolo_dir / "data.yaml"), ("all_match", sub_yaml)):
    m = model.val(data=str(yml), split=a.split, imgsz=imgsz, batch=cfg["yolo"]["batch"], conf=0.001, iou=0.6,
                  plots=(tag == "tout"), project=str(ROOT / "reports" / "yolo_eval"), name=f"{a.split}_{tag}",
                  exist_ok=True, verbose=False)
    t = metrics_table(m)
    print(f"\n=== {a.split} : {tag} ===\n{t}")
    result[f"metriques_{tag}"] = t.reset_index().to_dict("records")

rows = index[index["split"] == a.split]
paths = [ROOT / p for p in rows["yolo_image"]]
cnt = detection_counts(model, paths, label_of, a.conf, imgsz)
print(f"\n=== comptage à confiance >= {a.conf} (IoU >= 0,5) ===\n{cnt}")
result["comptage"] = cnt.reset_index().rename(columns={"index": "classe"}).to_dict("records")
nt = no_tumor_false_alarms(model, manifest, a.split, a.conf, imgsz)
print("\n=== fausses alarmes sur les images No Tumor ===\n", nt)
result["no_tumor"] = nt
plot_predictions(model, index, a.split, 8, a.conf, imgsz, cfg["seed"], fdir / f"yolo_{a.split}_predictions.png")
out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
print("\nEnregistré :", out_json, "et", fdir / f"yolo_{a.split}_predictions.png")