"""Entraîne YOLOv8 (Jour 5). À lancer dans un TERMINAL, depuis la racine du projet :
    python scripts/train_yolo.py
Lit config.yaml (section yolo). Copie le meilleur modèle dans models/yolo_best.pt."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ultralytics import YOLO

from src.utils import ROOT, load_config

cfg = load_config()
y = cfg["yolo"]
data = ROOT / cfg["paths"]["yolo"] / "data.yaml"
assert data.exists(), f"{data} introuvable : lancer la cellule 120 du notebook"
model = YOLO(y["model"])
model.train(data=str(data), imgsz=y["imgsz"], epochs=y["epochs"], batch=y["batch"], patience=y["patience"],
            workers=y["workers"], seed=cfg["seed"], device=0, plots=True, exist_ok=True,
            project=str(ROOT / cfg["paths"]["models"] / "yolo"), name=y["name"], **y["augment"])
best = ROOT / cfg["paths"]["models"] / "yolo" / y["name"] / "weights" / "best.pt"
shutil.copy(best, ROOT / cfg["paths"]["models"] / "yolo_best.pt")
print("Meilleur modèle copié dans", ROOT / cfg["paths"]["models"] / "yolo_best.pt")