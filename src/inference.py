"""Inférence pour l'application (Jour 7) : même prétraitement qu'à l'entraînement, sans Streamlit."""
import numpy as np
from PIL import Image, ImageDraw

from src.preprocessing import pil_to_gray, preprocess_array

DET_NAMES = {0: "Glioma", 1: "Meningioma", 2: "Pituitary"}
DET_COLORS = {0: (255, 0, 0), 1: (255, 215, 0), 2: (0, 220, 0)}


def load_classifier(path):
    import keras
    return keras.models.load_model(path)


def load_detector(path):
    from ultralytics import YOLO
    return YOLO(str(path))


def classify(model, pil_img, classes, size=224, pad_mode="background"):
    """Image PIL -> (classe prédite, probabilités par classe). Le modèle reçoit des pixels 0-255."""
    x = preprocess_array(pil_to_gray(pil_img), size, pad_mode)[None]
    p = np.asarray(model.predict(x, verbose=0))[0]
    return classes[int(p.argmax())], dict(zip(classes, map(float, p)))


def detect(model, pil_img, conf=0.5, imgsz=640):
    """Image PIL -> liste de boîtes (type, confiance, (x0, y0, x1, y1) en pixels), triées par confiance."""
    r = model.predict(source=pil_img.convert("RGB"), conf=conf, imgsz=imgsz, verbose=False)[0]
    out = [(DET_NAMES[int(c)], float(s), tuple(float(v) for v in xy))
           for c, s, xy in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy())]
    return sorted(out, key=lambda b: -b[1])


def draw_boxes(pil_img, boxes):
    """Dessine les boîtes et leur score sur une copie de l'image."""
    img = pil_img.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    inv = {v: k for k, v in DET_NAMES.items()}
    for name, score, (x0, y0, x1, y1) in boxes:
        col = DET_COLORS[inv[name]]
        d.rectangle([x0, y0, x1, y1], outline=col, width=max(2, img.width // 200))
        d.text((x0 + 3, max(y0 - 12, 0)), f"{name} {score:.2f}", fill=col)
    return img