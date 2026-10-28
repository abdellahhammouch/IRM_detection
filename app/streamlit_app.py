"""Application Streamlit (Jour 7) : classification + détection sur une IRM cérébrale.
PROJET PÉDAGOGIQUE : ce n'est pas un outil de diagnostic médical."""
import json

import pandas as pd
import streamlit as st
from PIL import Image

from src.inference import classify, detect, draw_boxes, load_classifier, load_detector
from src.utils import ROOT, load_config

cfg = load_config()
CLASSES = cfg["data"]["classes"]
MODELS = ROOT / cfg["paths"]["models"]
METRICS = ROOT / cfg["paths"]["metrics"]

st.set_page_config(page_title="IRM cérébrales : classification et détection", layout="wide")
st.title("IRM cérébrales : classification et détection")
st.warning("**Projet pédagogique : ce n'est pas un outil de diagnostic médical.** Les résultats ne remplacent "
           "jamais l'avis d'un professionnel de santé.")


@st.cache_resource(show_spinner="Chargement du classifieur...")
def get_classifier():
    return load_classifier(MODELS / "classifier_final.keras")


@st.cache_resource(show_spinner="Chargement du détecteur...")
def get_detector():
    return load_detector(MODELS / "yolo_best.pt")


def read_json(name):
    p = METRICS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


tab_pred, tab_metrics = st.tabs(["Analyser une image", "Performances des modèles"])

with tab_pred:
    conf = st.sidebar.slider("Seuil de confiance du détecteur", 0.1, 0.9, 0.5, 0.05,
                             help="0,50 : valeur choisie sur la validation. Plus bas = plus de boîtes (et de fausses alarmes).")
    up = st.file_uploader("Image IRM (jpg ou png)", type=["jpg", "jpeg", "png"])
    if up is not None:
        img = Image.open(up)
        c1, c2 = st.columns(2)
        label, probs = None, None
        try:
            label, probs = classify(get_classifier(), img, CLASSES, cfg["data"]["image_size"], cfg["preprocessing"]["pad_mode"])
        except Exception as e:
            c1.error(f"Classifieur indisponible : {e}")
        boxes = None
        try:
            boxes = detect(get_detector(), img, conf, cfg["yolo"]["imgsz"])
        except Exception as e:
            c2.error(f"Détecteur indisponible : {e}")
        if probs:
            c1.subheader(f"Catégorie : {label} ({probs[label]:.0%})")
            c1.image(img.convert("RGB"), width='stretch')
            c1.bar_chart(pd.Series(probs, name="probabilité"))
        if boxes is not None:
            c2.subheader(f"Boîtes détectées : {len(boxes)}")
            c2.image(draw_boxes(img, boxes), width='stretch')
            if boxes:
                c2.dataframe(pd.DataFrame([(n, round(s, 3)) for n, s, _ in boxes], columns=["type", "score"]), hide_index=True)
        if probs and boxes is not None:
            if label == "No Tumor" and boxes:
                st.info("Les deux modèles ne sont pas d'accord : le classifieur voit « No Tumor » mais le détecteur propose une boîte.")
            elif label != "No Tumor" and not boxes:
                st.info("Aucune boîte à ce seuil de confiance : essayez de le baisser (barre latérale).")
    else:
        st.write("Téléversez une image pour lancer l'analyse. Les modèles sont chargés une seule fois.")

with tab_metrics:
    st.caption("Évaluation finale sur le jeu de test, calculée une seule fois ; validation entre parenthèses quand disponible.")
    clf = read_json("classifier_test.json")
    if clf:
        st.subheader("Classifieur (MobileNetV2, fine-tuning)")
        a, b = st.columns(2)
        a.metric("F1 macro (test)", f"{clf['val_f1_macro']:.3f}")
        b.metric("Accuracy (test)", f"{clf['val_accuracy']:.3f}")
        st.dataframe(pd.DataFrame({"rappel": clf["recall"], "précision": clf["precision"], "F1": clf["f1"]}).round(3))
        st.write("Matrice de confusion (lignes : vraie classe ; colonnes : prédite)")
        st.dataframe(pd.DataFrame(clf["confusion_matrix"], index=CLASSES, columns=CLASSES))
    else:
        st.info("Métriques du classifieur absentes (reports/metrics/classifier_test.json).")
    det = read_json("yolo_test.json")
    if det:
        st.subheader("Détecteur (YOLOv8n)")
        st.dataframe(pd.DataFrame(det["metriques_tout"]).set_index("classe").round(3))
        nt = det.get("no_tumor")
        if nt:
            st.write(f"Fausses alarmes sur les images sans tumeur : {nt['avec_fausse_alarme']} sur {nt['images_no_tumor']}.")
    else:
        st.info("Métriques du détecteur absentes (reports/metrics/yolo_test.json).")