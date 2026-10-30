# Classification et détection de tumeurs sur IRM cérébrales

> **Projet pédagogique : ce n'est pas un outil de diagnostic médical.** Les résultats ne remplacent jamais l'avis d'un professionnel de santé.

IRM cérébrales en 4 classes (Glioma, Meningioma, No Tumor, Pituitary) avec boîtes englobantes au format YOLO.
Deux modèles : un **classifieur** (MobileNetV2, TensorFlow) et un **détecteur** (YOLOv8n, Ultralytics), un pipeline **Airflow** et une application **Streamlit**, le tout dans Docker.

## Résultats (jeu de test, évalué une seule fois)

| Modèle | Mesure | Test | Validation |
|---|---|---|---|
| Classifieur (545 images) | F1 macro / accuracy | **0,886 / 0,883** | 0,869 / 0,859 |
| | Rappel Glioma / Meningioma / No Tumor / Pituitary | 0,85 / 0,82 / 0,89 / 0,99 | 0,83 / 0,75 / 0,94 / 0,98 |
| Détecteur (491 images, 573 boîtes) | mAP50 / mAP50-95 | **0,949 / 0,732** | 0,942 / 0,726 |
| | Précision / rappel (seuil 0,50) | 0,92 / 0,89 | 0,91 / 0,90 |
| | Fausses alarmes sur 54 images No Tumor | 6 (11 %) | 4 (7 %) |

Le test n'a servi qu'une fois, après le choix des modèles et du seuil sur la validation (les scripts refusent de le relancer).
Comparaison des configurations du classifieur (validation, 3 graines) : base gelée 0,79-0,81 ; **fine-tuning 0,862 ± 0,011** ; fine-tuning + régularisation forte 0,848 ± 0,002.

## Méthode

- **Données** : 5 249 images. Des copies quasi identiques (jumeaux) sont regroupées (3 642 groupes) ; un groupe n'est jamais coupé entre apprentissage, validation et test (fuite de données évitée). Découpage 70/15/15, graine 42, validation et test : une image par groupe.
- **Manifeste** `manifests/manifest.csv` : une ligne par image, toutes les décisions sont des colonnes (rien n'est supprimé sur le disque). 15 images aux étiquettes en conflit sont exclues.
- **Prétraitement** : niveaux de gris sur 3 canaux, mise au carré avec bandes de la couleur du fond, 224×224 ; normalisation dans le modèle.
- **Augmentations** : rotation, décalage, zoom, contraste, luminosité ; coins vides remplis avec la couleur du fond. Fonctions de transformation des boîtes testées (`src/augmentation.py`).
- **Classifieur** : MobileNetV2 (ImageNet), tête entraînée puis 30 dernières couches dégelées (taux 1e-5), poids de classes, arrêt sur la perte de validation. Grad-CAM pour l'analyse.
- **Détecteur** : YOLOv8n, 640 px, 3 classes (Glioma 0, Meningioma 1, Pituitary 2), seuil de confiance 0,50 choisi sur la validation.
- **Airflow** : DAG `pipeline_irm` : vérification du manifeste → prétraitement → entraînement léger → évaluation (validation) → sauvegarde ; nouvel essai, échec si le manifeste est invalide ou si le F1 est sous 0,6.
- **Streamlit** : image téléversée → catégorie et probabilités, boîtes et scores (seuil réglable), métriques ; modèles chargés une seule fois.

## Structure

```
configs/config.yaml    paramètres (graine, chemins, modèles, seuils)
src/                   code (données, manifeste, prétraitement, augmentation, classifieur, YOLO, inférence, pipeline)
scripts/               train_yolo.py, eval_yolo.py (terminal)
dags/pipeline_dag.py   DAG Airflow
app/streamlit_app.py   application
tests/                 tests automatiques
manifests/             manifeste et rapport de découpage (suivis par Git)
pipeline.ipynb         notebook : toutes les étapes, dans l'ordre
data/ models/ reports/ non versionnés (données, modèles, métriques)
```

## Installation

```bash
git clone <dépôt> && cd IRM_detection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Placer les données dans `data/raw/` (structure `<Split>/<Classe>/{images,labels}/`). GPU NVIDIA (WSL2) : la première cellule du notebook charge les bibliothèques CUDA 12 pour TensorFlow ; YOLO s'entraîne dans un terminal séparé.

## Exécution

```bash
# Notebook : ouvrir pipeline.ipynb et tout exécuter (le test refuse de se recalculer : il lit le résultat existant)
python scripts/train_yolo.py                     # entraîne YOLO -> models/yolo_best.pt
python scripts/eval_yolo.py --split val --sweep  # choix du seuil de confiance (validation)
python scripts/eval_yolo.py --split test --conf 0.5   # UNE SEULE FOIS

echo "AIRFLOW_UID=$(id -u)" > .env
docker compose build && docker compose up -d
docker compose exec airflow airflow dags unpause pipeline_irm
docker compose exec airflow airflow dags trigger pipeline_irm          # ajouter --conf '{"simulate_failure": true}' pour un échec volontaire
```

Interfaces : Airflow http://localhost:8080 (sans mot de passe, réglage de démonstration locale) ; application http://localhost:8501. Les modèles (`models/classifier_final.keras`, `models/yolo_best.pt`) ne sont pas dans Git : ils sont lus par volume.

## Tests

```bash
python -m pytest tests -q     # 68 tests : boîtes, prétraitement, étiquettes, manifeste, inférence
```

## Limites connues

- **Raccourci de source** : en validation, la taille d'origine de l'image identifie presque la classe (Glioma 176/176 et No Tumor 0/54 en 512×512). Un petit arbre sur les seules métadonnées de fichier atteint un F1 de 0,59, sur 6 chiffres de pixels 0,65. Le prétraitement retire en partie ce signal ; il n'est pas écarté.
- Une seule graine pour les modèles finaux ; scores de validation légèrement optimistes (choix des modèles dessus) ; écarts de moins de ~0,015 de F1 non significatifs.
- Meningioma reste la classe la plus difficile en classification (confusions avec Pituitary et Glioma).
- Les cartes Grad-CAM sont grossières (7×7) et ne prouvent pas que le modèle regarde la lésion.
- Les boîtes tournées sont plus grandes que la lésion (×1,26 en médiane) ; YOLO n'utilise pas la rotation.
- Le DAG Airflow entraîne un modèle léger sur CPU (F1 ≈ 0,74) : le modèle final a été entraîné hors Airflow. YOLO n'est pas dans le DAG. L'entraînement CPU n'est pas parfaitement reproductible.
- Non testés : réseaux plus grands (ResNet50), comparaison `train_dedup`, retournement gauche-droite pour la classification.