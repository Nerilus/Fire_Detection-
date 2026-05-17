# Fire Detection — Vision IA (TensorFlow / MobileNetV2)

Classification d'images feu / non-feu par transfer learning.
Accuracy test : **99 %** — Recall feu : **100 %** (0 faux négatif sur 200 images de test).

## Structure du projet

```text
Fire_Detection-/
  fire.ipynb            # pipeline complet : données → entraînement → évaluation
  app.py                # serveur Flask : API /predict + Grad-CAM
  helper_functions.py   # utilitaires : courbes, matrice de confusion, métriques
  requirements.txt      # dépendances avec versions fixées
  presentation.html     # présentation interactive 30 slides (16:9)
  RAPPORT.md            # rapport technique complet
  templates/
    index.html          # interface web drag-and-drop
  fire_dataset/
    fire_images/        # 755 images de feu
    non_fire_images/    # 244 images sans feu
  mobilenet_v2.weights.h5   # meilleurs poids (EarlyStopping)
  fire_model.keras          # modèle complet sauvegardé
  training_logs/            # logs TensorBoard
```

## Modèle

**Architecture** : MobileNetV2 pré-entraîné (ImageNet) + tête Dense(256, ReLU) + Dropout(0.2) + Dense(1, sigmoid)

| Composant | Paramètres | Statut |
|-----------|-----------|--------|
| MobileNetV2 backbone | 2 334 720 | Gelés |
| Tête de classification | 208 897 | Entraînés |
| **Total** | **2 543 617** | |

**Pipeline entraînement** : class weighting (fire=0.44 / non\_fire=1.37), augmentation à la volée (rotation ±20°, flip, zoom ±10 %), EarlyStopping patience=5, ModelCheckpoint sur val\_loss.

**Résultats test (200 images)** :

| Métrique | Valeur |
|----------|--------|
| Accuracy | 99 % |
| Recall feu | 100 % |
| Faux négatifs | 0 |
| Fausses alarmes | 2 |

## Application web (Flask)

Le modèle est exposé via un serveur Flask avec interface drag-and-drop.

```bash
python app.py
# → http://localhost:5000
```

**Fonctionnalités** :
- Upload par glisser-déposer ou clic
- Prédiction en temps réel (< 1 s) avec score de confiance
- Visualisation **Grad-CAM** interactive (heatmap des zones décisives)
- Réponse JSON : `is_fire`, `confidence`, `raw_score`, `gradcam_data`

**Routes** :
- `GET /` — interface HTML
- `POST /predict` — reçoit une image, retourne JSON + Grad-CAM base64

## Présentation interactive

`presentation.html` — 30 slides 16:9, ouvrir directement dans un navigateur.

Navigation : flèches clavier ou boutons. Toggle **Détails / Présentation** (haut gauche) pour basculer entre la vue technique complète et un résumé simplifié.

Contenu :
- Slides 1–14 : projet, CNN, transfer learning, architecture, dataset, entraînement, résultats, erreurs, Grad-CAM, app web, améliorations, conclusion
- Slide 15 : séparateur fiches techniques
- Slides 16–26 : TensorFlow, Keras, MobileNetV2, ImageNet, NumPy/Pandas, Matplotlib/scikit-learn, Pillow, Flask, Grad-CAM mathématiques, TFLite/Edge, LoRaWAN/5G/Starlink
- Slides 27–30 : concurrents mondiaux (ForestWatch AU, Saskatchewan CA, FireLoc US) + tableau comparatif

## Prérequis

- Python 3.10+
- GPU optionnel (CPU suffisant pour l'inférence)

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
# ou : source .venv/bin/activate   (Linux / macOS)

pip install --upgrade pip
pip install -r requirements.txt
```

## Entraînement (notebook)

```bash
jupyter notebook fire.ipynb
```

1. Vérifier que `dataset` pointe vers `./fire_dataset`
2. Exécuter les cellules dans l'ordre
3. Phase 2 (fine-tuning, optionnel) : dégèle les 30 derniers layers de MobileNetV2 avec lr=1e-5

## Sorties générées

- `fire_model.keras` — modèle complet
- `mobilenet_v2.weights.h5` — meilleurs poids
- `training_logs/` — courbes TensorBoard
- Matrice de confusion + `classification_report` dans le notebook

```bash
tensorboard --logdir training_logs
```

## Bonnes pratiques

- Split train/val/test fixé avant tout traitement (`random_state=42`) — intersection train ∩ test = 0
- Vider les sorties du notebook avant chaque commit (`Kernel > Clear All Outputs`)
- Surveiller recall et F1, pas seulement l'accuracy — un faux négatif (feu manqué) est critique
