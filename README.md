# Fire Classification - Vision AI (TensorFlow)

Projet de classification d'images avec TensorFlow/Keras pour distinguer deux classes: feu et non-feu.

Pipeline: `fire.ipynb` — MobileNetV2 en transfer learning, sortie binaire `Dense(1, sigmoid)`, gestion du desequilibre via `class_weight`, augmentation dans le generateur d'entrainement.

## Structure du projet

```text
Fire_Detection-/
  fire.ipynb
  helper_functions.py
  requirements.txt
  fire_dataset/
    fire_images/
    non_fire_images/
  mobilenet_v2.weights.h5
  fire_model.keras
  training_logs/
```

Description rapide:
- `fire.ipynb`: preparation des donnees, entrainement, evaluation et visualisations
- `helper_functions.py`: fonctions utilitaires pour les images, les courbes et la matrice de confusion
- `requirements.txt`: dependances Python avec versions exactes
- `fire_dataset/`: dataset local utilise par le notebook
- `mobilenet_v2.weights.h5`: meilleurs poids sauvegardes pendant l'entrainement
- `fire_model.keras`: modele complet sauvegarde apres entrainement
- `training_logs/`: logs TensorBoard

## Objectif

Détecter la présence de feu sur une image et mesurer les performances du modèle sur un jeu de test local. Le projet peut évoluer vers de l'inférence sur flux vidéo.

## Prerequis

- Windows, Linux ou macOS
- Python 3.10+
- Jupyter Notebook ou VS Code avec l'extension Jupyter
- GPU optionnel pour accelerer l'entrainement

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Puis lancer Jupyter:

```bash
jupyter notebook
```

## Dataset attendu

Le notebook lit toutes les images presentes dans `fire_dataset/` et prend le nom du dossier parent comme label.

Structure attendue:

```text
fire_dataset/
  fire_images/
    image_001.png
    image_002.png
  non_fire_images/
    image_001.png
    image_002.png
```

Le code utilise un split train/test puis un sous-split validation sur le train.
Le desequilibre entre les classes (755 feu vs 244 non-feu) est compense automatiquement via `class_weight`.

## Execution

1. Ouvrir `fire.ipynb`.
2. Verifier que la variable `dataset` pointe vers `./fire_dataset` ou vers ton chemin local.
3. Executer les cellules dans l'ordre.
4. L'`EarlyStopping` arrete l'entrainement si la validation ne s'ameliore plus pendant 5 epochs.
5. La cellule de fine-tuning (Phase 2) peut etre executee apres convergence pour gagner 1-3 % supplementaires.

## Sorties generees

- Courbes d'apprentissage loss et accuracy (phases 1 et 2)
- Evaluation sur le jeu de test
- `classification_report` et metriques F1/precision/recall
- Matrice de confusion annotee
- Poids sauvegardes dans `mobilenet_v2.weights.h5`
- Modele complet dans `fire_model.keras`
- Logs TensorBoard dans `training_logs/`

Pour visualiser TensorBoard:

```bash
tensorboard --logdir training_logs
```

## Bonnes pratiques

- Conserver un split train/val/test stable pour comparer les essais (`random_state=42`, `tf.random.set_seed(42)`).
- Ne pas mélanger train et test — le split est effectué avant la création des générateurs.
- Surveiller precision, rappel et F1, pas seulement l'accuracy.
- Vider les sorties du notebook (`Kernel > Clear All Outputs`) avant chaque commit git.

## Notes

- `helper_functions.py` contient des fonctions utilitaires reutilisables: `make_confusion_matrix`, `calculate_results`, `plot_loss_curves`, `compare_historys`, `pred_and_plot`, etc.
- Le projet utilise une sortie sigmoide binaire (`Dense(1, sigmoid)`) plutot que softmax a 2 classes, ce qui est plus adapte et plus leger pour la classification binaire.
