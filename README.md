# Fire Classification - Vision AI (TensorFlow)

Projet de classification d'images avec TensorFlow/Keras pour distinguer deux classes: feu et non-feu.

Etat actuel du repo:
- entrainement principal via notebook Jupyter: `fire.ipynb`
- backbone: `MobileNetV2` en transfer learning
- sortie binaire avec une couche finale `softmax` sur 2 classes

## Structure du projet

```text
Ml-fire/
  fire.ipynb
  helper_functions.py
  fire_dataset/
    fire_images/
    non_fire_images/
  mobilenet_v2.weights.h5
  training_logs/
```

Description rapide:
- `fire.ipynb`: preparation des donnees, entrainement, evaluation et visualisations
- `helper_functions.py`: fonctions utilitaires pour les images, les courbes et la matrice de confusion
- `fire_dataset/`: dataset local utilise par le notebook
- `mobilenet_v2.weights.h5`: meilleurs poids sauvegardes pendant l'entraînement
- `training_logs/`: logs TensorBoard

## Objectif

Ce projet sert de base pour:
1. Detecter la presence de feu sur une image.
2. Mesurer ensuite la performance du modele sur un jeu de test local.
3. Evoluer plus tard vers de l'inference sur une image unique ou un flux video si besoin.

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
pip install tensorflow pandas numpy matplotlib scikit-learn pillow jupyter
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

Le code actuel utilise un split train/test puis un sous-split validation sur le train. Il est adapte a une classification binaire en deux dossiers.

## Execution

1. Ouvrir `fire.ipynb`.
2. Verifier que la variable `dataset` pointe vers `./fire_dataset` ou vers ton chemin local.
3. Executer les cellules dans l'ordre.
4. Laisser `EarlyStopping` arreter l'entrainement si la validation ne s'ameliore plus.

## Sorties generees

- Courbes d'apprentissage loss et accuracy
- Evaluation sur le jeu de test
- `classification_report`
- matrice de confusion
- poids sauvegardes dans `mobilenet_v2.weights.h5`
- logs TensorBoard dans `training_logs/`

Pour visualiser TensorBoard:

```bash
tensorboard --logdir training_logs
```

## Bonnes pratiques

- Verifier l'equilibre entre les classes avant l'entrainement.
- Conserver un split train/val/test stable pour comparer les essais.
- Eviter les fuites de donnees entre train et test.
- Surveiller precision, rappel et F1, pas seulement l'accuracy.

## Notes

- Le fichier `helper_functions.py` contient des fonctions utilitaires reutilisables pour les notebooks TensorFlow.
- Le projet est actuellement centre sur la classification binaire, pas sur la regression.

Si tu veux, je peux aussi te corriger le notebook pour le rendre plus robuste sur le split des classes et la compatibilite PNG/JPG.
