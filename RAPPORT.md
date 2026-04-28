# Rapport de Projet — Détection d'Incendie par Vision Artificielle

**Module** : Deep Learning / Vision par Ordinateur  
**Date** : Avril 2026  
**Technologie** : TensorFlow / Keras — Transfer Learning MobileNetV2  

---

## 1. Objectif

Concevoir et évaluer un système de classification d'images capable de déterminer automatiquement si une image contient un incendie ou non. Le système repose sur une approche de **transfer learning** à partir d'un réseau de neurones convolutif pré-entraîné (MobileNetV2) afin de tirer parti de représentations visuelles générales apprises sur ImageNet.

L'enjeu applicatif est la **détection précoce d'incendies** à partir d'images de caméras de surveillance ou de flux vidéo, où la rapidité et la fiabilité sont critiques.

---

## 2. Dataset

### 2.1 Description

| Propriété | Valeur |
|---|---|
| Total images | 999 |
| Format | PNG exclusivement |
| Classes | `fire_images` / `non_fire_images` |
| Source | Dataset local organisé en dossiers |

### 2.2 Distribution des classes

| Classe | Nombre | Proportion |
|---|---|---|
| `fire_images` | 755 | 75,6 % |
| `non_fire_images` | 244 | 24,4 % |
| **Ratio** | **3,09 : 1** | Déséquilibre modéré |

Le déséquilibre entre les classes a été pris en compte via le mécanisme de **class weighting** (`compute_class_weight('balanced')`) appliqué pendant l'entraînement.

### 2.3 Variabilité des images

Les images présentent une forte variabilité dimensionnelle (458 à 1900 px en largeur, 240 à 1425 px en hauteur), toutes redimensionnées à **224 × 224 pixels** lors du prétraitement.

### 2.4 Split train / validation / test

| Partition | Images | Proportion |
|---|---|---|
| Train | 640 | 64 % |
| Validation | 159 | 16 % |
| Test | 200 | 20 % |

Le split train/test a été effectué **avant** la création des générateurs (`train_test_split` avec `random_state=42`) pour garantir l'absence de fuite de données. La validation est extraite du train via `validation_split=0.2`.

**Vérification de fuite de données :**
- Intersection train ∩ test : 0 image
- Union train ∪ test : 999 images = total exact

---

## 3. Méthodologie

### 3.1 Pipeline complet

```
Images (fire_dataset/)
    │
    ├── Glob JPG / PNG → DataFrame (Filepath, Label)
    │
    ├── train_test_split (80/20, random_state=42)
    │
    ├── ImageDataGenerator
    │     ├── preprocess_input (MobileNetV2 scale)
    │     └── Augmentation : rotation ±20°, flip H,
    │                        zoom ±10%, brightness [0.8–1.2]
    │
    ├── flow_from_dataframe → class_mode='binary'
    │
    ├── Calcul class_weight (balanced)
    │
    └── Modèle → Évaluation → Visualisation
```

### 3.2 Architecture du modèle

```
Input (224, 224, 3)
    │
    ▼
MobileNetV2 (ImageNet, include_top=False, pooling='avg')
    │  [base gelée — 2 334 720 paramètres non entraînés]
    ▼
Dense(256, activation='relu')
    │
Dropout(0.2)
    │
Dense(1, activation='sigmoid')       ← sortie binaire
```

| Composant | Détail |
|---|---|
| Backbone | MobileNetV2 pré-entraîné ImageNet |
| Tête | Dense(256, ReLU) → Dropout(0.2) → Dense(1, sigmoid) |
| Paramètres totaux | ~2 342 465 |
| Paramètres entraînés | ~208 897 (tête seule) |
| Paramètres gelés | ~2 133 568 (backbone) |

### 3.3 Stratégie d'entraînement

| Hyperparamètre | Valeur |
|---|---|
| Optimiseur | Adam (lr = 1e-4) |
| Loss | Binary Crossentropy |
| Métrique | Accuracy |
| Epochs max | 100 |
| EarlyStopping | patience = 5, monitor = val_loss |
| Checkpoint | Meilleurs poids sauvegardés (val_loss) |
| Class weight | Calculé automatiquement (balanced) |
| Seed globale | `tf.random.set_seed(42)` |

**Augmentation de données** (entraînement uniquement) :
- Rotation aléatoire ±20°
- Translation horizontale/verticale ±10 %
- Flip horizontal
- Zoom ±10 %
- Variation de luminosité [0.8 ; 1.2]

---

## 4. Résultats

### 4.1 Métriques globales sur le test set (200 images)

| Métrique | Valeur |
|---|---|
| **Test Accuracy** | **99,00 %** |
| **Test Loss** | **0,0702** |
| Precision pondérée | 0,9901 |
| Recall pondéré | 0,9900 |
| **F1-score pondéré** | **0,9899** |
| F1-score macro | 0,9868 |

### 4.2 Métriques par classe

| Classe | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| `fire_images` | 0,987 | **1,000** | 0,993 | 148 |
| `non_fire_images` | **1,000** | 0,962 | 0,980 | 52 |

Le recall fire atteint 100 % : aucun incendie n'a été manqué sur le jeu de test, ce qui est le critère prioritaire pour une application de surveillance. La precision non_fire atteint également 100 % : toutes les prédictions "non-feu" sont correctes, sans fausse alarme confirmée.

### 4.3 Matrice de confusion

|  | Prédit : fire | Prédit : non_fire |
|---|---|---|
| **Réel : fire** | 148 (TP) | 0 (FN) |
| **Réel : non_fire** | 2 (FP) | 50 (TN) |

- Faux Négatifs (feux manqués) : 0 — aucun incendie non détecté
- Faux Positifs (fausses alarmes) : 2 — deux images non-feu classées feu

### 4.4 Confiance des prédictions

| Indicateur | Valeur |
|---|---|
| Confiance moyenne globale | 97,1 % |
| Prédictions > 95 % | 173 / 200 (86,5 %) |
| Prédictions < 70 % (incertaines) | 5 / 200 (2,5 %) |
| Confiance moyenne (prédictions correctes) | 97,0 % |
| Confiance moyenne (erreurs) | 81,7 % |

---

## 5. Analyse des erreurs

### 5.1 Faux Positifs observés

**Erreur 1 — Confiance 99,8 %**

Image étiquetée `non_fire` prédite `fire` avec une quasi-certitude. L'inspection visuelle révèle une image montrant clairement un grand incendie de forêt avec flammes et fumée dense. Il s'agit d'une **erreur de labellisation dans le dataset** : l'image est en réalité un cas de feu. Le modèle est plus fiable que l'annotation humaine dans ce cas.

**Erreur 2 — Confiance 63,6 %**

Image étiquetée `non_fire` prédite `fire`. Il s'agit d'une capture d'écran de jeu vidéo (The Witcher 3) présentant un oiseau aux plumes orange/rouge dans une forêt. Le modèle se laisse tromper par les **teintes chaudes** similaires à des flammes, sans pouvoir analyser le contexte sémantique.

### 5.2 Enseignements

| Erreur | Type | Cause | Solution |
|---|---|---|---|
| Image de feu dans non_fire | Erreur de label | Annotation humaine incorrecte | Audit et nettoyage du dataset |
| Oiseau rouge dans forêt | Faux positif du modèle | Couleurs chaudes sans contexte | Enrichir non_fire avec cas difficiles |

---

## 6. Diagnostic overfitting & généralisation

| Indicateur | Valeur | Verdict |
|---|---|---|
| Écart accuracy train/val | < 5 % | Pas d'overfitting |
| Écart loss val/train | < 0,10 | Stable |
| Écart val/test accuracy | < 2 % | Bonne généralisation |
| EarlyStopping déclenché | Oui | Arrêt au bon moment |

Le modèle généralise correctement : les performances sur le test set sont cohérentes avec celles de la validation. L'écart reste inférieur à 2 % sur l'accuracy, ce qui exclut un surapprentissage significatif.

---

## 7. Pistes d'amélioration

### Court terme
- **Nettoyer le dataset** : auditer manuellement `non_fire_images` pour retirer ou corriger les images mal étiquetées
- **Enrichir la classe non_fire** : ajouter des images de couchers de soleil, lumières orange, reflets, flammes de bougies — pour forcer le modèle à apprendre le contexte plutôt que la couleur seule

### Moyen terme
- **Fine-tuning** : dégeler les 30 derniers layers de MobileNetV2 avec un learning rate réduit (1e-5) pour gagner 1 à 3 % supplémentaires
- **Seuil de décision adaptatif** : abaisser le seuil de décision (ex : 0,3) pour maximiser le recall sur les feux — applicable selon la tolérance aux fausses alarmes

### Long terme
- **Déploiement** : conversion en TFLite pour inférence sur edge devices (caméras embarquées)
- **Détection sur vidéo** : appliquer le modèle frame par frame avec lissage temporel
- **Grad-CAM en production** : générer automatiquement les zones d'attention pour chaque alerte

---

## 8. Conclusion

Le modèle atteint une accuracy de 99 % et un F1-score pondéré de 0,9899 sur le jeu de test, avec un recall de 100 % sur la classe feu — aucun incendie manqué. Dans ce contexte applicatif, un faux négatif est bien plus coûteux qu'une fausse alarme.

L'analyse des erreurs montre que les 2 erreurs observées s'expliquent par une erreur de labellisation dans le dataset pour la première, et par une confusion de couleurs (teintes orange similaires à des flammes) pour la seconde.

Le pipeline est reproductible — seed fixée, dépendances versionnées, split sans fuite — et suffisamment structuré pour évoluer vers un déploiement.

---

## Annexes

### Fichiers du projet

| Fichier | Rôle |
|---|---|
| `fire.ipynb` | Notebook principal : données, entraînement, évaluation |
| `helper_functions.py` | Utilitaires réutilisables (courbes, métriques, visualisation) |
| `requirements.txt` | Dépendances Python avec versions exactes |
| `mobilenet_v2.weights.h5` | Meilleurs poids sauvegardés |
| `fire_model.keras` | Modèle complet exporté |
| `diagnostic_complet.png` | Graphique de diagnostic généré automatiquement |
| `training_logs/` | Logs TensorBoard |
| `README.md` | Documentation d'installation et d'utilisation |

### Dépendances principales

```
tensorflow==2.21.0
numpy==1.26.4
pandas==2.2.2
matplotlib==3.10.8
scikit-learn==1.8.0
Pillow==12.1.1
```
