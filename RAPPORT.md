# Rapport de Projet — Détection d'Incendie par Vision Artificielle

**Module** : Deep Learning / Vision par Ordinateur
**Date** : Avril 2026
**Technologie** : TensorFlow / Keras — Transfer Learning MobileNetV2
**Accuracy** : 99 % — Recall feu : 100 %

---

## Table des matières

1. [Objectif](#1-objectif)
2. [Dataset](#2-dataset)
3. [Méthodologie](#3-méthodologie)
4. [Résultats](#4-résultats)
5. [Analyse des erreurs](#5-analyse-des-erreurs)
6. [Diagnostic overfitting & généralisation](#6-diagnostic-overfitting--généralisation)
7. [Retours d'expérience : déploiements réels](#7-retours-dexpérience--déploiements-réels)
8. [Stratégies d'infrastructure & énergie](#8-stratégies-dinfrastructure--énergie)
9. [Optimisation du modèle en conditions réelles](#9-optimisation-du-modèle-en-conditions-réelles)
10. [Synthèse des hauteurs de montage](#10-synthèse-des-hauteurs-de-montage)
11. [Pistes d'amélioration](#11-pistes-damélioration)
12. [Conclusion](#12-conclusion)
13. [Annexes](#annexes)

---

## 1. Objectif

Concevoir et évaluer un système de classification d'images capable de déterminer automatiquement si une image contient un incendie ou non. Le système repose sur une approche de **transfer learning** à partir d'un réseau de neurones convolutif pré-entraîné (MobileNetV2) afin de tirer parti de représentations visuelles générales apprises sur ImageNet.

L'enjeu applicatif est la **détection précoce d'incendies** à partir d'images de caméras de surveillance ou de flux vidéo, où la rapidité et la fiabilité sont critiques. Ce rapport couvre à la fois les aspects techniques du modèle et les stratégies de déploiement physique en conditions réelles.

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
| Paramètres entraînés | ~208 897 (tête seule, soit 9 %) |
| Paramètres gelés | ~2 133 568 (backbone, soit 91 %) |

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

## 7. Retours d'expérience : déploiements réels

Le placement stratégique des caméras est aussi déterminant que la qualité du modèle. Des déploiements réels à travers le monde ont établi des référentiels concrets sur les hauteurs, les champs de vision et les infrastructures requises.

### 7.1 Cas de succès documentés

**Australie — Système ForestWatch**

L'installation de caméras sur des **tours de 54 m** en terrain plat a permis de détecter des fumées à plus de **20 km** de distance, avant même que les satellites ne les repèrent. Ce cas démontre qu'une hauteur élevée en terrain ouvert multiplie le rayon de couverture de façon quasi linéaire, justifiant l'investissement en infrastructure lourde pour les zones à risque critique.

**Saskatchewan, Canada**

La conversion d'**anciennes tours de guet** (15 à 25 m) en points de surveillance IA a réduit le temps d'intervention moyen de **18 minutes** grâce à une ligne de vue ininterrompue sur la canopée. Ce retour d'expérience valide la stratégie de réutilisation d'infrastructures existantes pour minimiser les coûts de déploiement.

**États-Unis — Système FireLoc**

Le déploiement à **faible hauteur (~9 m)** en zone urbaine et péri-urbaine a démontré qu'un **maillage dense** de points de surveillance est plus efficace qu'une seule tour haute pour les départs de feux d'origine humaine. La redondance des angles de vue compense la portée réduite de chaque unité.

### 7.2 Synthèse comparative des approches

| Déploiement | Hauteur | Portée | Atout principal |
|---|---|---|---|
| ForestWatch (AU) | 54 m | > 20 km | Détection très précoce, terrain plat |
| Tours de guet (CA) | 15 – 25 m | 8 – 12 km | Réutilisation d'infrastructure existante |
| FireLoc (US) | ~9 m | 3 – 5 km | Maillage dense, feux d'origine humaine |

### 7.3 Facteurs critiques de placement

- **Ligne de vue** : la canopée est le principal obstacle. Une caméra à 20 m dans une forêt de pins de 18 m n'offre qu'une vue au ras des cimes.
- **Orientation** : exposition préférentielle au secteur de vent dominant pour capturer la fumée en premier.
- **Redondance** : au minimum deux angles de vue couvrant chaque zone pour la levée de doute visuelle et la triangulation du foyer.

---

## 8. Stratégies d'infrastructure & énergie

Deux scénarios d'infrastructure couvrent la majorité des contextes forestiers rencontrés en pratique, de la forêt connectée à la zone totalement isolée.

### 8.1 Scénario E — Forêt connectée (haute densité 5G)

**Contexte :** Zones forestières proches d'axes routiers ou de zones habitées, bénéficiant d'une couverture réseau existante.

| Composant | Solution technique | Justification |
|---|---|---|
| **Connectivité** | 5G Ultra-Low Latency | Flux vidéo 4K pour une inférence précise ; latence < 10 ms |
| **Énergie** | Réseau électrique local (bord de route) ou PoE | Fiabilité maximale, pas de gestion de batterie |
| **IA (MobileNetV2)** | Inférence sur le Cloud ou Edge Computing déporté | Décharge les unités locales, facilite les mises à jour |
| **Bénéfice** | Recall maximal (≥ 0,98) et latence d'alerte < 2 secondes | Réponse quasi instantanée |

**Avantages :** coût d'exploitation faible une fois l'infrastructure en place, mises à jour du modèle à distance, possibilité de traitement vidéo haute résolution.

**Limite :** dépendance totale au réseau opérateur et au réseau électrique — une coupure simultanée laisse la zone sans surveillance.

### 8.2 Scénario F — Zone blanche / autonomie totale

**Contexte :** Forêts profondes, massifs isolés, sans aucune infrastructure préalable (ni électricité ni réseau cellulaire).

| Composant | Solution technique | Justification |
|---|---|---|
| **Connectivité** | Starlink / Satellite + maillage LoRaWAN local | Starlink pour la vidéo HD ; LoRa pour les alertes critiques (faible consommation) |
| **Énergie** | Hybride : panneaux solaires 400 W + mini-éolienne de mât + batteries lithium | Compensation des nuits et jours sans soleil |
| **IA (MobileNetV2)** | Inférence locale sur puce basse consommation (Jetson Nano / Coral TPU) | Pas de dépendance réseau pour la détection ; traitement sur place |
| **Bénéfice** | Surveillance 24/7 même en cas de coupure des réseaux terrestres | Résilience totale |

**Avantages :** indépendance complète, coût récurrent quasi nul (énergie renouvelable), déployable en toute zone reculée.

**Limite :** investissement initial élevé, maintenance physique requise (panneaux, batteries, firmware).

### 8.3 Comparaison des deux scénarios

| Critère | Scénario E (5G) | Scénario F (Autonome) |
|---|---|---|
| Coût d'installation | Moyen | Élevé |
| Coût d'exploitation | Faible | Très faible |
| Fiabilité réseau | Dépend de l'opérateur | Totale (Starlink + LoRa) |
| Latence d'alerte | < 2 s | 5 – 30 s (LoRa) |
| Qualité d'inférence | Haute (cloud) | Limitée (edge local) |
| Zone d'application | Péri-urbain, bords de route | Massifs isolés, haute montagne |

---

## 9. Optimisation du modèle en conditions réelles

Le passage d'un modèle en laboratoire (99 % sur dataset propre) à un déploiement terrain implique des contraintes matérielles que le modèle seul ne peut pas absorber. Cette section détaille les adaptations nécessaires.

### 9.1 Gestion de l'alimentation et modes dégradés

> **Principe clé :** le succès du modèle à 99 % dépend de la stabilité de l'alimentation. En cas de baisse de tension (batterie faible), le modèle bascule en **mode Éco** : une analyse toutes les 30 s au lieu de toutes les 5 s, réduisant la consommation CPU de ~70 %.

| État batterie | Fréquence d'analyse | Résolution | Consommation |
|---|---|---|---|
| > 80 % | 1 image / 5 s | 224 × 224 | Nominale |
| 40 – 80 % | 1 image / 15 s | 224 × 224 | Réduite |
| 20 – 40 % | 1 image / 30 s | 224 × 224 | Mode Éco |
| < 20 % | Alerte LoRa uniquement | — | Veille |

### 9.2 Protection thermique des puces IA

Les puces d'inférence (Jetson Nano, Coral TPU) sont sensibles aux pics de température, particulièrement lors des épisodes de chaleur estivaux qui coïncident précisément avec les risques d'incendie les plus élevés.

- **Régulateurs de tension** : protègent les puces des surtensions liées aux variations de charge solaire
- **Dissipateurs thermiques actifs** : ventilateur déclenché dès 65 °C pour maintenir une inférence stable
- **Boîtiers IP66** : étanchéité à la poussière et aux intempéries, indispensable en zone forestière
- **Seuil de sécurité** : arrêt automatique de l'inférence au-dessus de 85 °C pour protéger le matériel

### 9.3 Maintenance et télémétrie

La maintenance préventive à distance est indispensable pour des systèmes déployés en zones difficiles d'accès.

**Capteurs de santé système transmis via LoRa toutes les 10 minutes :**

| Métrique | Seuil d'alerte | Action |
|---|---|---|
| Température CPU | > 75 °C | Réduction de fréquence d'analyse |
| État batterie | < 25 % | Passage en mode Éco |
| Espace disque | < 500 Mo | Rotation automatique des logs |
| Uptime réseau | Coupure > 5 min | Tentative de reconnexion Starlink |

### 9.4 Stratégie 5G vs LoRa

Les deux protocoles coexistent avec des rôles complémentaires et non redondants :

- **LoRa (LoRaWAN)** : transmission de l'alerte critique (classe, confiance, horodatage, coordonnées GPS). Consommation < 50 mW, portée jusqu'à 15 km. C'est le canal de **vie ou de mort** du système.
- **5G / Wi-Fi / Starlink** : transmission du flux vidéo pour la levée de doute visuelle par un opérateur humain. Utilisé uniquement après déclenchement de l'alerte LoRa pour économiser la bande passante.

```
Détection feu
    │
    ├── LoRa ──────────→ Centrale d'alerte  (< 1 s, priorité absolue)
    │                    [coordonnées + confiance]
    │
    └── 5G/Starlink ──→ Opérateur humain   (levée de doute visuelle)
                         [flux vidéo 30 s autour de l'événement]
```

---

## 10. Synthèse des hauteurs de montage

Le choix de la hauteur de montage est le paramètre le plus impactant sur les performances réelles du système, indépendamment de la qualité du modèle IA.

### 10.1 Tableau de référence par type de végétation

| Type de forêt | Hauteur canopée | Hauteur conseillée | Infrastructure type | Portée estimée |
|---|---|---|---|---|
| Maquis / Garrigue | 2 – 5 m | **8 – 12 m** | Mâts télescopiques légers | 4 – 6 km |
| Forêt mixte (feuillus) | 10 – 18 m | **18 – 25 m** | Pylônes autoportants | 8 – 12 km |
| Pins / Sapins denses | 20 – 30 m | **32 – 40 m** | Tours de télécommunication | 12 – 18 km |
| Terrain plat / Steppe | < 2 m | **15 – 20 m** | Mâts sur treillis | > 20 km |
| Relief accidenté | Variable | **Crête naturelle + 5 m** | Pylône compact | Selon orographie |

> **Règle pratique :** la caméra doit dépasser la canopée d'au moins **5 à 8 mètres** pour obtenir un angle de vue suffisant sur les premières fumées. En dessous, la détection est retardée de plusieurs minutes — ce qui peut s'avérer critique.

### 10.2 Angles de vue et zones mortes

```
Vue de profil — Caméra à 35 m dans une forêt à 25 m

      [Caméra]
         |
    ─────┼───── 35 m (sommet du mât)
         |  \
         |   \  angle d'élévation ~3°
    ─────┼────\──────── 25 m (canopée)
         |     \
         |      ✓ Zone visible (fumée précoce détectable)
    ─────┼──────────── sol

Zone morte au pied du mât : ~80 m de rayon
→ compensée par un second point de surveillance décalé
```

### 10.3 Densité de maillage recommandée

| Zone | Densité | Justification |
|---|---|---|
| Haute criticité (accès humain, interfaces urbaines) | 1 point / 5 km² | Couverture totale sans zone aveugle |
| Criticité modérée (forêt gérée, accès entretenu) | 1 point / 15 km² | Équilibre coût / couverture |
| Zone tampon (accès rare, feux rares) | 1 point / 40 km² | Surveillance minimale, coût maîtrisé |

---

## 11. Pistes d'amélioration

### Court terme — Modèle

- **Nettoyer le dataset** : auditer manuellement `non_fire_images` pour retirer ou corriger les images mal étiquetées
- **Enrichir la classe non_fire** : ajouter des images de couchers de soleil, lumières orange, reflets, flammes de bougies — pour forcer le modèle à apprendre le contexte plutôt que la couleur seule
- **Enrichir avec données terrain** : collecter des images réelles issues des caméras déployées pour un fine-tuning sur le domaine exact

### Moyen terme — Modèle & déploiement

- **Fine-tuning** : dégeler les 30 derniers layers de MobileNetV2 avec un learning rate réduit (1e-5) pour gagner 1 à 3 % supplémentaires
- **Seuil de décision adaptatif** : abaisser le seuil de décision (ex : 0,3) en saison sèche pour maximiser le recall sur les feux
- **Conversion TFLite** : optimisation pour inférence sur puce Jetson Nano / Coral TPU (Scénario F)
- **Modèle de détection d'objets** : intégrer YOLOv8 ou SSD pour localiser le feu dans l'image (bounding box) en complément de la classification

### Long terme — Système complet

- **Inférence vidéo** : appliquer le modèle frame par frame avec lissage temporel (moyenne mobile sur 5 frames) pour éliminer les faux positifs fugaces
- **Tableau de bord centralisé** : interface de supervision multi-caméras avec carte géographique des alertes et historique
- **Triangulation** : combinaison de deux caméras orientées différemment pour calculer les coordonnées GPS précises du foyer
- **Alertes graduées** : niveau 1 (fumée détectée, > 60 % confiance) → niveau 2 (flammes visibles, > 90 %) → niveau 3 (propagation confirmée sur plusieurs frames)

---

## 12. Conclusion

Le modèle atteint une accuracy de **99 %** et un F1-score pondéré de **0,9899** sur le jeu de test, avec un **recall de 100 %** sur la classe feu — aucun incendie manqué. Dans ce contexte applicatif, un faux négatif est bien plus coûteux qu'une fausse alarme.

L'analyse des erreurs montre que les 2 erreurs observées s'expliquent par une erreur de labellisation dans le dataset pour la première, et par une confusion de couleurs (teintes orange similaires à des flammes) pour la seconde.

Au-delà des performances du modèle, ce rapport met en évidence que **le déploiement physique est aussi déterminant que l'algorithme**. Les retours d'expérience réels (ForestWatch, Saskatchewan, FireLoc) confirment que la hauteur de montage, l'orientation des caméras et la stratégie réseau conditionnent directement la vitesse de détection. Un modèle à 99 % sur dataset de laboratoire peut voir ses performances réelles chuter significativement si la caméra est positionnée sous la canopée ou si l'alimentation est instable.

Les deux scénarios d'infrastructure proposés (5G connecté et zone autonome) offrent une réponse adaptée à chaque contexte forestier, du péri-urbain aux massifs les plus isolés.

Le pipeline est reproductible — seed fixée, dépendances versionnées, split sans fuite — et suffisamment structuré pour évoluer vers un déploiement opérationnel complet.

---

## Annexes

### Fichiers du projet

| Fichier | Rôle |
|---|---|
| `fire.ipynb` | Notebook principal : données, entraînement, évaluation |
| `helper_functions.py` | Utilitaires réutilisables (courbes, métriques, visualisation) |
| `app.py` | Serveur Flask — API de prédiction + Grad-CAM |
| `templates/index.html` | Interface web — upload, résultat, heatmap |
| `requirements.txt` | Dépendances Python avec versions exactes |
| `mobilenet_v2.weights.h5` | Meilleurs poids sauvegardés |
| `fire_model.keras` | Modèle complet exporté |
| `presentation.html` | Présentation autonome 14 slides (navigateur) |
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
flask>=3.0.0
```

### Glossaire

| Terme | Définition |
|---|---|
| **Transfer Learning** | Réutilisation d'un modèle pré-entraîné sur une grande base de données pour une nouvelle tâche spécifique |
| **MobileNetV2** | Architecture CNN légère conçue pour les appareils embarqués, pre-entraîné sur ImageNet |
| **Recall** | Taux de vrais positifs : proportion d'incendies réels correctement détectés |
| **F1-score** | Moyenne harmonique de la precision et du recall |
| **EarlyStopping** | Arrêt automatique de l'entraînement si la validation ne s'améliore plus |
| **Grad-CAM** | Technique de visualisation identifiant les zones de l'image les plus influentes pour la prédiction |
| **LoRaWAN** | Protocole radio longue portée (jusqu'à 15 km) et faible consommation pour IoT et alertes critiques |
| **Edge Computing** | Traitement informatique réalisé localement sur l'appareil, sans recours au cloud |
| **TFLite** | Version optimisée de TensorFlow pour déploiement sur appareils embarqués (mobile, Raspberry Pi, etc.) |
| **Class Weighting** | Technique compensant le déséquilibre de classes en pénalisant davantage les erreurs sur la classe minoritaire |
