# CNN Hôpital — Plateforme intelligente d’analyse d’imagerie médicale des plaies

## Présentation du projet

Ce projet est réalisé dans le cadre d’un cours de Deep Learning.  
L’objectif est de concevoir une plateforme intelligente d’aide à l’analyse d’images de plaies cutanées.

Le système vise à combiner plusieurs briques d’intelligence artificielle :

- un modèle de classification d’images capable d’identifier automatiquement le type de plaie ;
- un mécanisme de détection hors-domaine afin de rejeter les images non reconnues par le système ;
- une recherche par similarité visuelle pour retrouver des cas historiques proches ;
- une interface interactive Streamlit ;
- un suivi expérimental avec MLflow ;
- une surveillance de dérive des données avec Evidently AI ;
- éventuellement un assistant LLM/RAG avec traçabilité Langfuse.

Le projet suit le déroulé du sujet fourni, qui demande notamment une classification CNN, un autoencoder ou modèle équivalent pour la détection OOD, un suivi MLflow, une recherche par embeddings, une interface Streamlit, une analyse de dérive avec Evidently AI, et des extensions optionnelles comme Grad-CAM, RAG/LLM, Langfuse et Docker.

---

## Objectifs principaux

Les objectifs principaux du projet sont les suivants :

1. explorer et préparer un dataset d’images de plaies ;
2. mettre en place une stratégie de data augmentation adaptée au contexte médical ;
3. entraîner et comparer plusieurs architectures CNN ;
4. suivre les expériences avec MLflow ;
5. évaluer rigoureusement les modèles entraînés ;
6. implémenter un mécanisme de détection hors-domaine ;
7. extraire des embeddings visuels pour la recherche par similarité ;
8. proposer une interface Streamlit multi-pages ;
9. produire un rapport d’analyse complet.

---

## Structure du projet

```text
CNN_Hopital/
├── core/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model_utils.py
│   ├── inference.py
│   ├── autoencoder.py
│   ├── ood_detection.py
│   ├── pipeline.py
│   ├── grad_cam.py
│   ├── image_similarity.py
│   ├── database.py
│   ├── ollama_client.py
│   └── langfuse_client.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── splits/
│   │       ├── train.csv
│   │       ├── val.csv
│   │       └── test.csv
│   ├── ood/
│   │   ├── cat.jpg
│   │   ├── landscape.jpg
│   │   ├── xray.jpg
│   │   ├── random_object.jpg
│   │   └── healthy_skin_or_other.jpg
│   └── base_connaissances_medicales.jsonl
│
├── notebooks/
│   ├── 01_exploration_dataset.ipynb
│   ├── 02_data_augmentation.ipynb
│   ├── 03_training_cnn.ipynb
│   ├── 04_evaluation_models.ipynb
│   └── autoencoder_ood_analysis.ipynb
│
├── scripts/
│   ├── train_cnn.py
│   ├── evaluate_cnn.py
│   ├── predict_image.py
│   ├── train_autoencoder.py
│   ├── evaluate_autoencoder.py
│   ├── evaluate_autoencoder_latent_gmm.py
│   ├── analyze_image.py
│   └── drift_monitoring.py
│
├── app-streamlit/
│   ├── Home.py
│   └── pages/
│       ├── 1_Dataset_Explorer.py
│       ├── 2_Training.py
│       ├── 3_Prediction.py
│       ├── 4_Explainability.py
│       └── 5_AI_Assistant.py
│
├── models/
├── reports/
│   └── figures/
├── tests/
├── mlruns/
├── requirements.txt
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## Données

Le dataset utilisé est organisé en sous-dossiers, chaque sous-dossier correspondant à une classe de plaie.

Exemple attendu :

```text
data/raw/
├── Abrasions/
├── Bruises/
├── Burns/
├── Cut/
├── Ingrown_nails/
├── Laceration/
└── Stab_wound/
```

Les images brutes ne doivent pas être versionnées dans Git.  
Le dossier `data/raw/` est donc exclu via `.gitignore`.

Un petit jeu d’images hors-domaine est disponible pour tester le filtre OOD :

```text
data/ood/
├── cat.jpg
├── landscape.jpg
├── xray.jpg
├── random_object.jpg
└── healthy_skin_or_other.jpg
```

Ces images ne servent pas à entraîner le modèle. Elles permettent uniquement de vérifier si le pipeline rejette correctement des images qui ne correspondent pas au domaine des plaies.

---

## Installation

Créer un environnement virtuel :

```bash
python -m venv .venv
source .venv/bin/activate
```

Installer les dépendances principales :

```bash
pip install -r requirements.txt
```

Dépendances principales utilisées :

```text
torch
torchvision
numpy
pandas
matplotlib
scikit-learn
Pillow
tqdm
mlflow
streamlit
opencv-python
evidently
chromadb
sentence-transformers
```

---

## État d’avancement du projet

Le projet suit le déroulé du sujet, en commençant par la partie Deep Learning : exploration du dataset, data augmentation, rééquilibrage des classes, puis entraînement des premiers modèles CNN.

---

## 1. Exploration et préparation du dataset

Un premier notebook d’analyse exploratoire a été réalisé :

```text
notebooks/01_exploration_dataset.ipynb
```

Cette analyse a permis de vérifier la structure du dataset, organisé en sous-dossiers correspondant aux classes de plaies, et d’extraire les informations principales :

- nombre total d’images : **431** ;
- nombre de classes : **7** ;
- aucune image illisible détectée ;
- résolution médiane des images : **640 × 640 pixels** ;
- fort déséquilibre entre les classes, avec un ratio max/min de **5,3**.

### Distribution des classes

| Classe        | Nombre d’images | Pourcentage |
| ------------- | --------------: | ----------: |
| Bruises       |             122 |     28,31 % |
| Abrasions     |              85 |     19,72 % |
| Laceration    |              61 |     14,15 % |
| Burns         |              59 |     13,69 % |
| Cut           |              50 |     11,60 % |
| Ingrown_nails |              31 |      7,19 % |
| Stab_wound    |              23 |      5,34 % |

Le dataset a ensuite été séparé en trois sous-ensembles stratifiés :

```text
data/processed/splits/train.csv
data/processed/splits/val.csv
data/processed/splits/test.csv
```

Le split retenu est de type **70 % / 15 % / 15 %**, afin de conserver un jeu de validation et un jeu de test distincts tout en respectant la distribution des classes.

---

## 2. Data augmentation et rééquilibrage

Un second notebook a été créé :

```text
notebooks/02_data_augmentation.ipynb
```

Il définit une stratégie d’augmentation en ligne appliquée uniquement au jeu d’entraînement.

Les transformations retenues restent volontairement modérées afin de conserver la cohérence clinique des images :

- redimensionnement en **224 × 224 pixels** ;
- retournements horizontaux et verticaux ;
- rotations modérées ;
- légères variations de luminosité, contraste et saturation ;
- faibles translations et zooms.

Certaines transformations ont été exclues, notamment :

- l’inversion de couleurs ;
- les déformations géométriques agressives ;
- les modifications trop fortes de teinte ou de contraste.

Ces choix évitent d’altérer artificiellement l’apparence médicale des plaies.

Le notebook prépare également deux stratégies de rééquilibrage à comparer pendant l’entraînement :

- pondération de la fonction de perte avec des poids de classes ;
- échantillonnage pondéré avec `WeightedRandomSampler`.

---

## 3. Factorisation du code

Les éléments validés dans les notebooks ont commencé à être factorisés dans le dossier `core/` afin de rendre le projet plus reproductible.

### `core/data_processing.py`

Ce fichier contient notamment :

- la classe `WoundDataset` ;
- les transformations d’entraînement et d’évaluation ;
- le chargement des splits ;
- le mapping des classes ;
- le calcul des poids de classes ;
- la création des `DataLoader` ;
- le support du `WeightedRandomSampler`.

### `core/model_utils.py`

Ce fichier permet de construire plusieurs architectures CNN, avec transfer learning ou entraînement from scratch.

Architectures disponibles :

- `vgg16` ;
- `resnet50` ;
- `efficientnet_b0` ;
- `mobilenet_v3_large` ;
- `custom_cnn`.

`mobilenet_v3_large` ajoute une comparaison avec un modèle compact optimisé pour l’inférence rapide.  
`custom_cnn` est une baseline simple entraînée from scratch, utile pour comparer les résultats avec les modèles pré-entraînés ImageNet.

### `core/inference.py`

Ce fichier centralise le chargement d’un checkpoint entraîné et la prédiction sur une image unique.

Il fournit notamment :

- la détection automatique du device disponible : `mps`, `cuda` ou `cpu` ;
- le chargement du checkpoint PyTorch ;
- la reconstruction du modèle avec l’architecture et le mapping de classes sauvegardés ;
- les transformations d’inférence compatibles ImageNet ;
- la fonction `predict_image`, qui retourne la classe prédite, la confiance et le top-K des classes.

### `core/autoencoder.py`

Ce fichier contient l’autoencoder convolutif utilisé pour la détection hors-domaine.

Le modèle apprend à reconstruire les images du domaine connu. Deux stratégies d’évaluation sont comparées :

- erreur de reconstruction pixel, avec seuil calibré sur la validation ;
- modélisation de l’espace latent avec un `GaussianMixture` de scikit-learn.

### `core/ood_detection.py`

Ce fichier fournit les fonctions d’inférence OOD sur image unique :

- chargement du checkpoint autoencoder ;
- chargement du seuil `ood_threshold.json` ;
- calcul de l’erreur de reconstruction ;
- décision `Image acceptée` ou `Image hors domaine`.

### `core/pipeline.py`

Ce fichier assemble le pipeline complet :

1. détection OOD par autoencoder ;
2. rejet de l’image si elle dépasse le seuil ;
3. classification CNN uniquement si l’image est acceptée.

---

## 4. Premier entraînement CNN

Un premier script d’entraînement a été créé :

```text
scripts/train_cnn.py
```

Ce script permet de lancer un entraînement CNN reproductible depuis la racine du projet :

```bash
python -m scripts.train_cnn
```

Le script gère :

- la détection du device disponible : `mps`, `cuda` ou `cpu` ;
- le chargement des données ;
- la construction du modèle choisi via `--architecture` ;
- le choix entre pré-entraînement ImageNet, gel du backbone et fine-tuning partiel ;
- l’entraînement sur plusieurs epochs ;
- l’évaluation sur le jeu de validation ;
- le scheduler `ReduceLROnPlateau` ;
- l’early stopping ;
- la sauvegarde du meilleur modèle ;
- la sauvegarde de l’historique d’entraînement ;
- le suivi expérimental avec MLflow.

Exemples :

```bash
python -m scripts.train_cnn --architecture mobilenet_v3_large --pretrained --freeze-backbone --weighted-sampler
python -m scripts.train_cnn --architecture custom_cnn --weighted-sampler
```

Une première exécution a été validée avec succès sur Mac via le backend `mps`.

À ce stade, le pipeline d’entraînement est opérationnel et permet de lancer plusieurs expériences comparables. Il reste à compléter l’évaluation détaillée et l’analyse des résultats.

---

## Stratégie expérimentale CNN

Le sujet demande d’entraîner au moins deux architectures CNN différentes et de comparer leurs performances. Il suggère notamment ResNet50, EfficientNet, VGG16, DenseNet, MobileNet ou Inception. Il demande également d’explorer le transfer learning, le fine-tuning, et éventuellement l’entraînement from scratch si les ressources le permettent.

La stratégie retenue consiste donc à comparer plusieurs familles de modèles complémentaires.

### Architectures retenues

| Architecture      | Nom dans le code        | Rôle dans l’étude          | Justification                                                            |
| ----------------- | ----------------------- | -------------------------- | ------------------------------------------------------------------------ |
| VGG16             | `vgg16`                 | Baseline classique         | Architecture CNN historique, simple à interpréter, utile comme référence |
| ResNet50          | `resnet50`              | Modèle central             | Architecture résiduelle robuste, explicitement suggérée dans le sujet    |
| EfficientNet-B0   | `efficientnet_b0`       | Modèle moderne compact     | Bon compromis entre performance, nombre de paramètres et coût de calcul  |
| MobileNetV3-Large | `mobilenet_v3_large`    | Modèle léger pour l’app    | Inférence rapide, adaptée à une interface Streamlit interactive          |
| CNN custom        | `custom_cnn`            | Baseline from scratch      | Référence pédagogique sans poids ImageNet                                |

### Positionnement des architectures

#### VGG16

VGG16 est utilisé comme baseline classique.  
Son architecture séquentielle permet d’obtenir une référence simple, même si elle est plus lourde et moins moderne que ResNet ou EfficientNet.

#### ResNet50

ResNet50 est utilisé comme modèle de référence principal.  
Son architecture résiduelle permet d’entraîner des réseaux plus profonds en limitant les problèmes de gradient. C’est une bonne base de comparaison pour un projet de classification d’images médicales.

#### EfficientNet-B0

EfficientNet-B0 est utilisé comme modèle moderne et plus compact.  
Il permet de tester une architecture optimisée qui cherche un meilleur équilibre entre profondeur, largeur et résolution d’entrée.

#### MobileNetV3-Large

MobileNetV3-Large est retenu pour tester un modèle plus léger et rapide en inférence.  
Ce choix est cohérent avec l’objectif d’intégrer la prédiction dans une application Streamlit avec similarité visuelle, Grad-CAM et détection OOD.

#### CNN custom

Le CNN custom est entraîné from scratch, sans poids pré-entraînés.  
Il sert de baseline pédagogique pour mesurer l’apport réel du transfer learning sur un petit dataset médical.

---

## Stratégies d’entraînement

### 1. Transfer learning

La première stratégie consiste à utiliser des poids pré-entraînés sur ImageNet.

Le backbone du modèle est initialisé avec des poids pré-entraînés, puis la dernière couche de classification est remplacée pour prédire les 7 classes de plaies du dataset.

Cette stratégie est particulièrement pertinente ici car le dataset est petit : seulement 431 images. Un entraînement complet from scratch risquerait de surapprendre rapidement.

### 2. Gel du backbone

Dans un premier temps, le backbone est gelé et seule la tête de classification est entraînée.

Objectif :

- stabiliser l’apprentissage ;
- limiter l’overfitting ;
- exploiter les features génériques déjà apprises sur ImageNet ;
- obtenir une première comparaison fiable entre architectures.

### 3. Fine-tuning partiel

Dans un second temps, les dernières couches du backbone peuvent être dégelées afin d’adapter les représentations visuelles au domaine des plaies.

Objectif :

- adapter les features génériques au domaine médical ;
- améliorer les performances sur les classes visuellement proches ;
- limiter le surapprentissage en ne dégelant pas tout le réseau.

### 4. Entraînement from scratch

Un entraînement from scratch est prévu avec `custom_cnn`, principalement pour répondre à la question de réflexion sur l’intérêt du transfer learning.

Compte tenu de la faible taille du dataset, cette stratégie n’est pas considérée comme prioritaire. Elle servira plutôt de comparaison pédagogique pour montrer les limites d’un apprentissage sans pré-entraînement sur un petit dataset médical.

---

### Choix de MobileNetV3-Large

Pourquoi MobileNet plutôt que DenseNet ou Inception ?

MobileNetV3-Large est le plus intéressant pour notre cas, parce que le projet vise une plateforme interactive Streamlit avec prédiction, similarité visuelle, éventuellement Grad-CAM et détection OOD. Un modèle léger est donc utile pour avoir une inférence plus rapide et une intégration plus fluide.

DenseNet est intéressant techniquement, mais plus proche de ResNet dans l’idée “réseau profond performant”. Inception est plus ancien et un peu moins pratique à manipuler dans torchvision, notamment à cause de ses spécificités d’entrée/sortie. MobileNet apporte une vraie comparaison différente : modèle compact vs modèle lourd.

## Hyperparamètres de départ

Les premiers entraînements utiliseront les hyperparamètres suivants :

| Hyperparamètre                  |                        Valeur initiale |
| ------------------------------- | -------------------------------------: |
| Taille d’entrée                 |                              224 × 224 |
| Batch size                      |                                     16 |
| Optimizer                       |                                  AdamW |
| Learning rate transfer learning |                                   1e-4 |
| Learning rate fine-tuning       |                                   1e-5 |
| Weight decay                    |                                   1e-4 |
| Dropout                         |                                    0.3 |
| Nombre d’epochs                 |                                     20 |
| Early stopping                  |                                    Oui |
| Scheduler                       | ReduceLROnPlateau ou CosineAnnealingLR |
| Fonction de perte               |                       CrossEntropyLoss |
| Rééquilibrage principal         |                  WeightedRandomSampler |
| Rééquilibrage comparatif        |                          Class weights |

Ces valeurs sont susceptibles d’être ajustées à partir des résultats observés dans MLflow.

---

## Runs expérimentaux prévus

Les expériences minimales prévues sont les suivantes :

| Run | Architecture      | Pré-entraînement | Stratégie           | Rééquilibrage         |
| --: | ----------------- | ---------------- | ------------------- | --------------------- |
|   1 | ResNet50          | ImageNet         | Transfer learning   | WeightedRandomSampler |
|   2 | EfficientNet-B0   | ImageNet         | Transfer learning   | WeightedRandomSampler |
|   3 | MobileNetV3-Large | ImageNet         | Transfer learning   | WeightedRandomSampler |
|   4 | ResNet50          | ImageNet         | Fine-tuning partiel | WeightedRandomSampler |

Des runs complémentaires pourront être ajoutés :

| Run | Architecture      | Pré-entraînement | Stratégie         | Objectif                              |
| --: | ----------------- | ---------------- | ----------------- | ------------------------------------- |
|   5 | CNN custom        | Non              | From scratch      | Comparaison avec le transfer learning |
|   6 | VGG16             | ImageNet         | Transfer learning | Baseline classique                    |
|   7 | EfficientNet-B0   | ImageNet         | Transfer learning | Class weights                         |
|   8 | MobileNetV3-Large | ImageNet         | Fine-tuning léger | Comparaison modèle léger fine-tuné    |

---

## Suivi expérimental avec MLflow

MLflow sera utilisé pour tracer et comparer les expériences.

Pour chaque run, les éléments suivants devront être enregistrés :

### Paramètres

- architecture ;
- taille d’image ;
- batch size ;
- nombre d’epochs ;
- optimizer ;
- learning rate ;
- weight decay ;
- dropout ;
- stratégie de rééquilibrage ;
- utilisation ou non du pré-entraînement ;
- activation ou non du fine-tuning.

### Métriques

- train loss ;
- validation loss ;
- train accuracy ;
- validation accuracy ;
- accuracy finale ;
- precision macro ;
- recall macro ;
- F1-score macro ;
- F1-score par classe.

### Artefacts

- modèle entraîné ;
- historique d’entraînement ;
- courbes loss/accuracy ;
- matrice de confusion ;
- classification report ;
- configuration du run.

L’interface MLflow permettra de comparer les modèles et d’identifier le meilleur compromis entre performance globale et performance sur les classes minoritaires.

---

## Évaluation attendue des modèles

Les modèles seront évalués avec plusieurs métriques, car l’accuracy seule est insuffisante sur un dataset déséquilibré.

Métriques prévues :

- accuracy globale ;
- accuracy par classe ;
- precision par classe ;
- recall par classe ;
- F1-score par classe ;
- F1-score macro ;
- matrice de confusion ;
- analyse des erreurs de classification.

Une attention particulière sera portée aux classes minoritaires :

- `Ingrown_nails` ;
- `Stab_wound`.

L’objectif est d’éviter qu’un modèle obtienne une bonne accuracy globale uniquement parce qu’il prédit correctement les classes majoritaires comme `Bruises` ou `Abrasions`.

---

## Modèle retenu et inférence

Après comparaison des runs sur le jeu de validation, le modèle retenu est le **ResNet50 fine-tuné**.

Le checkpoint original est :

```text
models/resnet50_pretrained-True_freeze-False_finetune-True_weighted-True_classweights-False_lr-1e-05_resnet50_finetune_final_100epochs_20260625_1618_best.pt
```

Une copie courte a été créée pour faciliter l’utilisation dans les scripts et l’interface :

```text
models/resnet50_best.pt
```

Ce fichier contient les poids du modèle, l’architecture, la taille d’image, le mapping des classes et les informations nécessaires pour reconstruire le modèle sans dépendre de l’ordre local des dossiers.

### Résultats du meilleur modèle

Sur validation, le ResNet50 fine-tuné est le meilleur modèle au macro-F1 :

| Modèle | Validation accuracy | Validation macro-F1 |
| ------ | ------------------: | ------------------: |
| ResNet50 fine-tuné | 0.8615 | 0.8656 |
| MobileNetV3-Large frozen | 0.8615 | 0.8477 |
| ResNet50 frozen | 0.7538 | 0.7219 |

L’évaluation finale sur le jeu de test donne :

| Métrique | Valeur |
| -------- | -----: |
| Accuracy test | 0.8923 |
| Macro precision test | 0.9005 |
| Macro recall test | 0.8838 |
| Macro-F1 test | 0.8898 |
| Weighted-F1 test | 0.8919 |
| Erreurs | 7 / 65 |

Les artefacts de test sont sauvegardés dans :

```text
reports/evaluation/resnet50_pretrained-True_freeze-False_finetune-True_weighted-True_classweights-False_lr-1e-05_resnet50_finetune_final_100epochs_20260625_1618/test/
```

Ce dossier contient notamment :

- `evaluation_summary.json` ;
- `classification_report.csv` et `classification_report.json` ;
- `confusion_matrix.png` ;
- `predictions.csv` ;
- `misclassified_examples.png`.

### Charger le meilleur modèle en Python

Exemple d’utilisation depuis la racine du projet :

```python
from core.inference import load_model_from_checkpoint

model, checkpoint, device = load_model_from_checkpoint("models/resnet50_best.pt")

print(checkpoint["architecture"])
print(checkpoint["class_names"])
print(device)
```

### Prédire une image en Python

```python
from core.inference import predict_image

result = predict_image(
    image_path="data/raw/Burns/exemple.jpg",
    checkpoint_path="models/resnet50_best.pt",
    top_k=3,
)

print(result["predicted_class"])
print(result["confidence"])
print(result["top_k"])
```

La fonction retourne un dictionnaire de ce type :

```json
{
    "image_path": "data/raw/Burns/exemple.jpg",
    "architecture": "resnet50",
    "predicted_class": "Burns",
    "confidence": 0.92,
    "top_k": [
        {"rank": 1, "class": "Burns", "probability": 0.92},
        {"rank": 2, "class": "Cut", "probability": 0.05},
        {"rank": 3, "class": "Laceration", "probability": 0.03}
    ]
}
```

### Prédire une image en ligne de commande

Un script CLI a été ajouté :

```text
scripts/predict_image.py
```

Exemple avec affichage lisible :

```bash
python3 scripts/predict_image.py \
  --checkpoint models/resnet50_best.pt \
  --image data/raw/Burns/exemple.jpg \
  --top-k 3
```

Exemple avec sortie JSON :

```bash
python3 scripts/predict_image.py \
  --checkpoint models/resnet50_best.pt \
  --image data/raw/Burns/exemple.jpg \
  --top-k 3 \
  --json
```

Le chemin d’image doit pointer vers une image existante du projet ou vers un chemin absolu.

### Relancer l’évaluation du meilleur modèle

```bash
python3 scripts/evaluate_cnn.py \
  --checkpoint models/resnet50_best.pt \
  --split test \
  --batch-size 16 \
  --num-workers 0
```

---

## Détection hors-domaine par autoencoder

Le projet intègre un autoencoder convolutif pour détecter les images hors-domaine avant classification CNN.

L’objectif est d’éviter de forcer le ResNet50 à classer une image qui ne ressemble pas au domaine des plaies, par exemple un paysage, une radiographie ou un objet de bureau.

### Entraînement de l’autoencoder

Script :

```text
scripts/train_autoencoder.py
```

Commande utilisée pour le modèle final :

```bash
python -m scripts.train_autoencoder \
  --epochs 100 \
  --batch-size 16 \
  --latent-dim 256 \
  --lr 1e-4 \
  --run-suffix final
```

Le checkpoint final est :

```text
models/conv_autoencoder_latent-256_lr-0.0001_final_best.pt
```

Le meilleur autoencoder a été entraîné pendant 71 epochs effectives avec une meilleure loss validation de reconstruction d’environ `0.01391`.

Les artefacts d’entraînement sont sauvegardés dans :

```text
reports/ood/conv_autoencoder_latent-256_lr-0.0001_final/
```

Ce dossier contient notamment :

- `autoencoder_history.json` ;
- `autoencoder_training_curves.png` ;
- `reconstruction_examples.png`.

### Évaluation par erreur de reconstruction

Script :

```text
scripts/evaluate_autoencoder.py
```

Commande :

```bash
python3 scripts/evaluate_autoencoder.py \
  --checkpoint models/conv_autoencoder_latent-256_lr-0.0001_final_best.pt \
  --threshold-percentile 95 \
  --batch-size 16 \
  --num-workers 0 \
  --ood-dir data/ood
```

Le seuil est calibré sur le percentile 95 des erreurs de reconstruction du split validation.

Résultats obtenus avec le seuil P95 :

| Split | Images rejetées | Total | Taux de rejet |
| ----- | --------------: | ----: | ------------: |
| Validation | 4 | 65 | 6,15 % |
| Test | 3 | 65 | 4,62 % |
| OOD | 3 | 5 | 60,00 % |

Détail des images OOD :

| Image | Décision |
| ----- | -------- |
| `landscape.jpg` | Rejetée |
| `random_object.jpg` | Rejetée |
| `xray.jpg` | Rejetée |
| `cat.jpg` | Acceptée |
| `healthy_skin_or_other.jpg` | Acceptée |

Artefacts :

```text
reports/ood/conv_autoencoder_latent-256_lr-0.0001_final/evaluation/
```

Ce dossier contient :

- `ood_threshold.json` ;
- `reconstruction_errors.csv` ;
- `threshold_comparison.csv` ;
- `reconstruction_error_distribution.png` ;
- `reconstruction_error_boxplot.png`.

### Évaluation par espace latent + GMM

Une seconde approche a été ajoutée pour ne pas dépendre uniquement de l’erreur pixel.

Script :

```text
scripts/evaluate_autoencoder_latent_gmm.py
```

Principe :

1. extraire les embeddings latents avec `model.encode(image)` ;
2. standardiser les latents avec `StandardScaler` ;
3. entraîner un `GaussianMixture` sur les latents du train ;
4. utiliser `- log_likelihood` comme score d’anomalie ;
5. calibrer un seuil sur la validation.

Commande retenue pour l’essai final :

```bash
python3 scripts/evaluate_autoencoder_latent_gmm.py \
  --checkpoint models/conv_autoencoder_latent-256_lr-0.0001_final_best.pt \
  --threshold-percentile 90 \
  --gmm-components 7 \
  --covariance-type diag \
  --batch-size 16 \
  --num-workers 0 \
  --ood-dir data/ood \
  --image-extensions .jpg,.jpeg
```

Résultats obtenus :

| Split | Images rejetées | Total | Taux de rejet |
| ----- | --------------: | ----: | ------------: |
| Validation | 7 | 65 | 10,77 % |
| Test | 9 | 65 | 13,85 % |
| OOD | 3 | 5 | 60,00 % |

Détail des images OOD :

| Image | Décision |
| ----- | -------- |
| `xray.jpg` | Rejetée |
| `random_object.jpg` | Rejetée |
| `landscape.jpg` | Rejetée |
| `cat.jpg` | Acceptée |
| `healthy_skin_or_other.jpg` | Acceptée |

Artefacts :

```text
reports/ood/conv_autoencoder_latent-256_lr-0.0001_final/evaluation_latent_gmm/
```

Ce dossier contient :

- `latent_gmm_threshold.json` ;
- `latent_gmm_scores.csv` ;
- `latent_gmm_threshold_comparison.csv` ;
- `latent_gmm_config_comparison_jpg_only.csv` ;
- `latent_gmm_config_ood_details_jpg_only.csv` ;
- `latent_gmm_score_distribution.png` ;
- `latent_gmm_score_boxplot.png`.

### Analyse des résultats OOD

L’approche par erreur de reconstruction P95 est actuellement la plus simple et la plus défendable pour le pipeline principal :

- elle rejette les OOD très éloignées du domaine (`landscape`, `xray`, `random_object`) ;
- elle rejette peu d’images valides du split test ;
- elle reste facile à expliquer dans le rapport.

L’approche GMM latent est plus avancée, mais elle ne détecte pas davantage d’images OOD sur le petit jeu actuel. Elle rejette aussi plus d’images valides au seuil retenu. Elle est donc utile comme comparaison expérimentale, mais pas encore comme filtre principal.

Les deux méthodes montrent une limite importante : `cat.jpg` et `healthy_skin_or_other.jpg` peuvent être acceptées. Cela indique que l’autoencoder seul ne suffit pas toujours pour une détection OOD robuste. Pour améliorer le système, il faudra ajouter davantage d’exemples OOD, ou combiner le score autoencoder avec des embeddings CNN.

### Notebook d’analyse OOD

Le notebook suivant synthétise les résultats et affiche les chemins des images OOD utilisées :

```text
notebooks/autoencoder_ood_analysis.ipynb
```

Il charge :

- `reconstruction_errors.csv` ;
- `ood_threshold.json` ;
- les images de `data/ood/`.

Il produit des tableaux par split, les taux de rejet, les chemins absolus/relatifs des images et une visualisation des OOD triées par erreur de reconstruction.

---

## Pipeline complet OOD + classification

Le pipeline complet est disponible dans :

```text
core/pipeline.py
scripts/analyze_image.py
```

Il suit cette logique :

1. calculer le score OOD avec l’autoencoder ;
2. comparer ce score au seuil sauvegardé ;
3. rejeter l’image si elle est hors-domaine ;
4. sinon, charger `models/resnet50_best.pt` et prédire la classe de plaie.

Commande exemple pour une image du domaine :

```bash
python3 scripts/analyze_image.py \
  --image "data/raw/Abrasions/abrasions (57).jpg" \
  --cnn-checkpoint models/resnet50_best.pt \
  --autoencoder-checkpoint models/conv_autoencoder_latent-256_lr-0.0001_final_best.pt \
  --ood-threshold reports/ood/conv_autoencoder_latent-256_lr-0.0001_final/evaluation/ood_threshold.json \
  --top-k 3 \
  --json
```

Commande exemple pour une image OOD :

```bash
python3 scripts/analyze_image.py \
  --image data/ood/xray.jpg \
  --cnn-checkpoint models/resnet50_best.pt \
  --autoencoder-checkpoint models/conv_autoencoder_latent-256_lr-0.0001_final_best.pt \
  --ood-threshold reports/ood/conv_autoencoder_latent-256_lr-0.0001_final/evaluation/ood_threshold.json \
  --top-k 3 \
  --json
```

Si l’image est rejetée, aucune classification CNN n’est effectuée. Cela évite de produire une prédiction artificiellement confiante sur une image manifestement hors-domaine.

---

## Recherche par similarité visuelle

La recherche par similarité est maintenant disponible côté backend.

Elle repose sur :

- `core/image_similarity.py` pour charger le meilleur CNN, extraire les embeddings et interroger ChromaDB ;
- `scripts/build_similarity_index.py` pour construire l’index complet du dataset ;
- `scripts/search_similar_images.py` pour rechercher les top-K images similaires à une image requête ;
- `scripts/evaluate_similarity.py` pour évaluer la qualité de la recherche et générer des figures pour le rapport.

Le modèle utilisé par défaut est :

```text
models/resnet50_best.pt
```

L’index est sauvegardé dans :

```text
data/processed/chroma/
```

Ce dossier ne doit pas être versionné. Il est déjà couvert par `.gitignore` via `data/processed/`.

### Construire l’index ChromaDB

Commande recommandée :

```bash
python3 scripts/build_similarity_index.py \
  --checkpoint models/resnet50_best.pt \
  --splits train,val,test \
  --batch-size 32
```

Cette commande :

1. lit les fichiers `train.csv`, `val.csv` et `test.csv` ;
2. charge le modèle ResNet50 sauvegardé ;
3. extrait un embedding CNN normalisé L2 pour chaque image ;
4. indexe les embeddings dans ChromaDB avec distance cosinus ;
5. écrit un résumé JSON dans `reports/similarity/index_summary.json`.

Résultat obtenu :

```text
Images indexées : 431
Documents dans ChromaDB : 431
Architecture : resnet50
Collection : wound_image_embeddings
```

Artefact utile pour le rapport :

```text
reports/similarity/index_summary.json
```

### Rechercher des images similaires

Commande simple :

```bash
python3 scripts/search_similar_images.py \
  --image "data/raw/Bruises/bruises (37).jpg" \
  --checkpoint models/resnet50_best.pt \
  --top-k 5
```

Commande recommandée pour une image déjà présente dans l’index :

```bash
python3 scripts/search_similar_images.py \
  --image "data/raw/Bruises/bruises (37).jpg" \
  --checkpoint models/resnet50_best.pt \
  --top-k 5 \
  --exclude-query
```

L’option `--exclude-query` retire l’image requête des résultats. Elle est importante pour l’évaluation et pour une interface Streamlit, car l’utilisateur veut voir des cas similaires différents de l’image uploadée.

Sortie JSON exploitable par une interface :

```bash
python3 scripts/search_similar_images.py \
  --image "data/raw/Bruises/bruises (37).jpg" \
  --checkpoint models/resnet50_best.pt \
  --top-k 5 \
  --exclude-query \
  --json
```

Chaque voisin retourné contient :

- le rang ;
- le chemin image ;
- la classe ;
- le score de similarité ;
- la distance ChromaDB.

### Générer une visualisation query + top-K

Pour produire une figure directement exploitable dans le rapport :

```bash
python3 scripts/search_similar_images.py \
  --image "data/raw/Bruises/bruises (37).jpg" \
  --checkpoint models/resnet50_best.pt \
  --top-k 5 \
  --exclude-query \
  --save-figure reports/similarity/query_topk_bruises_37.png
```

Figure générée :

```text
reports/similarity/query_topk_bruises_37.png
```

Cette figure affiche :

- l’image requête ;
- les top-K voisins ;
- la classe de chaque voisin ;
- le score de similarité.

### Évaluer la recherche par similarité

Commande recommandée sur le split test :

```bash
python3 scripts/evaluate_similarity.py \
  --query-splits test \
  --top-k 5 \
  --num-figures 4
```

Cette évaluation exclut automatiquement l’image requête de ses propres voisins.

Métriques produites :

- top-1 accuracy ;
- hit@K ;
- precision@K moyenne ;
- MRR ;
- métriques par classe.

Résultats obtenus sur le split test :

| Métrique | Valeur |
| -------- | -----: |
| Nombre de requêtes | 65 |
| Top-1 accuracy | 0,8769 |
| Hit@5 | 0,9538 |
| Precision@5 moyenne | 0,8615 |
| MRR | 0,9103 |

Artefacts générés :

```text
reports/similarity/evaluation/similarity_evaluation_summary.json
reports/similarity/evaluation/similarity_evaluation_details.csv
reports/similarity/evaluation/similarity_evaluation_by_class.csv
reports/similarity/evaluation/figures/
```

Les figures `query_topk_*.png` peuvent être intégrées directement au rapport académique.

### Utilisation dans Streamlit

Pour brancher la similarité dans une page Streamlit, utiliser directement les fonctions du module `core.image_similarity`.

Exemple minimal :

```python
from core.image_similarity import search_similar_images

results = search_similar_images(
    query_image_path=image_path,
    checkpoint_path="models/resnet50_best.pt",
    persist_dir="data/processed/chroma",
    collection_name="wound_image_embeddings",
    top_k=5,
    exclude_image_path=image_path,
)
```

Dans l’application Streamlit, il faut prévoir :

1. vérifier que `data/processed/chroma/` existe ;
2. si l’index est absent, afficher un message demandant de lancer `scripts/build_similarity_index.py` ;
3. après une prédiction acceptée par le filtre OOD, appeler `search_similar_images` ;
4. afficher les voisins avec `st.image`, leur classe et leur score ;
5. rappeler que les voisins similaires ne constituent pas une preuve médicale.

Commandes à exécuter avant de tester Streamlit avec la similarité :

```bash
pip install -r requirements.txt

python3 scripts/build_similarity_index.py \
  --checkpoint models/resnet50_best.pt \
  --splits train,val,test \
  --batch-size 32

python3 scripts/evaluate_similarity.py \
  --query-splits test \
  --top-k 5 \
  --num-figures 4

streamlit run app-streamlit/Home.py
```

---

## Commandes utiles

### Lancer l’entraînement CNN

```bash
python -m scripts.train_cnn
```

Exemples de runs ciblés :

```bash
python -m scripts.train_cnn --architecture resnet50 --pretrained --freeze-backbone --weighted-sampler
python -m scripts.train_cnn --architecture mobilenet_v3_large --pretrained --freeze-backbone --weighted-sampler
python -m scripts.train_cnn --architecture custom_cnn --weighted-sampler
```

Remarque : la baseline custom est appelée `custom_cnn` dans le code.

### Lancer MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Puis ouvrir :

```text
http://localhost:5000
```

### Lancer Streamlit

```bash
streamlit run app-streamlit/Home.py
```

### Prédire une image avec le meilleur modèle

```bash
python3 scripts/predict_image.py --checkpoint models/resnet50_best.pt --image data/raw/Burns/exemple.jpg --top-k 3
```

### Construire l’index de similarité visuelle

```bash
python3 scripts/build_similarity_index.py \
  --checkpoint models/resnet50_best.pt \
  --splits train,val,test \
  --batch-size 32
```

### Rechercher les images similaires à une image requête

```bash
python3 scripts/search_similar_images.py \
  --image "data/raw/Bruises/bruises (37).jpg" \
  --checkpoint models/resnet50_best.pt \
  --top-k 5 \
  --exclude-query
```

### Générer une figure query + top-K

```bash
python3 scripts/search_similar_images.py \
  --image "data/raw/Bruises/bruises (37).jpg" \
  --checkpoint models/resnet50_best.pt \
  --top-k 5 \
  --exclude-query \
  --save-figure reports/similarity/query_topk_bruises_37.png
```

### Évaluer la recherche par similarité

```bash
python3 scripts/evaluate_similarity.py \
  --query-splits test \
  --top-k 5 \
  --num-figures 4
```

### Entraîner l’autoencoder OOD

```bash
python -m scripts.train_autoencoder --epochs 100 --batch-size 16 --latent-dim 256 --lr 1e-4 --run-suffix final
```

### Évaluer le filtre OOD par reconstruction

```bash
python3 scripts/evaluate_autoencoder.py \
  --checkpoint models/conv_autoencoder_latent-256_lr-0.0001_final_best.pt \
  --threshold-percentile 95 \
  --ood-dir data/ood
```

### Évaluer le filtre OOD par GMM latent

```bash
python3 scripts/evaluate_autoencoder_latent_gmm.py \
  --checkpoint models/conv_autoencoder_latent-256_lr-0.0001_final_best.pt \
  --threshold-percentile 90 \
  --gmm-components 7 \
  --covariance-type diag \
  --ood-dir data/ood \
  --image-extensions .jpg,.jpeg
```

### Lancer le pipeline complet

```bash
python3 scripts/analyze_image.py \
  --image data/ood/xray.jpg \
  --cnn-checkpoint models/resnet50_best.pt \
  --autoencoder-checkpoint models/conv_autoencoder_latent-256_lr-0.0001_final_best.pt \
  --ood-threshold reports/ood/conv_autoencoder_latent-256_lr-0.0001_final/evaluation/ood_threshold.json \
  --top-k 3 \
  --json
```

---

## Prochaines étapes

Les prochaines étapes du projet sont :

1. intégrer le pipeline `OOD + ResNet50` dans la page Streamlit de prédiction ;
2. brancher dans Streamlit la recherche par similarité déjà disponible dans `core/image_similarity.py` ;
3. ajouter Grad-CAM pour expliquer les prédictions du meilleur modèle ;
4. enrichir le jeu OOD avec davantage d’images variées ;
5. exploiter les rapports CNN, OOD et similarité dans le rapport académique.

---

## Limites identifiées à ce stade

Le dataset est relativement petit et déséquilibré.  
Cela peut entraîner :

- du surapprentissage ;
- une forte variabilité selon le split ;
- une mauvaise généralisation sur les classes minoritaires ;
- des confusions entre classes visuellement proches.

Pour limiter ces risques, le projet utilise :

- un split stratifié ;
- de la data augmentation modérée ;
- du transfer learning ;
- du rééquilibrage par échantillonnage ou pondération ;
- une évaluation par classe ;
- une analyse d’erreurs ;
- un filtre OOD avant classification ;
- une comparaison reconstruction pixel vs GMM latent ;
- un suivi expérimental avec MLflow.

---

## Avertissement

Ce projet est réalisé dans un cadre pédagogique.  
Il ne constitue pas un dispositif médical et ne doit pas être utilisé pour établir un diagnostic clinique réel.

Les prédictions du modèle doivent être interprétées comme une aide expérimentale à l’analyse d’image et non comme une décision médicale.
