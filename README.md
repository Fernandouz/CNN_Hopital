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
- un assistant LLM/RAG pédagogique avec traçabilité Langfuse optionnelle.

Le projet suit le déroulé du sujet fourni, qui demande notamment une classification CNN, un autoencoder ou modèle équivalent pour la détection OOD, un suivi MLflow, une recherche par embeddings, une interface Streamlit, une analyse de dérive avec Evidently AI, ainsi que des extensions valorisantes comme Grad-CAM, RAG/LLM, Langfuse et Docker.

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
│   ├── vae.py
│   ├── ood_detection.py
│   ├── vae_ood_detection.py
│   ├── pipeline.py
│   ├── grad_cam.py
│   ├── image_similarity.py
│   ├── database.py
│   ├── rag_pipeline.py
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
│   ├── train_vae.py
│   ├── evaluate_vae_ood.py
│   ├── generate_grad_cam_report.py
│   ├── create_medical_kb.py
│   ├── analyze_image.py
│   └── drift_monitoring.py
│
├── app-streamlit/
│   ├── Home.py
│   └── pages/
│       ├── 1_Dataset_Explorer.py
│       ├── 2_Training.py
│       ├── 3_Prediction.py
│       └── 5_AI_Assistant.py
│
├── models/
├── reports/
│   └── figures/
├── tests/
├── mlruns/
├── requirements.txt
├── .streamlit/
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

Les index et artefacts générés ne doivent pas non plus être versionnés :

- `data/processed/` pour les splits, uploads temporaires et index de similarité visuelle ;
- `data/chroma_kb/` pour l’index ChromaDB de la base de connaissances RAG ;
- `models/`, `mlruns/`, `mlflow.db` et `reports/`.

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
seaborn
scikit-learn
Pillow
tqdm
mlflow
streamlit
opencv-python
evidently
chromadb
sentence-transformers
ollama
langfuse
transformers
accelerate
pytest
```

---

## État d’avancement du projet

Le projet couvre maintenant le cœur Deep Learning et plusieurs extensions valorisantes : exploration du dataset, entraînement CNN avec MLflow, détection OOD, similarité visuelle, Grad-CAM, interface Streamlit multi-pages et assistant RAG médical pédagogique.

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
3. classification CNN uniquement si l’image est acceptée ;
4. génération optionnelle d’une explication Grad-CAM si `include_grad_cam=True`.

La sortie est directement exploitable côté Streamlit : si l’image est rejetée, `prediction` et `grad_cam` valent `None`; si elle est acceptée, `prediction` contient la classe et le top-K, et `grad_cam` contient le chemin PNG de l’overlay Grad-CAM à afficher.

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

Les runs CNN sont regroupés dans l’experiment MLflow :

```text
wound-classification-app
```

Le même script est utilisé par la page Streamlit `2_Training.py`. L’application construit une commande équivalente à `python -m scripts.train_cnn` à partir des paramètres choisis dans l’interface.

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

| Architecture      | Nom dans le code     | Rôle dans l’étude       | Justification                                                            |
| ----------------- | -------------------- | ----------------------- | ------------------------------------------------------------------------ |
| VGG16             | `vgg16`              | Baseline classique      | Architecture CNN historique, simple à interpréter, utile comme référence |
| ResNet50          | `resnet50`           | Modèle central          | Architecture résiduelle robuste, explicitement suggérée dans le sujet    |
| EfficientNet-B0   | `efficientnet_b0`    | Modèle moderne compact  | Bon compromis entre performance, nombre de paramètres et coût de calcul  |
| MobileNetV3-Large | `mobilenet_v3_large` | Modèle léger pour l’app | Inférence rapide, adaptée à une interface Streamlit interactive          |
| CNN custom        | `custom_cnn`         | Baseline from scratch   | Référence pédagogique sans poids ImageNet                                |

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

Les runs CNN lancés depuis le CLI ou depuis Streamlit sont regroupés dans l’experiment :

```text
wound-classification-app
```

Le backend MLflow utilisé par le projet est :

```text
sqlite:///mlflow.db
```

Depuis l’application Streamlit, MLflow est démarré automatiquement sur `127.0.0.1:5001` au lancement d’un entraînement. Le process MLflow lancé par l’app est arrêté automatiquement lorsque le serveur Streamlit se ferme normalement.

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

| Modèle                   | Validation accuracy | Validation macro-F1 |
| ------------------------ | ------------------: | ------------------: |
| ResNet50 fine-tuné       |              0.8615 |              0.8656 |
| MobileNetV3-Large frozen |              0.8615 |              0.8477 |
| ResNet50 frozen          |              0.7538 |              0.7219 |

L’évaluation finale sur le jeu de test donne :

| Métrique             | Valeur |
| -------------------- | -----: |
| Accuracy test        | 0.8923 |
| Macro precision test | 0.9005 |
| Macro recall test    | 0.8838 |
| Macro-F1 test        | 0.8898 |
| Weighted-F1 test     | 0.8919 |
| Erreurs              | 7 / 65 |

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
    { "rank": 1, "class": "Burns", "probability": 0.92 },
    { "rank": 2, "class": "Cut", "probability": 0.05 },
    { "rank": 3, "class": "Laceration", "probability": 0.03 }
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

| Split      | Images rejetées | Total | Taux de rejet |
| ---------- | --------------: | ----: | ------------: |
| Validation |               4 |    65 |        6,15 % |
| Test       |               3 |    65 |        4,62 % |
| OOD        |               3 |     5 |       60,00 % |

Détail des images OOD :

| Image                       | Décision |
| --------------------------- | -------- |
| `landscape.jpg`             | Rejetée  |
| `random_object.jpg`         | Rejetée  |
| `xray.jpg`                  | Rejetée  |
| `cat.jpg`                   | Acceptée |
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

| Split      | Images rejetées | Total | Taux de rejet |
| ---------- | --------------: | ----: | ------------: |
| Validation |               7 |    65 |       10,77 % |
| Test       |               9 |    65 |       13,85 % |
| OOD        |               3 |     5 |       60,00 % |

Détail des images OOD :

| Image                       | Décision |
| --------------------------- | -------- |
| `xray.jpg`                  | Rejetée  |
| `random_object.jpg`         | Rejetée  |
| `landscape.jpg`             | Rejetée  |
| `cat.jpg`                   | Acceptée |
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

### Amélioration avec Variational Autoencoder

Une extension VAE a été ajoutée pour améliorer l’analyse hors-domaine avec une structure latente probabiliste.

Fichiers ajoutés :

```text
core/vae.py
core/vae_ood_detection.py
scripts/train_vae.py
scripts/evaluate_vae_ood.py
```

Contrairement à l’autoencoder classique, le VAE encode chaque image sous la forme d’une distribution latente :

```text
image -> encodeur -> mu, logvar -> z -> decodeur -> reconstruction
```

La fonction de perte combine :

```text
loss = reconstruction_loss + beta * KL_divergence
```

Scores OOD disponibles :

- `reconstruction` : erreur MSE entre image originale et reconstruction ;
- `kl` : divergence KL de la distribution latente ;
- `combined` : `reconstruction_error + alpha * kl_divergence`.

Commande de debug rapide :

```bash
python3 scripts/train_vae.py \
  --epochs 1 \
  --batch-size 4 \
  --latent-dim 32 \
  --beta 0.001 \
  --run-suffix debug_smoke \
  --max-train-batches 1 \
  --max-eval-batches 1
```

Commande recommandée pour un entraînement exploitable :

```bash
python3 scripts/train_vae.py \
  --epochs 100 \
  --batch-size 16 \
  --latent-dim 256 \
  --beta 0.001 \
  --lr 1e-4 \
  --patience 10 \
  --run-suffix final
```

Le checkpoint sera sauvegardé dans :

```text
models/conv_vae_latent-256_beta-0.001_lr-0.0001_final_best.pt
```

Les courbes et exemples de reconstruction seront sauvegardés dans :

```text
reports/ood/conv_vae_latent-256_beta-0.001_lr-0.0001_final/
```

Évaluation recommandée avec score combiné :

```bash
python3 scripts/evaluate_vae_ood.py \
  --checkpoint models/conv_vae_latent-256_beta-0.001_lr-0.0001_final_best.pt \
  --threshold-percentile 95 \
  --score combined \
  --alpha 0.001 \
  --batch-size 16 \
  --ood-dir data/ood \
  --image-extensions .jpg,.jpeg
```

Évaluations comparatives utiles :

```bash
python3 scripts/evaluate_vae_ood.py \
  --checkpoint models/conv_vae_latent-256_beta-0.001_lr-0.0001_final_best.pt \
  --threshold-percentile 95 \
  --score reconstruction \
  --batch-size 16 \
  --ood-dir data/ood \
  --image-extensions .jpg,.jpeg

python3 scripts/evaluate_vae_ood.py \
  --checkpoint models/conv_vae_latent-256_beta-0.001_lr-0.0001_final_best.pt \
  --threshold-percentile 95 \
  --score kl \
  --batch-size 16 \
  --ood-dir data/ood \
  --image-extensions .jpg,.jpeg
```

Artefacts générés :

```text
reports/ood/conv_vae_latent-256_beta-0.001_lr-0.0001_final/evaluation_combined/
├── vae_ood_scores.csv
├── vae_ood_threshold.json
├── vae_combined_score_distribution.png
├── vae_combined_score_boxplot.png
└── vae_reconstruction_examples.png
```

Pour brancher le VAE dans Streamlit ou dans le pipeline, utiliser :

```python
from core.vae_ood_detection import detect_ood_vae

ood_result = detect_ood_vae(
    image_path=image_path,
    vae_checkpoint_path="models/conv_vae_latent-256_beta-0.001_lr-0.0001_final_best.pt",
    threshold_path="reports/ood/conv_vae_latent-256_beta-0.001_lr-0.0001_final/evaluation_combined/vae_ood_threshold.json",
)
```

Le résultat contient :

- `is_ood` ;
- `anomaly_score` ;
- `threshold` ;
- `reconstruction_error` ;
- `kl_divergence` ;
- `score` ;
- `decision`.

### Conclusion expérimentale sur le VAE

Le VAE a été testé comme amélioration probabiliste de l’autoencoder, mais les résultats obtenus ne justifient pas son utilisation comme filtre OOD principal.

Deux runs principaux ont été observés :

```text
conv_vae_latent-256_beta-0.001_lr-0.0001_final
conv_vae_latent-256_beta-0.001_lr-0.0001_final_patience40
```

Même avec une patience plus élevée, le VAE n’améliore pas la reconstruction. Le meilleur modèle apparaît tôt, puis la validation se dégrade.

Comparaison des pertes validation :

| Modèle                 | Meilleure epoch | Meilleure val loss |
| ---------------------- | --------------: | -----------------: |
| Autoencoder classique  |              61 |            0,01391 |
| VAE `final`            |               9 |            0,06790 |
| VAE `final_patience40` |              10 |            0,06697 |

L’écart est important : le VAE reconstruit environ 5 fois moins bien que l’autoencoder classique.

La comparaison qualitative confirme ce résultat. L’autoencoder produit des reconstructions floues mais encore structurées, tandis que le VAE produit principalement des images moyennes grisâtres, avec très peu de détails utiles.

Statistiques mesurées sur un batch de validation :

| Élément                | Moyenne reconstruction | Écart-type reconstruction |
| ---------------------- | ---------------------: | ------------------------: |
| Images originales      |                 0,5726 |                    0,2502 |
| Autoencoder classique  |                 0,5689 |                    0,2216 |
| VAE `final_patience40` |                 0,5290 |                    0,0374 |

L’écart-type très faible du VAE montre que ses reconstructions sont presque constantes. Le modèle tend donc vers une solution dégénérée : au lieu de reconstruire chaque image, il génère une moyenne visuelle du dataset.

Comparaison OOD :

| Méthode                         | Seuil |    Test rejeté |    OOD rejeté |
| ------------------------------- | ----- | -------------: | ------------: |
| Autoencoder reconstruction      | P95   |  3/65 = 4,62 % | 3/5 = 60,00 % |
| VAE KL `final_patience40`       | P95   |  4/65 = 6,15 % | 1/5 = 20,00 % |
| VAE combined `final_patience40` | P95   | 9/65 = 13,85 % | 1/5 = 20,00 % |

Le VAE est donc moins performant sur les deux objectifs :

- il rejette plus d’images valides du split test ;
- il détecte moins d’images hors-domaine ;
- il produit des reconstructions trop pauvres pour être interprétables ;
- il est moins stable que l’autoencoder classique.

Conclusion retenue pour le projet :

```text
Filtre OOD principal : autoencoder convolutionnel classique avec seuil P95
VAE : expérimentation comparative avancée, non retenue pour le pipeline final
```

> Un Variational Autoencoder a été testé afin d’ajouter une modélisation probabiliste de l’espace latent et d’améliorer l’estimation de l’incertitude. Cependant, sur ce dataset limité, le VAE n’a pas permis d’obtenir des reconstructions exploitables. Les sorties générées sont fortement lissées et proches d’une image moyenne, ce qui entraîne une erreur de reconstruction élevée et une séparation insuffisante entre images du domaine et images hors-domaine. En comparaison, l’autoencoder convolutionnel classique obtient une meilleure reconstruction et un meilleur taux de rejet OOD. Le VAE est donc conservé comme expérimentation comparative, mais le filtre OOD principal reste l’autoencoder classique.

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
4. sinon, charger `models/resnet50_best.pt` et prédire la classe de plaie ;
5. optionnellement, générer l’explication Grad-CAM pour la prédiction.

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

Commande exemple avec explicabilité Grad-CAM activée :

```bash
python3 scripts/analyze_image.py \
  --image "data/raw/Abrasions/abrasions (57).jpg" \
  --cnn-checkpoint models/resnet50_best.pt \
  --autoencoder-checkpoint models/conv_autoencoder_latent-256_lr-0.0001_final_best.pt \
  --ood-threshold reports/ood/conv_autoencoder_latent-256_lr-0.0001_final/evaluation/ood_threshold.json \
  --top-k 3 \
  --include-grad-cam \
  --json
```

Dans ce cas, si l’image est acceptée par le filtre OOD, la sortie contient aussi :

```json
{
  "grad_cam": {
    "target_class": "Abrasions",
    "target_class_idx": 0,
    "alpha": 0.45,
    "overlay_path": "reports/xai/pipeline/abrasions (57)_gradcam_overlay.png"
  }
}
```

Pour une image rejetée comme hors-domaine, `grad_cam` reste à `null`, car le modèle CNN ne doit pas expliquer une prédiction qui n’a pas été acceptée.

Le pipeline interactif ne sauvegarde volontairement que l’overlay Grad-CAM. Les images originale et heatmap seule restent générées par `scripts/generate_grad_cam_report.py`, qui sert au rapport académique.

### Utilisation du pipeline avec Grad-CAM dans Streamlit

L’application Streamlit peut appeler directement `core.pipeline.analyze_image` :

```python
from core.pipeline import analyze_image

result = analyze_image(
    image_path=image_path,
    cnn_checkpoint_path="models/resnet50_best.pt",
    autoencoder_checkpoint_path="models/conv_autoencoder_latent-256_lr-0.0001_final_best.pt",
    ood_threshold_path="reports/ood/conv_autoencoder_latent-256_lr-0.0001_final/evaluation/ood_threshold.json",
    top_k=3,
    include_grad_cam=True,
    grad_cam_output_dir="reports/xai/pipeline",
)

if result["status"] == "accepted":
    prediction = result["prediction"]
    grad_cam = result["grad_cam"]
    # Afficher prediction["predicted_class"], prediction["top_k"]
    # Afficher grad_cam["overlay_path"] avec st.image(...)
else:
    # Afficher result["ood"]["decision"] et ne pas afficher de Grad-CAM.
    pass
```

Le champ le plus utile côté interface est :

```text
result["grad_cam"]["overlay_path"]
```

Il pointe vers l’image superposée heatmap + image originale, prête à être affichée.

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

| Métrique            | Valeur |
| ------------------- | -----: |
| Nombre de requêtes  |     65 |
| Top-1 accuracy      | 0,8769 |
| Hit@5               | 0,9538 |
| Precision@5 moyenne | 0,8615 |
| MRR                 | 0,9103 |

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

Dans l’application Streamlit, la page Prediction intègre maintenant cette logique :

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

## Interface Streamlit

L’application Streamlit se lance depuis la racine du projet :

```bash
streamlit run app-streamlit/Home.py
```

Pages disponibles :

| Page | Fichier | Rôle |
| ---- | ------- | ---- |
| Accueil | `app-streamlit/Home.py` | Statistiques clés, navigation et rappel pédagogique |
| Dataset Explorer | `app-streamlit/pages/1_Dataset_Explorer.py` | Distribution des classes et inspection du dataset |
| Training | `app-streamlit/pages/2_Training.py` | Configuration et lancement de `python -m scripts.train_cnn` |
| Prediction | `app-streamlit/pages/3_Prediction.py` | Upload image, OOD, classification, Grad-CAM et similarité |
| AI Assistant | `app-streamlit/pages/5_AI_Assistant.py` | Assistant RAG médical pédagogique avec ChromaDB, Ollama/HuggingFace et Langfuse optionnel |

### Page Training

La page Training expose les principaux paramètres du script `scripts/train_cnn.py` :

- architecture : `resnet50`, `efficientnet_b0`, `mobilenet_v3_large`, `vgg16`, `custom_cnn` ;
- nombre d’epochs ;
- batch size ;
- taille d’image ;
- learning rate ;
- weight decay ;
- dropout ;
- patience d’early stopping ;
- nombre de workers ;
- pré-entraînement ImageNet ;
- gel du backbone ;
- fine-tuning ;
- `WeightedRandomSampler` ;
- poids de classes ;
- suffixe de run MLflow.

Au lancement, l’app utilise l’interpréteur `.venv/bin/python` s’il existe, ajoute la racine du projet au `PYTHONPATH`, affiche les logs d’entraînement dans un terminal compact, puis écrit le run dans l’experiment MLflow `wound-classification-app`.

MLflow est démarré automatiquement sur :

```text
http://127.0.0.1:5001
```

L’app affiche aussi les experiments détectés dans `mlflow.db` et un lien direct vers l’experiment CNN.

### Page Prediction

La page Prediction conserve le dernier résultat dans `st.session_state` pendant la session Streamlit. L’image uploadée, la prédiction, le statut OOD, l’overlay Grad-CAM et les cas similaires restent donc visibles lorsque l’utilisateur change de page puis revient.

Le pipeline affiché est :

```text
upload image
  -> détection OOD
  -> classification CNN si image acceptée
  -> top-3 prédictions
  -> overlay Grad-CAM sous l’image uploadée
  -> cas historiques similaires si l’index ChromaDB est disponible
```

L’ancienne page Explainability a été retirée temporairement : l’explicabilité utile à l’utilisateur est maintenant directement intégrée dans la page Prediction, juste sous la photo uploadée.

### Page AI Assistant

La page AI Assistant est intégrée au pipeline RAG de la partie 5. Elle utilise :

- `core/rag_pipeline.py` pour la recherche documentaire et la génération de réponse ;
- `core/ollama_client.py` pour appeler Ollama localement, avec fallback HuggingFace si Ollama est absent ;
- `core/langfuse_client.py` pour tracer les appels LLM si les clés Langfuse sont configurées ;
- `data/base_connaissances_medicales.jsonl` comme base de connaissances médicale pédagogique ;
- `scripts/create_medical_kb.py` pour indexer cette base dans ChromaDB.

La page propose deux modes :

1. **Diagnostic courant** : réutilise automatiquement le dernier diagnostic valide produit par la page Prediction pendant la session Streamlit.
2. **Diagnostic manuel** : permet de sélectionner une classe et une confiance simulée pour tester le RAG sans passer par une image.

Dans les deux cas, un bouton unique lance l’analyse RAG. La page affiche ensuite :

- la recommandation générée ;
- le backend utilisé (`ollama` ou `huggingface`) ;
- le modèle LLM ;
- les documents médicaux récupérés dans ChromaDB ;
- le prompt augmenté envoyé au LLM dans un volet technique.

Préparer la base de connaissances avant la première utilisation :

```bash
python3 scripts/create_medical_kb.py
```

Cette commande crée `data/chroma_kb/`, qui est ignoré par Git. Pour reconstruire l’index après modification du JSONL :

```bash
python3 scripts/create_medical_kb.py --reset
```

Pour utiliser Ollama localement :

```bash
ollama pull llama3.2
ollama serve
```

Langfuse est optionnel. Si `LANGFUSE_PUBLIC_KEY` et `LANGFUSE_SECRET_KEY` ne sont pas définies dans `.env`, le pipeline utilise un client no-op et continue de fonctionner sans traçabilité.

---

## Explicabilité visuelle avec Grad-CAM

La partie 6 du projet ajoute une brique d’explicabilité visuelle pour comprendre quelles zones de l’image influencent la prédiction du CNN.

L’objectif n’est pas de produire une preuve médicale, mais de vérifier si le modèle semble se concentrer sur la plaie ou sur des éléments périphériques comme l’arrière-plan, la peau saine, les artefacts d’image ou les marques d’eau.

### Principe général

Le pipeline Grad-CAM s’enchaîne ainsi :

```text
checkpoint CNN entraîné
        ↓
core/grad_cam.py
        ↓
scripts/generate_grad_cam_report.py
        ↓
reports/xai/resnet50/test/
        ↓
figures + CSV + JSON exploitables dans le rapport
```

Le checkpoint utilisé par défaut est :

```text
models/resnet50_best.pt
```

Ce modèle correspond au ResNet50 fine-tuné, retenu comme modèle principal pour l’analyse XAI.

### Module central : `core/grad_cam.py`

Le fichier `core/grad_cam.py` contient le code réutilisable pour générer les explications Grad-CAM.

Il prend en charge :

- le chargement d’un checkpoint CNN existant ;
- la reconstruction du modèle avec son mapping de classes ;
- la prédiction de l’image ;
- la sélection automatique de la dernière couche convolutive ;
- la génération d’une heatmap Grad-CAM ;
- la superposition de la heatmap sur l’image originale ;
- le retour des informations utiles : classe prédite, confiance, top-3, image originale, heatmap et overlay.

Les architectures actuellement prises en charge sont :

| Architecture       | Couche cible Grad-CAM                              |
| ------------------ | -------------------------------------------------- |
| `resnet50`         | `model.layer4[-1]`                                 |
| `efficientnet_b0`  | `model.features[-1]`                               |
| `mobilenet_v3_large` | `model.features[-1]`                             |
| `vgg16`            | dernière couche `Conv2d` de `model.features`       |
| `custom_cnn`       | dernière couche `Conv2d` de `model.features`       |

Fonction simple pour expliquer une image :

```python
from core.grad_cam import explain_image_with_grad_cam

result = explain_image_with_grad_cam(
    checkpoint_path="models/resnet50_best.pt",
    image_path="data/raw/Abrasions/abrasions (41).jpg",
)
```

Pour traiter plusieurs images efficacement, utiliser plutôt la classe `GradCAMExplainer`, qui charge le modèle une seule fois :

```python
from core.grad_cam import GradCAMExplainer

explainer = GradCAMExplainer(
    checkpoint_path="models/resnet50_best.pt",
)

result = explainer.explain_image(
    image_path="data/raw/Abrasions/abrasions (41).jpg",
)
```

Par défaut, la heatmap est générée pour la classe prédite. Il est aussi possible de forcer une classe cible, par nom ou par indice :

```python
result = explainer.explain_image(
    image_path="data/raw/Abrasions/abrasions (41).jpg",
    target_class="Abrasions",
)
```

### Générer les artefacts XAI pour le rapport

Le script principal est :

```text
scripts/generate_grad_cam_report.py
```

Commande recommandée :

```bash
python3 scripts/generate_grad_cam_report.py \
  --checkpoint models/resnet50_best.pt \
  --split test \
  --num-correct 5 \
  --num-errors 3
```

Cette commande :

1. charge le checkpoint ResNet50 ;
2. lit le split demandé, par exemple `data/processed/splits/test.csv` ;
3. refait les prédictions sur toutes les images du split ;
4. sélectionne automatiquement les 5 bonnes classifications les plus confiantes ;
5. sélectionne automatiquement les 3 erreurs les plus confiantes ;
6. génère une explication Grad-CAM pour chaque image sélectionnée ;
7. sauvegarde les images individuelles et une grille de synthèse ;
8. écrit les métadonnées dans des fichiers CSV et JSON.

Sorties générées :

```text
reports/xai/resnet50/test/
├── grad_cam_predictions.csv
├── grad_cam_examples.csv
├── grad_cam_examples.json
├── grad_cam_summary_grid.png
└── examples/
    ├── *_original.png
    ├── *_heatmap.png
    └── *_gradcam_overlay.png
```

Le fichier le plus directement exploitable dans le rapport est :

```text
reports/xai/resnet50/test/grad_cam_summary_grid.png
```

Il affiche côte à côte, pour chaque exemple :

- l’image originale ;
- la classe réelle ;
- l’overlay Grad-CAM ;
- la classe prédite ;
- le score de confiance.

Les fichiers tabulaires sont utiles pour commenter les résultats :

```text
reports/xai/resnet50/test/grad_cam_examples.csv
reports/xai/resnet50/test/grad_cam_examples.json
```

Chaque ligne contient notamment :

- le chemin de l’image ;
- la classe réelle ;
- la classe prédite ;
- la confiance du modèle ;
- le type d’exemple : `correct` ou `error` ;
- la classe cible utilisée pour Grad-CAM ;
- les chemins vers l’image originale, la heatmap et l’overlay.

### Interprétation attendue dans le rapport

L’analyse doit répondre aux questions suivantes :

- le modèle regarde-t-il principalement la zone de plaie ?
- les erreurs correspondent-elles à des zones d’attention ambiguës ?
- le modèle semble-t-il parfois utiliser l’arrière-plan ou des artefacts visuels ?
- les prédictions très confiantes sont-elles cohérentes avec les heatmaps ?
- les erreurs entre classes proches, comme `Cut` et `Laceration`, peuvent-elles s’expliquer visuellement ?

Exemple d’interprétation possible :

> Les heatmaps Grad-CAM montrent que le modèle se concentre souvent sur les zones rouges, ouvertes ou texturées correspondant à la plaie. Sur plusieurs bonnes classifications, la zone chaude recouvre directement la lésion. Certaines erreurs restent toutefois explicables : le modèle confond parfois des coupures et lacérations lorsque la forme linéaire de la blessure est proche. Dans quelques cas, l’attention peut aussi s’étendre à des éléments périphériques, ce qui rappelle que le modèle reste un outil pédagogique non validé médicalement.

### Limite médicale

Les heatmaps Grad-CAM aident à interpréter le comportement du modèle, mais elles ne prouvent pas que la décision est cliniquement correcte.

Cette application reste un projet pédagogique. Elle ne constitue pas un dispositif médical et ne remplace pas l’avis d’un professionnel de santé.

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
mlflow ui \
  --host 127.0.0.1 \
  --backend-store-uri sqlite:///mlflow.db \
  --port 5001 \
  --allowed-hosts "*" \
  --cors-allowed-origins "*"
```

Puis ouvrir :

```text
http://127.0.0.1:5001
```

Depuis Streamlit, cette commande est lancée automatiquement quand un entraînement démarre depuis la page Training.

### Lancer Streamlit

```bash
streamlit run app-streamlit/Home.py
```

### Préparer l’assistant IA / RAG

Indexer la base de connaissances médicale :

```bash
python3 scripts/create_medical_kb.py
```

Forcer la réindexation après modification de `data/base_connaissances_medicales.jsonl` :

```bash
python3 scripts/create_medical_kb.py --reset
```

Lancer le LLM local recommandé :

```bash
ollama pull llama3.2
ollama serve
```

Sans Ollama, le pipeline tente un fallback HuggingFace local si `transformers` et `accelerate` sont installés.

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

### Générer le rapport Grad-CAM

```bash
python3 scripts/generate_grad_cam_report.py \
  --checkpoint models/resnet50_best.pt \
  --split test \
  --num-correct 5 \
  --num-errors 3
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

### Entraîner le VAE OOD

```bash
python3 scripts/train_vae.py \
  --epochs 100 \
  --batch-size 16 \
  --latent-dim 256 \
  --beta 0.001 \
  --lr 1e-4 \
  --patience 10 \
  --run-suffix final
```

### Évaluer le VAE OOD

```bash
python3 scripts/evaluate_vae_ood.py \
  --checkpoint models/conv_vae_latent-256_beta-0.001_lr-0.0001_final_best.pt \
  --threshold-percentile 95 \
  --score combined \
  --alpha 0.001 \
  --batch-size 16 \
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

## État actuel et prochaines étapes

L’application Streamlit est maintenant branchée sur le pipeline principal :

1. prédiction avec filtre OOD ;
2. classification CNN ;
3. affichage du top-3 ;
4. overlay Grad-CAM directement sous l’image uploadée ;
5. recherche de cas similaires ;
6. assistant RAG médical pédagogique avec base ChromaDB ;
7. lancement d’entraînement depuis la page Training avec suivi MLflow.

Les prochaines étapes utiles sont :

1. enrichir le jeu OOD avec davantage d’images variées ;
2. lancer plusieurs runs comparables dans `wound-classification-app` ;
3. exploiter les rapports CNN, OOD, similarité et XAI dans le rapport académique ;
4. ajouter un rapport Evidently complet sur les embeddings de référence et de production simulée ;
5. compléter les tests automatisés autour du RAG, de la similarité et du pipeline Streamlit.

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
