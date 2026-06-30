from pathlib import Path
import hashlib
import os
import textwrap

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from core.inference import (
    get_inference_transform,
    load_model_from_checkpoint,
)


DEFAULT_COLLECTION_NAME = "wound_image_embeddings"


class CNNFeatureExtractor(nn.Module):
    """
    Extrait les embeddings visuels avant la couche finale de classification.

    Le module est compatible avec les architectures construites dans
    `core.model_utils.build_model` et avec les checkpoints sauvegardes par
    `scripts/train_cnn.py`.
    """

    def __init__(self, model, architecture):
        super().__init__()
        self.model = model
        self.architecture = architecture.lower()

    def forward(self, images):
        if self.architecture == "custom_cnn":
            features = self.model.features(images)
            features = self.model.global_pool(features)
            return torch.flatten(features, 1)

        if self.architecture.startswith("resnet"):
            x = self.model.conv1(images)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)

        if self.architecture == "efficientnet_b0":
            x = self.model.features(images)
            x = self.model.avgpool(x)
            return torch.flatten(x, 1)

        if self.architecture == "mobilenet_v3_large":
            x = self.model.features(images)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            return self.model.classifier[:3](x)

        if self.architecture == "vgg16":
            x = self.model.features(images)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            return self.model.classifier[:6](x)

        raise ValueError(
            f"Architecture non supportee pour l'extraction d'embeddings : "
            f"{self.architecture}"
        )


def l2_normalize_embeddings(embeddings, eps=1e-12):
    """Normalise des embeddings en norme L2 pour la similarite cosinus."""
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)

    return embeddings / norms


class CNNEmbeddingExtractor:
    """
    Charge un checkpoint CNN et extrait des embeddings normalises.

    Exemple :
        extractor = CNNEmbeddingExtractor("models/best.pt")
        embedding = extractor.extract_image("data/raw/classe/image.jpg")
    """

    def __init__(self, checkpoint_path, device=None, normalize=True):
        self.checkpoint_path = Path(checkpoint_path)
        self.model, self.checkpoint, self.device = load_model_from_checkpoint(
            self.checkpoint_path,
            device=device,
        )
        self.architecture = self.checkpoint["architecture"]
        self.img_size = self.checkpoint.get("img_size", 224)
        self.normalize = normalize
        self.transform = get_inference_transform(img_size=self.img_size)

        self.feature_extractor = CNNFeatureExtractor(
            model=self.model,
            architecture=self.architecture,
        ).to(self.device)
        self.feature_extractor.eval()

    def _load_image_tensor(self, image_path):
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

    @torch.no_grad()
    def extract_tensor_batch(self, image_tensors):
        """Extrait les embeddings d'un batch de tenseurs deja transformes."""
        if image_tensors.ndim == 3:
            image_tensors = image_tensors.unsqueeze(0)

        image_tensors = image_tensors.to(self.device)
        embeddings = self.feature_extractor(image_tensors)
        embeddings = embeddings.detach().cpu().numpy().astype(np.float32)

        if self.normalize:
            embeddings = l2_normalize_embeddings(embeddings)

        return embeddings

    def extract_image(self, image_path):
        """Extrait un embedding normalise pour une image unique."""
        image_tensor = self._load_image_tensor(image_path)
        return self.extract_tensor_batch(image_tensor)[0]

    def extract_images(self, image_paths, batch_size=32):
        """Extrait des embeddings pour une liste d'images."""
        image_paths = [Path(path) for path in image_paths]
        all_embeddings = []

        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            batch_tensors = [
                self._load_image_tensor(path)
                for path in batch_paths
            ]
            batch = torch.stack(batch_tensors, dim=0)
            all_embeddings.append(self.extract_tensor_batch(batch))

        if not all_embeddings:
            return np.empty((0, 0), dtype=np.float32)

        return np.vstack(all_embeddings)


def extract_image_embedding(image_path, checkpoint_path, device=None):
    """Fonction pratique pour extraire l'embedding d'une seule image."""
    extractor = CNNEmbeddingExtractor(
        checkpoint_path=checkpoint_path,
        device=device,
        normalize=True,
    )
    return extractor.extract_image(image_path)


def compute_cosine_similarities(query_embedding, candidate_embeddings):
    """
    Calcule les similarites cosinus.

    Les embeddings sont normalises par securite, ce qui rend le produit
    scalaire equivalent a la similarite cosinus.
    """
    query_embedding = l2_normalize_embeddings(query_embedding)[0]
    candidate_embeddings = l2_normalize_embeddings(candidate_embeddings)

    return candidate_embeddings @ query_embedding


def find_top_k_similar(query_embedding, candidate_embeddings, metadata, top_k=5):
    """Retourne les top-K voisins les plus similaires hors base vectorielle."""
    similarities = compute_cosine_similarities(
        query_embedding=query_embedding,
        candidate_embeddings=candidate_embeddings,
    )

    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = []

    for rank, idx in enumerate(top_indices, start=1):
        item = dict(metadata[idx])
        item.update({
            "rank": rank,
            "score": float(similarities[idx]),
        })
        results.append(item)

    return results


def _require_chromadb():
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError(
            "ChromaDB n'est pas installe. Ajoute `chromadb` dans "
            "`requirements.txt` puis execute `pip install -r requirements.txt`."
        ) from exc

    return chromadb


def _make_document_id(image_path, index=None):
    raw = f"{index or ''}:{Path(image_path).as_posix()}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"img_{digest}"


def get_chroma_collection(
    persist_dir="data/processed/chroma",
    collection_name=DEFAULT_COLLECTION_NAME,
):
    """Ouvre ou cree une collection ChromaDB en distance cosinus."""
    chromadb = _require_chromadb()
    client = chromadb.PersistentClient(path=str(persist_dir))

    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def _resolve_existing_path(path):
    try:
        return Path(path).expanduser().resolve()
    except OSError:
        return Path(path).expanduser().absolute()


def _is_same_image_path(left_path, right_path):
    return _resolve_existing_path(left_path) == _resolve_existing_path(right_path)


def index_images_in_chroma(
    image_paths,
    labels,
    checkpoint_path,
    persist_dir="data/processed/chroma",
    collection_name=DEFAULT_COLLECTION_NAME,
    batch_size=32,
    device=None,
):
    """
    Indexe des images dans ChromaDB avec leurs embeddings CNN.

    `labels` doit contenir la classe reelle associee a chaque image.
    """
    image_paths = [Path(path) for path in image_paths]
    labels = list(labels)

    if len(image_paths) != len(labels):
        raise ValueError("image_paths et labels doivent avoir la meme taille.")

    extractor = CNNEmbeddingExtractor(
        checkpoint_path=checkpoint_path,
        device=device,
        normalize=True,
    )
    collection = get_chroma_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]
        embeddings = extractor.extract_images(batch_paths, batch_size=batch_size)

        ids = [
            _make_document_id(path, index=start + offset)
            for offset, path in enumerate(batch_paths)
        ]
        metadatas = [
            {
                "image_path": str(path),
                "class": label,
                "architecture": extractor.architecture,
                "checkpoint": str(Path(checkpoint_path)),
            }
            for path, label in zip(batch_paths, batch_labels)
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

    return collection


def search_similar_images(
    query_image_path,
    checkpoint_path,
    persist_dir="data/processed/chroma",
    collection_name=DEFAULT_COLLECTION_NAME,
    top_k=5,
    exclude_image_path=None,
    device=None,
):
    """Recherche les images les plus similaires a une image requete."""
    extractor = CNNEmbeddingExtractor(
        checkpoint_path=checkpoint_path,
        device=device,
        normalize=True,
    )
    query_embedding = extractor.extract_image(query_image_path)
    collection = get_chroma_collection(
        persist_dir=persist_dir,
        collection_name=collection_name,
    )

    return search_chroma_by_embedding(
        query_embedding=query_embedding,
        collection=collection,
        top_k=top_k,
        exclude_image_path=exclude_image_path,
    )


def search_chroma_by_embedding(
    query_embedding,
    collection,
    top_k=5,
    exclude_image_path=None,
):
    """Recherche les voisins ChromaDB a partir d'un embedding deja calcule."""
    n_results = top_k

    if exclude_image_path is not None:
        # On recupere quelques voisins en plus pour pouvoir retirer l'image
        # requete lorsqu'elle est deja presente dans l'index.
        n_results = min(collection.count(), top_k + 10)

    raw_results = collection.query(
        query_embeddings=[np.asarray(query_embedding, dtype=np.float32).tolist()],
        n_results=n_results,
        include=["metadatas", "distances"],
    )

    results = []
    ids = raw_results.get("ids", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    for doc_id, metadata, distance in zip(
        ids,
        metadatas,
        distances,
    ):
        image_path = metadata.get("image_path")

        if exclude_image_path is not None and _is_same_image_path(
            image_path,
            exclude_image_path,
        ):
            continue

        rank = len(results) + 1
        results.append({
            "rank": rank,
            "id": doc_id,
            "image_path": image_path,
            "class": metadata.get("class"),
            "score": float(1.0 - distance),
            "distance": float(distance),
        })

        if len(results) >= top_k:
            break

    return results


def save_similarity_figure(query_image_path, results, output_path, title=None):
    """Sauvegarde une figure query + top-K pour le rapport."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))
    os.environ.setdefault("MPLBACKEND", "Agg")

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    query_image_path = Path(query_image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_results = len(results)
    num_columns = num_results + 1

    fig, axes = plt.subplots(
        1,
        num_columns,
        figsize=(3.2 * num_columns, 3.8),
        constrained_layout=True,
    )

    if num_columns == 1:
        axes = [axes]

    query_image = Image.open(query_image_path).convert("RGB")
    axes[0].imshow(query_image)
    axes[0].set_title(
        "Query\n" + textwrap.shorten(query_image_path.name, width=28),
        fontsize=10,
    )
    axes[0].axis("off")

    for axis, result in zip(axes[1:], results):
        result_path = Path(result["image_path"])
        result_image = Image.open(result_path).convert("RGB")

        axis.imshow(result_image)
        axis.set_title(
            (
                f"Top {result['rank']} | {result['class']}\n"
                f"score={result['score']:.3f}\n"
                f"{textwrap.shorten(result_path.name, width=24)}"
            ),
            fontsize=9,
        )
        axis.axis("off")

    if title:
        fig.suptitle(title, fontsize=12)

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return output_path
