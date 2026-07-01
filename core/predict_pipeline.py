"""Adaptateur Streamlit pour prediction, OOD et similarite visuelle."""

from pathlib import Path

from core.config import (
    AUTOENCODER_CHECKPOINT,
    AVAILABLE_CHECKPOINTS,
    CHROMA_DIR,
    DEFAULT_ARCHITECTURE,
    OOD_THRESHOLD_PATH,
)
from core.image_similarity import (
    DEFAULT_COLLECTION_NAME,
    get_chroma_collection,
    search_similar_images,
)
from core.inference import predict_image
from core.ood_detection import detect_ood


def load_pipeline_resources(architecture=DEFAULT_ARCHITECTURE):
    """Valide et retourne les ressources utilisees par la page Prediction."""
    if architecture not in AVAILABLE_CHECKPOINTS:
        raise ValueError(f"Architecture inconnue : {architecture}")

    checkpoint_path = Path(AVAILABLE_CHECKPOINTS[architecture])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint CNN introuvable : {checkpoint_path}")

    has_ood_filter = AUTOENCODER_CHECKPOINT.exists() and OOD_THRESHOLD_PATH.exists()

    return {
        "architecture": architecture,
        "checkpoint_path": checkpoint_path,
        "autoencoder_checkpoint_path": AUTOENCODER_CHECKPOINT,
        "ood_threshold_path": OOD_THRESHOLD_PATH,
        "has_ood_filter": has_ood_filter,
        "chroma_dir": CHROMA_DIR,
        "collection_name": DEFAULT_COLLECTION_NAME,
    }


def _run_ood_filter(resources, image_path):
    if not resources["has_ood_filter"]:
        return {
            "is_ood": False,
            "score": 0.0,
            "threshold": float("inf"),
            "decision": "Filtre OOD indisponible : image acceptee par defaut",
            "enabled": False,
        }

    result = detect_ood(
        image_path=image_path,
        autoencoder_checkpoint_path=resources["autoencoder_checkpoint_path"],
        threshold_path=resources["ood_threshold_path"],
    )
    result["score"] = result.get("anomaly_score", result.get("score", 0.0))
    result["enabled"] = True
    return result


def _format_classification(prediction):
    return {
        "architecture": prediction["architecture"],
        "classe_predite": prediction["predicted_class"],
        "confiance": prediction["confidence"],
        "top3": [
            {
                "rang": item["rank"],
                "classe": item["class"],
                "score": item["probability"],
            }
            for item in prediction["top_k"]
        ],
    }


def _search_similar_cases(resources, image_path, k_similar):
    try:
        collection = get_chroma_collection(
            persist_dir=resources["chroma_dir"],
            collection_name=resources["collection_name"],
        )
        if collection.count() == 0:
            return []

        results = search_similar_images(
            query_image_path=image_path,
            checkpoint_path=resources["checkpoint_path"],
            persist_dir=resources["chroma_dir"],
            collection_name=resources["collection_name"],
            top_k=k_similar,
            exclude_image_path=image_path,
        )
    except (ImportError, FileNotFoundError, ValueError, RuntimeError):
        return []

    return [
        {
            "rang": item["rank"],
            "filepath": item["image_path"],
            "classe": item["class"],
            "similarite": item["score"],
        }
        for item in results
    ]


def run_full_pipeline(resources, image_path, k_similar=5):
    """Execute le pipeline complet attendu par l'interface Streamlit."""
    ood_result = _run_ood_filter(resources, image_path)

    if ood_result["is_ood"]:
        return {
            "ood": ood_result,
            "classification": None,
            "similar_cases": [],
        }

    prediction = predict_image(
        image_path=image_path,
        checkpoint_path=resources["checkpoint_path"],
        top_k=3,
    )

    return {
        "ood": ood_result,
        "classification": _format_classification(prediction),
        "similar_cases": _search_similar_cases(resources, image_path, k_similar),
    }
