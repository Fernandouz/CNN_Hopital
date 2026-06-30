from pathlib import Path
import json

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from core.autoencoder import ConvAutoencoder, reconstruction_error


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_ood_transform(img_size=224):
    """
    Transform utilisée pour l'autoencoder.

    Important :
    on ne normalise pas avec ImageNet mean/std car l'autoencoder
    reconstruit des images dans [0, 1] avec une sortie Sigmoid.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])


def load_threshold(threshold_path):
    threshold_path = Path(threshold_path)

    if not threshold_path.exists():
        raise FileNotFoundError(
            f"Fichier de seuil OOD introuvable : {threshold_path}")

    with open(threshold_path, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)

    if "threshold" not in threshold_data:
        raise KeyError(
            "Le fichier threshold JSON doit contenir une clé 'threshold'.")

    return threshold_data


def load_autoencoder_from_checkpoint(checkpoint_path, device=None):
    checkpoint_path = Path(checkpoint_path)

    if device is None:
        device = get_device()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint autoencoder introuvable : {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = ConvAutoencoder(
        latent_dim=checkpoint["latent_dim"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint, device


@torch.no_grad()
def compute_image_reconstruction_error(image_path, autoencoder_checkpoint_path):
    """
    Calcule l'erreur de reconstruction d'une image unique.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    model, checkpoint, device = load_autoencoder_from_checkpoint(
        autoencoder_checkpoint_path
    )

    img_size = checkpoint.get("img_size", 224)
    transform = get_ood_transform(img_size=img_size)

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    reconstruction = model(image_tensor)

    error = reconstruction_error(
        image_tensor,
        reconstruction,
        reduction="none"
    )

    return float(error.item())


def detect_ood(image_path, autoencoder_checkpoint_path, threshold_path):
    """
    Détecte si une image est hors domaine à partir de l'erreur de reconstruction.

    Retour :
    - is_ood : bool
    - anomaly_score : erreur de reconstruction
    - threshold : seuil retenu
    - decision : texte lisible
    """
    threshold_data = load_threshold(threshold_path)
    threshold = float(threshold_data["threshold"])

    anomaly_score = compute_image_reconstruction_error(
        image_path=image_path,
        autoencoder_checkpoint_path=autoencoder_checkpoint_path,
    )

    is_ood = anomaly_score > threshold

    result = {
        "image_path": str(image_path),
        "is_ood": bool(is_ood),
        "anomaly_score": anomaly_score,
        "threshold": threshold,
        "threshold_method": threshold_data.get("method", "unknown"),
        "threshold_percentile": threshold_data.get("threshold_percentile"),
        "decision": "Image hors domaine" if is_ood else "Image acceptée",
    }

    return result
