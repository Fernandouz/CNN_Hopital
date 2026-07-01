from pathlib import Path
import json

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from core.vae import ConvVAE, combined_ood_score, kl_divergence, reconstruction_error


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_vae_ood_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])


def load_vae_threshold(threshold_path):
    threshold_path = Path(threshold_path)

    if not threshold_path.exists():
        raise FileNotFoundError(f"Fichier de seuil VAE introuvable : {threshold_path}")

    with open(threshold_path, "r", encoding="utf-8") as file:
        threshold_data = json.load(file)

    if "threshold" not in threshold_data:
        raise KeyError("Le fichier JSON doit contenir une cle 'threshold'.")

    return threshold_data


def load_vae_from_checkpoint(checkpoint_path, device=None):
    checkpoint_path = Path(checkpoint_path)

    if device is None:
        device = get_device()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint VAE introuvable : {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = ConvVAE(latent_dim=checkpoint["latent_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint, device


def select_score(reconstruction_value, kl_value, score_name, alpha):
    if score_name == "reconstruction":
        return reconstruction_value

    if score_name == "kl":
        return kl_value

    if score_name == "combined":
        return float(
            combined_ood_score(
                reconstruction_errors=reconstruction_value,
                kl_values=kl_value,
                alpha=alpha,
            )
        )

    raise ValueError(f"Score VAE non supporte : {score_name}")


@torch.no_grad()
def compute_image_vae_ood_score(image_path, vae_checkpoint_path, threshold_data=None):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    model, checkpoint, device = load_vae_from_checkpoint(vae_checkpoint_path)

    img_size = checkpoint.get("img_size", 224)
    transform = get_vae_ood_transform(img_size=img_size)

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    reconstruction, mu, logvar = model(image_tensor)
    reconstruction_value = float(
        reconstruction_error(
            image_tensor,
            reconstruction,
            reduction="none",
        ).item()
    )
    kl_value = float(kl_divergence(mu, logvar, reduction="none").item())

    score_name = "combined"
    alpha = checkpoint.get("beta", 0.001)

    if threshold_data is not None:
        score_name = threshold_data.get("score", score_name)
        alpha = float(threshold_data.get("alpha", alpha))

    anomaly_score = select_score(
        reconstruction_value=reconstruction_value,
        kl_value=kl_value,
        score_name=score_name,
        alpha=alpha,
    )

    return {
        "image_path": str(image_path),
        "score": score_name,
        "alpha": alpha,
        "anomaly_score": float(anomaly_score),
        "reconstruction_error": reconstruction_value,
        "kl_divergence": kl_value,
        "latent_dim": checkpoint["latent_dim"],
        "beta": checkpoint.get("beta"),
    }


def detect_ood_vae(image_path, vae_checkpoint_path, threshold_path):
    threshold_data = load_vae_threshold(threshold_path)
    threshold = float(threshold_data["threshold"])

    score_data = compute_image_vae_ood_score(
        image_path=image_path,
        vae_checkpoint_path=vae_checkpoint_path,
        threshold_data=threshold_data,
    )

    is_ood = score_data["anomaly_score"] > threshold

    return {
        **score_data,
        "is_ood": bool(is_ood),
        "threshold": threshold,
        "threshold_method": threshold_data.get("method", "unknown"),
        "threshold_percentile": threshold_data.get("threshold_percentile"),
        "decision": "Image hors domaine" if is_ood else "Image acceptee",
    }
