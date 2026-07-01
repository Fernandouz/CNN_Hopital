from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from core.inference import get_inference_transform, load_model_from_checkpoint


@dataclass
class GradCAMResult:
    """Resultat complet d'une explication Grad-CAM pour une image."""

    image_path: str
    architecture: str
    predicted_class: str
    predicted_class_idx: int
    confidence: float
    target_class: str
    target_class_idx: int
    top_k: list[dict]
    original_rgb: np.ndarray
    heatmap: np.ndarray
    overlay_rgb: np.ndarray


def get_last_conv_layer(model, architecture):
    """
    Retourne la couche convolutive cible pour Grad-CAM.

    La couche choisie est volontairement la derniere couche spatiale du backbone :
    elle conserve une semantique forte tout en gardant une carte 2D exploitable.
    """
    architecture = architecture.lower()

    if architecture.startswith("resnet"):
        return model.layer4[-1]

    if architecture == "efficientnet_b0":
        return model.features[-1]

    if architecture == "mobilenet_v3_large":
        return model.features[-1]

    if architecture == "vgg16":
        for layer in reversed(model.features):
            if isinstance(layer, nn.Conv2d):
                return layer

    if architecture == "custom_cnn":
        for layer in reversed(model.features):
            if isinstance(layer, nn.Conv2d):
                return layer

    raise ValueError(
        f"Architecture non supportee pour Grad-CAM : {architecture}"
    )


def _resolve_image_path(image_path, project_root=None):
    image_path = Path(image_path)

    if image_path.is_absolute():
        return image_path

    root = Path(project_root) if project_root is not None else Path.cwd()

    if str(image_path).startswith("../data/"):
        return root / str(image_path).replace("../data/", "data/")

    return root / image_path


def _load_original_rgb(image_path, img_size):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
    return np.asarray(image).astype(np.float32) / 255.0


def _resolve_target_class(target_class, predicted_idx, class_names):
    if target_class is None:
        return predicted_idx

    if isinstance(target_class, str):
        if target_class not in class_names:
            raise ValueError(
                f"Classe cible inconnue : {target_class}. "
                f"Classes disponibles : {class_names}"
            )
        return class_names.index(target_class)

    target_idx = int(target_class)

    if target_idx < 0 or target_idx >= len(class_names):
        raise ValueError(
            f"Indice de classe cible invalide : {target_idx}. "
            f"Nombre de classes : {len(class_names)}"
        )

    return target_idx


def _predict_top_k(model, input_tensor, class_names, top_k):
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    k = min(top_k, len(class_names))
    top_probs, top_indices = torch.topk(probabilities, k=k)

    predictions = []
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
        class_idx = int(idx.item())
        predictions.append({
            "rank": rank,
            "class": class_names[class_idx],
            "class_idx": class_idx,
            "probability": float(prob.item()),
        })

    return predictions


def _make_overlay(original_rgb, heatmap, alpha=0.45):
    # Utilise l'utilitaire officiel de pytorch-grad-cam pour une superposition
    # stable et lisible, avec sortie RGB en uint8.
    from pytorch_grad_cam.utils.image import show_cam_on_image

    return show_cam_on_image(
        original_rgb.astype(np.float32),
        heatmap.astype(np.float32),
        use_rgb=True,
        image_weight=1.0 - alpha,
    )


class GradCAMExplainer:
    """Explainer Grad-CAM reutilisable pour un checkpoint CNN."""

    def __init__(self, checkpoint_path, device=None, project_root=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.project_root = Path(project_root) if project_root is not None else None
        self.model, self.checkpoint, self.device = load_model_from_checkpoint(
            checkpoint_path=self.checkpoint_path,
            device=device,
        )
        self.architecture = self.checkpoint["architecture"]
        self.class_names = self.checkpoint["class_names"]
        self.img_size = self.checkpoint.get("img_size", 224)
        self.transform = get_inference_transform(img_size=self.img_size)
        self.target_layers = [
            get_last_conv_layer(self.model, self.architecture)
        ]

    def explain_image(
        self,
        image_path,
        target_class=None,
        top_k=3,
        alpha=0.45,
    ):
        """
        Genere une explication Grad-CAM.

        `target_class` peut etre `None` (classe predite), un nom de classe ou un
        indice de classe.
        """
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        image_path = _resolve_image_path(
            image_path,
            project_root=self.project_root,
        )

        if not image_path.exists():
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        pil_image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        top_predictions = _predict_top_k(
            model=self.model,
            input_tensor=input_tensor,
            class_names=self.class_names,
            top_k=top_k,
        )

        best_prediction = top_predictions[0]
        predicted_idx = best_prediction["class_idx"]
        target_idx = _resolve_target_class(
            target_class=target_class,
            predicted_idx=predicted_idx,
            class_names=self.class_names,
        )

        targets = [ClassifierOutputTarget(target_idx)]

        with GradCAM(model=self.model, target_layers=self.target_layers) as cam:
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets,
            )[0]

        original_rgb = _load_original_rgb(image_path, img_size=self.img_size)
        overlay_rgb = _make_overlay(
            original_rgb=original_rgb,
            heatmap=grayscale_cam,
            alpha=alpha,
        )

        return GradCAMResult(
            image_path=str(image_path),
            architecture=self.architecture,
            predicted_class=best_prediction["class"],
            predicted_class_idx=predicted_idx,
            confidence=best_prediction["probability"],
            target_class=self.class_names[target_idx],
            target_class_idx=target_idx,
            top_k=top_predictions,
            original_rgb=(original_rgb * 255).astype(np.uint8),
            heatmap=grayscale_cam.astype(np.float32),
            overlay_rgb=overlay_rgb,
        )


def explain_image_with_grad_cam(
    checkpoint_path,
    image_path,
    target_class=None,
    top_k=3,
    device=None,
    project_root=None,
    alpha=0.45,
):
    """
    Genere une explication Grad-CAM pour une image et un checkpoint CNN.

    Les tableaux RGB retournes sont directement affichables avec PIL,
    matplotlib ou Streamlit.
    """
    explainer = GradCAMExplainer(
        checkpoint_path=checkpoint_path,
        device=device,
        project_root=project_root,
    )
    return explainer.explain_image(
        image_path=image_path,
        target_class=target_class,
        top_k=top_k,
        alpha=alpha,
    )


def save_grad_cam_result(result, output_dir, stem=None):
    """Sauvegarde image originale, heatmap et superposition Grad-CAM."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stem is None:
        stem = Path(result.image_path).stem

    original_path = output_dir / f"{stem}_original.png"
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    overlay_path = output_dir / f"{stem}_gradcam_overlay.png"

    Image.fromarray(result.original_rgb).save(original_path)

    heatmap_uint8 = (np.clip(result.heatmap, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(heatmap_uint8).save(heatmap_path)

    Image.fromarray(result.overlay_rgb).save(overlay_path)

    return {
        "original_path": str(original_path),
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
    }
