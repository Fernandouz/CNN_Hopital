from pathlib import Path

from PIL import Image

from core.grad_cam import GradCAMExplainer
from core.inference import predict_image
from core.ood_detection import detect_ood


def _normalize_target_class(target_class):
    if target_class is None:
        return None

    if isinstance(target_class, str) and target_class.isdigit():
        return int(target_class)

    return target_class


def _build_prediction_from_grad_cam_result(result):
    return {
        "image_path": result.image_path,
        "architecture": result.architecture,
        "predicted_class": result.predicted_class,
        "confidence": result.confidence,
        "top_k": [
            {
                "rank": item["rank"],
                "class": item["class"],
                "probability": item["probability"],
            }
            for item in result.top_k
        ],
    }


def _save_grad_cam_overlay(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = output_dir / f"{Path(result.image_path).stem}_gradcam_overlay.png"
    Image.fromarray(result.overlay_rgb).save(overlay_path)

    return str(overlay_path)


def analyze_image(
    image_path,
    cnn_checkpoint_path,
    autoencoder_checkpoint_path,
    ood_threshold_path,
    top_k=3,
    include_grad_cam=False,
    grad_cam_output_dir="reports/xai/pipeline",
    grad_cam_target_class=None,
    grad_cam_alpha=0.45,
):
    """
    Pipeline complet :
    1. détection hors domaine avec autoencoder ;
    2. si l'image est OOD, rejet ;
    3. sinon, classification CNN ;
    4. optionnellement, explication Grad-CAM pour Streamlit ou rapport.
    """

    ood_result = detect_ood(
        image_path=image_path,
        autoencoder_checkpoint_path=autoencoder_checkpoint_path,
        threshold_path=ood_threshold_path,
    )

    if ood_result["is_ood"]:
        return {
            "status": "rejected",
            "reason": "Image rejetée par le filtre OOD",
            "ood": ood_result,
            "prediction": None,
            "grad_cam": None,
        }

    if include_grad_cam:
        explainer = GradCAMExplainer(
            checkpoint_path=cnn_checkpoint_path,
        )
        grad_cam_result = explainer.explain_image(
            image_path=image_path,
            target_class=_normalize_target_class(grad_cam_target_class),
            top_k=top_k,
            alpha=grad_cam_alpha,
        )

        overlay_path = _save_grad_cam_overlay(
            result=grad_cam_result,
            output_dir=grad_cam_output_dir,
        )

        prediction = _build_prediction_from_grad_cam_result(grad_cam_result)
        grad_cam = {
            "target_class": grad_cam_result.target_class,
            "target_class_idx": grad_cam_result.target_class_idx,
            "alpha": grad_cam_alpha,
            "overlay_path": overlay_path,
        }
    else:
        prediction = predict_image(
            image_path=image_path,
            checkpoint_path=cnn_checkpoint_path,
            top_k=top_k,
        )
        grad_cam = None

    return {
        "status": "accepted",
        "reason": "Image acceptée par le filtre OOD puis classifiée par le CNN",
        "ood": ood_result,
        "prediction": prediction,
        "grad_cam": grad_cam,
    }
