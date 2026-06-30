from core.ood_detection import detect_ood
from core.inference import predict_image


def analyze_image(
    image_path,
    cnn_checkpoint_path,
    autoencoder_checkpoint_path,
    ood_threshold_path,
    top_k=3,
):
    """
    Pipeline complet :
    1. détection hors domaine avec autoencoder ;
    2. si l'image est OOD, rejet ;
    3. sinon, classification CNN.
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
        }

    prediction = predict_image(
        image_path=image_path,
        checkpoint_path=cnn_checkpoint_path,
        top_k=top_k,
    )

    return {
        "status": "accepted",
        "reason": "Image acceptée par le filtre OOD puis classifiée par le CNN",
        "ood": ood_result,
        "prediction": prediction,
    }
