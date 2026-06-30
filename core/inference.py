from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from core.data_processing import IMAGENET_MEAN, IMAGENET_STD
from core.model_utils import build_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_inference_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_checkpoint(checkpoint_path, device=None):
    checkpoint_path = Path(checkpoint_path)

    if device is None:
        device = get_device()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    return checkpoint


def load_model_from_checkpoint(checkpoint_path, device=None):
    if device is None:
        device = get_device()

    checkpoint = load_checkpoint(checkpoint_path, device=device)

    model = build_model(
        architecture=checkpoint["architecture"],
        num_classes=len(checkpoint["class_names"]),
        pretrained=False,
        dropout_rate=0.3
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint, device


@torch.no_grad()
def predict_image(
    image_path,
    checkpoint_path,
    top_k=3,
):
    model, checkpoint, device = load_model_from_checkpoint(checkpoint_path)

    img_size = checkpoint.get("img_size", 224)
    class_names = checkpoint["class_names"]

    transform = get_inference_transform(img_size=img_size)

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = torch.topk(probabilities, k=top_k)

    top_predictions = []

    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
        idx = int(idx.item())
        top_predictions.append({
            "rank": rank,
            "class": class_names[idx],
            "probability": float(prob.item())
        })

    best_prediction = top_predictions[0]

    result = {
        "image_path": str(image_path),
        "architecture": checkpoint["architecture"],
        "predicted_class": best_prediction["class"],
        "confidence": best_prediction["probability"],
        "top_k": top_predictions,
    }

    return result
