"""Configuration centrale du projet CNN Hopital."""

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_PROCESSED_DIR / "splits"
CHROMA_DIR = DATA_PROCESSED_DIR / "chroma"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TMP_UPLOAD_DIR = DATA_PROCESSED_DIR / "_streamlit_uploads"

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
SEED = 42

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

DEFAULT_ARCHITECTURE = "resnet50"

AVAILABLE_CHECKPOINTS = {
    "resnet50": MODELS_DIR / "resnet50_best.pt",
    "efficientnet_b0": MODELS_DIR
    / "efficientnet_b0_pretrained-True_freeze-True_finetune-False_weighted-True_classweights-False_lr-0.0001_efficientnet_b0_frozen_final_100epochs_20260625_1618_best.pt",
    "mobilenet_v3_large": MODELS_DIR
    / "mobilenet_v3_large_pretrained-True_freeze-True_finetune-False_weighted-True_classweights-False_lr-0.0001_mobilenet_v3_large_frozen_final_100epochs_20260625_1618_best.pt",
}

AUTOENCODER_CHECKPOINT = (
    MODELS_DIR / "conv_autoencoder_latent-256_lr-0.0001_final_best.pt"
)
OOD_THRESHOLD_PATH = (
    REPORTS_DIR
    / "ood"
    / "conv_autoencoder_latent-256_lr-0.0001_final"
    / "evaluation"
    / "ood_threshold.json"
)


def get_class_names():
    """Retourne les classes depuis les splits, sinon depuis data/raw."""
    train_csv = SPLIT_DIR / "train.csv"
    if train_csv.exists():
        import pandas as pd

        train_df = pd.read_csv(train_csv)
        if "class" in train_df.columns:
            return sorted(train_df["class"].dropna().unique().tolist())

    if DATA_RAW_DIR.exists():
        return sorted(
            path.name
            for path in DATA_RAW_DIR.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        )

    return [
        "Abrasions",
        "Bruises",
        "Burns",
        "Cut",
        "Ingrown_nails",
        "Laceration",
        "Stab_wound",
    ]


CLASS_NAMES = get_class_names()
CLASS_TO_IDX = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: class_name for class_name, idx in CLASS_TO_IDX.items()}
