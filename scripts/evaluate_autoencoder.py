from pathlib import Path
import sys
import argparse
import json
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from core.autoencoder import ConvAutoencoder, reconstruction_error
from core.data_processing import load_splits, build_class_mapping, WoundDataset


class OODImageDataset(Dataset):
    """Dataset simple pour évaluer des images hors-domaine sans label."""

    def __init__(self, image_paths, transform):
        self.image_paths = [Path(path) for path in image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, str(image_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate autoencoder reconstruction errors and calibrate OOD threshold."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Chemin du checkpoint autoencoder .pt."
    )

    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile utilisé pour définir le seuil OOD sur validation."
    )

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--ood-dir",
        type=str,
        default="data/ood",
        help="Dossier contenant les images hors-domaine à tester."
    )

    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_autoencoder_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])


def load_autoencoder_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model = ConvAutoencoder(
        latent_dim=checkpoint["latent_dim"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def build_loader(dataframe, class_to_idx, transform, batch_size, num_workers):
    dataset = WoundDataset(
        dataframe=dataframe,
        transform=transform,
        class_to_idx=class_to_idx,
        project_root=PROJECT_ROOT,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return loader


def build_ood_loader(ood_dir, transform, batch_size, num_workers):
    ood_dir = Path(ood_dir)

    if not ood_dir.is_absolute():
        ood_dir = PROJECT_ROOT / ood_dir

    image_paths = sorted([
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        for path in ood_dir.glob(pattern)
    ])

    if not image_paths:
        return None, []

    dataset = OODImageDataset(
        image_paths=image_paths,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return loader, image_paths


@torch.no_grad()
def compute_reconstruction_errors(model, loader, device, split_name):
    rows = []

    sample_index = 0

    for images, labels in tqdm(loader, desc=f"Compute errors [{split_name}]"):
        images = images.to(device)

        reconstructions = model(images)
        errors = reconstruction_error(
            images,
            reconstructions,
            reduction="none"
        )

        for batch_idx, error in enumerate(errors.cpu().numpy().tolist()):
            rows.append({
                "split": split_name,
                "sample_index": sample_index,
                "reconstruction_error": float(error),
            })
            sample_index += 1

    return pd.DataFrame(rows)


@torch.no_grad()
def compute_ood_reconstruction_errors(model, loader, device, split_name="ood"):
    rows = []

    sample_index = 0

    for images, paths in tqdm(loader, desc=f"Compute errors [{split_name}]"):
        images = images.to(device)

        reconstructions = model(images)
        errors = reconstruction_error(
            images,
            reconstructions,
            reduction="none"
        )

        for image_path, error in zip(paths, errors.cpu().numpy().tolist()):
            rows.append({
                "split": split_name,
                "sample_index": sample_index,
                "image_path": image_path,
                "reconstruction_error": float(error),
            })
            sample_index += 1

    return pd.DataFrame(rows)


def save_error_distribution(errors_df, threshold, output_dir):
    plt.figure(figsize=(9, 5))

    for split_name in errors_df["split"].unique():
        split_errors = errors_df[
            errors_df["split"] == split_name
        ]["reconstruction_error"]

        plt.hist(
            split_errors,
            bins=30,
            alpha=0.5,
            label=split_name
        )

    plt.axvline(
        threshold,
        linestyle="--",
        label=f"Seuil OOD = {threshold:.6f}"
    )

    plt.xlabel("Erreur de reconstruction MSE")
    plt.ylabel("Nombre d'images")
    plt.title("Distribution des erreurs de reconstruction")
    plt.legend()
    plt.tight_layout()

    path = output_dir / "reconstruction_error_distribution.png"
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def save_error_boxplot(errors_df, threshold, output_dir):
    splits = list(errors_df["split"].unique())
    values = [
        errors_df[errors_df["split"] == split]["reconstruction_error"].values
        for split in splits
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(values, tick_labels=splits)
    plt.axhline(
        threshold,
        linestyle="--",
        label=f"Seuil OOD = {threshold:.6f}"
    )

    plt.ylabel("Erreur de reconstruction MSE")
    plt.title("Erreurs de reconstruction par split")
    plt.legend()
    plt.tight_layout()

    path = output_dir / "reconstruction_error_boxplot.png"
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def main():
    args = parse_args()

    device = get_device()
    checkpoint_path = PROJECT_ROOT / args.checkpoint

    print(f"Device utilisé : {device}")
    print(f"Checkpoint : {checkpoint_path}")

    model, checkpoint = load_autoencoder_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device
    )

    run_name = checkpoint_path.stem.replace("_best", "")

    output_dir = PROJECT_ROOT / "reports" / "ood" / run_name / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_dir = PROJECT_ROOT / "data" / "processed" / "splits"

    train_df, val_df, test_df = load_splits(split_dir)
    class_names, class_to_idx, idx_to_class = build_class_mapping(train_df)

    transform = get_autoencoder_transform(
        img_size=checkpoint.get("img_size", 224)
    )

    train_loader = build_loader(
        dataframe=train_df,
        class_to_idx=class_to_idx,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    val_loader = build_loader(
        dataframe=val_df,
        class_to_idx=class_to_idx,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    test_loader = build_loader(
        dataframe=test_df,
        class_to_idx=class_to_idx,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    ood_loader, ood_image_paths = build_ood_loader(
        ood_dir=args.ood_dir,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    train_errors = compute_reconstruction_errors(
        model=model,
        loader=train_loader,
        device=device,
        split_name="train"
    )

    val_errors = compute_reconstruction_errors(
        model=model,
        loader=val_loader,
        device=device,
        split_name="val"
    )

    test_errors = compute_reconstruction_errors(
        model=model,
        loader=test_loader,
        device=device,
        split_name="test"
    )

    error_frames = [train_errors, val_errors, test_errors]

    if ood_loader is not None:
        ood_errors = compute_ood_reconstruction_errors(
            model=model,
            loader=ood_loader,
            device=device,
            split_name="ood"
        )
        error_frames.append(ood_errors)
    else:
        ood_errors = pd.DataFrame()

    errors_df = pd.concat(
        error_frames,
        ignore_index=True
    )

    threshold = float(
        np.percentile(
            val_errors["reconstruction_error"],
            args.threshold_percentile
        )
    )

    errors_df["is_above_threshold"] = (
        errors_df["reconstruction_error"] > threshold
    )

    errors_path = output_dir / "reconstruction_errors.csv"
    errors_df.to_csv(errors_path, index=False)

    threshold_data = {
        "method": "validation_reconstruction_error_percentile",
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "model_checkpoint": str(checkpoint_path),
        "latent_dim": checkpoint["latent_dim"],
        "img_size": checkpoint.get("img_size", 224),
        "train_error_mean": float(train_errors["reconstruction_error"].mean()),
        "train_error_std": float(train_errors["reconstruction_error"].std()),
        "val_error_mean": float(val_errors["reconstruction_error"].mean()),
        "val_error_std": float(val_errors["reconstruction_error"].std()),
        "test_error_mean": float(test_errors["reconstruction_error"].mean()),
        "test_error_std": float(test_errors["reconstruction_error"].std()),
        "val_rejected_count": int(val_errors["reconstruction_error"].gt(threshold).sum()),
        "val_total": int(len(val_errors)),
        "ood_dir": str(PROJECT_ROOT / args.ood_dir),
        "ood_total": int(len(ood_errors)),
        "ood_rejected_count": int(ood_errors["reconstruction_error"].gt(threshold).sum()) if not ood_errors.empty else 0,
        "ood_error_mean": float(ood_errors["reconstruction_error"].mean()) if not ood_errors.empty else None,
        "ood_error_std": float(ood_errors["reconstruction_error"].std()) if not ood_errors.empty else None,
    }

    threshold_path = output_dir / "ood_threshold.json"
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=4)

    dist_path = save_error_distribution(
        errors_df=errors_df,
        threshold=threshold,
        output_dir=output_dir
    )

    boxplot_path = save_error_boxplot(
        errors_df=errors_df,
        threshold=threshold,
        output_dir=output_dir
    )

    print("\nÉvaluation autoencoder terminée.")
    print(
        f"Seuil OOD percentile {args.threshold_percentile} : {threshold:.8f}")
    print(f"Scores sauvegardés : {errors_path}")
    print(f"Seuil sauvegardé : {threshold_path}")
    print(f"Figure distribution : {dist_path}")
    print(f"Figure boxplot : {boxplot_path}")


if __name__ == "__main__":
    main()
