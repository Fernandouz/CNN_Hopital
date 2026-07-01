from pathlib import Path
import argparse
import json
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from core.data_processing import WoundDataset, build_class_mapping, load_splits
from core.vae import ConvVAE, combined_ood_score, kl_divergence, reconstruction_error


class OODImageDataset(Dataset):
    """Simple dataset for unlabeled out-of-domain images."""

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
        description="Evaluate VAE OOD detection with reconstruction, KL, or combined scores."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Chemin du checkpoint VAE .pt.",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95.0,
        help="Percentile validation utilise pour definir le seuil OOD.",
    )
    parser.add_argument(
        "--score",
        type=str,
        default="combined",
        choices=["reconstruction", "kl", "combined"],
        help="Score OOD utilise pour calibrer le seuil.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="Poids de la KL dans le score combine.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--ood-dir",
        type=str,
        default="data/ood",
        help="Dossier contenant les images hors-domaine a tester.",
    )
    parser.add_argument(
        "--image-extensions",
        type=str,
        default=".jpg,.jpeg,.png,.bmp,.webp",
        help="Extensions image OOD separees par des virgules.",
    )
    parser.add_argument("--num-reconstruction-examples", type=int, default=8)

    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_vae_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])


def load_vae_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model = ConvVAE(latent_dim=checkpoint["latent_dim"])
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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def parse_image_extensions(raw_extensions):
    extensions = []

    for extension in raw_extensions.split(","):
        extension = extension.strip().lower()
        if not extension:
            continue
        if not extension.startswith("."):
            extension = f".{extension}"
        extensions.append(extension)

    return tuple(extensions)


def build_ood_loader(ood_dir, transform, batch_size, num_workers, image_extensions):
    ood_dir = Path(ood_dir)
    if not ood_dir.is_absolute():
        ood_dir = PROJECT_ROOT / ood_dir

    image_paths = sorted([
        path
        for path in ood_dir.iterdir()
        if path.is_file() and path.suffix.lower() in image_extensions
    ])

    if not image_paths:
        return None, []

    dataset = OODImageDataset(
        image_paths=image_paths,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return loader, image_paths


def select_score(reconstruction_values, kl_values, score_name, alpha):
    if score_name == "reconstruction":
        return reconstruction_values

    if score_name == "kl":
        return kl_values

    if score_name == "combined":
        return combined_ood_score(
            reconstruction_errors=reconstruction_values,
            kl_values=kl_values,
            alpha=alpha,
        )

    raise ValueError(f"Score non supporte : {score_name}")


@torch.no_grad()
def compute_vae_scores(model, loader, device, split_name, score_name, alpha, has_paths=False):
    rows = []
    sample_index = 0

    for batch in tqdm(loader, desc=f"Compute VAE scores [{split_name}]"):
        if has_paths:
            images, paths = batch
        else:
            images, _ = batch
            paths = [None] * images.size(0)

        images = images.to(device)
        reconstructions, mu, logvar = model(images)
        reconstruction_values = reconstruction_error(
            images,
            reconstructions,
            reduction="none",
        ).cpu().numpy()
        kl_values = kl_divergence(
            mu,
            logvar,
            reduction="none",
        ).cpu().numpy()
        anomaly_scores = select_score(
            reconstruction_values=reconstruction_values,
            kl_values=kl_values,
            score_name=score_name,
            alpha=alpha,
        )

        for image_path, recon, kl_value, anomaly_score in zip(
            paths,
            reconstruction_values,
            kl_values,
            anomaly_scores,
        ):
            row = {
                "split": split_name,
                "sample_index": sample_index,
                "reconstruction_error": float(recon),
                "kl_divergence": float(kl_value),
                "anomaly_score": float(anomaly_score),
            }
            if image_path is not None:
                row["image_path"] = image_path
            rows.append(row)
            sample_index += 1

    return pd.DataFrame(rows)


def save_score_distribution(scores_df, threshold, score_name, output_dir):
    plt.figure(figsize=(9, 5))

    for split_name in scores_df["split"].unique():
        split_scores = scores_df[scores_df["split"] == split_name]["anomaly_score"]
        plt.hist(
            split_scores,
            bins=30,
            alpha=0.5,
            label=split_name,
        )

    plt.axvline(
        threshold,
        linestyle="--",
        color="black",
        label=f"Seuil OOD = {threshold:.6f}",
    )
    plt.xlabel(f"Score OOD VAE ({score_name})")
    plt.ylabel("Nombre d'images")
    plt.title("Distribution des scores OOD VAE")
    plt.legend()
    plt.tight_layout()

    path = output_dir / f"vae_{score_name}_score_distribution.png"
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def save_score_boxplot(scores_df, threshold, score_name, output_dir):
    splits = list(scores_df["split"].unique())
    values = [
        scores_df[scores_df["split"] == split]["anomaly_score"].values
        for split in splits
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(values, tick_labels=splits)
    plt.axhline(
        threshold,
        linestyle="--",
        color="black",
        label=f"Seuil OOD = {threshold:.6f}",
    )
    plt.ylabel(f"Score OOD VAE ({score_name})")
    plt.title("Scores OOD VAE par split")
    plt.legend()
    plt.tight_layout()

    path = output_dir / f"vae_{score_name}_score_boxplot.png"
    plt.savefig(path, dpi=300)
    plt.close()

    return path


@torch.no_grad()
def save_reconstruction_examples(model, loader, device, output_path, max_images=8):
    model.eval()
    images, _ = next(iter(loader))
    images = images[:max_images].to(device)

    reconstructions, _, _ = model(images)

    images = images.cpu()
    reconstructions = reconstructions.cpu()
    n = images.size(0)

    plt.figure(figsize=(3 * n, 6))

    for i in range(n):
        original = images[i].permute(1, 2, 0).numpy()
        reconstructed = reconstructions[i].permute(1, 2, 0).numpy()

        ax = plt.subplot(2, n, i + 1)
        ax.imshow(original)
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Original", fontsize=12)

        ax = plt.subplot(2, n, n + i + 1)
        ax.imshow(reconstructed)
        ax.axis("off")
        if i == 0:
            ax.set_ylabel("Reconstruction", fontsize=12)

    plt.suptitle("VAE reconstruction examples")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def build_threshold_summary(scores_df, threshold, args, checkpoint, checkpoint_path):
    summary = {
        "method": "validation_vae_score_percentile",
        "score": args.score,
        "alpha": args.alpha,
        "threshold_percentile": args.threshold_percentile,
        "threshold": float(threshold),
        "model_checkpoint": str(checkpoint_path),
        "model_type": checkpoint.get("model_type", "conv_vae"),
        "latent_dim": checkpoint["latent_dim"],
        "beta": checkpoint.get("beta"),
        "img_size": checkpoint.get("img_size", 224),
        "ood_dir": str(PROJECT_ROOT / args.ood_dir),
    }

    for split_name in scores_df["split"].unique():
        split_scores = scores_df[scores_df["split"] == split_name]
        prefix = split_name

        summary[f"{prefix}_total"] = int(len(split_scores))
        summary[f"{prefix}_rejected_count"] = int(
            split_scores["anomaly_score"].gt(threshold).sum()
        )
        summary[f"{prefix}_score_mean"] = float(split_scores["anomaly_score"].mean())
        summary[f"{prefix}_score_std"] = float(split_scores["anomaly_score"].std())
        summary[f"{prefix}_reconstruction_mean"] = float(
            split_scores["reconstruction_error"].mean()
        )
        summary[f"{prefix}_kl_mean"] = float(split_scores["kl_divergence"].mean())

    return summary


def main():
    args = parse_args()

    device = get_device()
    checkpoint_path = PROJECT_ROOT / args.checkpoint

    print(f"Device utilise : {device}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Score OOD : {args.score}")
    print(f"Alpha KL : {args.alpha}")

    model, checkpoint = load_vae_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    run_name = checkpoint_path.stem.replace("_best", "")
    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "ood"
        / run_name
        / f"evaluation_{args.score}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    split_dir = PROJECT_ROOT / "data" / "processed" / "splits"
    train_df, val_df, test_df = load_splits(split_dir)
    class_names, class_to_idx, idx_to_class = build_class_mapping(train_df)

    transform = get_vae_transform(img_size=checkpoint.get("img_size", 224))

    train_loader = build_loader(
        dataframe=train_df,
        class_to_idx=class_to_idx,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        dataframe=val_df,
        class_to_idx=class_to_idx,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    test_loader = build_loader(
        dataframe=test_df,
        class_to_idx=class_to_idx,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    ood_loader, ood_image_paths = build_ood_loader(
        ood_dir=args.ood_dir,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_extensions=parse_image_extensions(args.image_extensions),
    )

    train_scores = compute_vae_scores(
        model=model,
        loader=train_loader,
        device=device,
        split_name="train",
        score_name=args.score,
        alpha=args.alpha,
    )
    val_scores = compute_vae_scores(
        model=model,
        loader=val_loader,
        device=device,
        split_name="val",
        score_name=args.score,
        alpha=args.alpha,
    )
    test_scores = compute_vae_scores(
        model=model,
        loader=test_loader,
        device=device,
        split_name="test",
        score_name=args.score,
        alpha=args.alpha,
    )

    score_frames = [train_scores, val_scores, test_scores]

    if ood_loader is not None:
        ood_scores = compute_vae_scores(
            model=model,
            loader=ood_loader,
            device=device,
            split_name="ood",
            score_name=args.score,
            alpha=args.alpha,
            has_paths=True,
        )
        score_frames.append(ood_scores)
    else:
        ood_scores = pd.DataFrame()

    scores_df = pd.concat(score_frames, ignore_index=True)
    threshold = float(
        np.percentile(
            val_scores["anomaly_score"],
            args.threshold_percentile,
        )
    )
    scores_df["is_above_threshold"] = scores_df["anomaly_score"] > threshold

    scores_path = output_dir / "vae_ood_scores.csv"
    scores_df.to_csv(scores_path, index=False)

    threshold_data = build_threshold_summary(
        scores_df=scores_df,
        threshold=threshold,
        args=args,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
    )
    threshold_path = output_dir / "vae_ood_threshold.json"
    with open(threshold_path, "w", encoding="utf-8") as file:
        json.dump(threshold_data, file, indent=4)

    distribution_path = save_score_distribution(
        scores_df=scores_df,
        threshold=threshold,
        score_name=args.score,
        output_dir=output_dir,
    )
    boxplot_path = save_score_boxplot(
        scores_df=scores_df,
        threshold=threshold,
        score_name=args.score,
        output_dir=output_dir,
    )
    reconstruction_path = save_reconstruction_examples(
        model=model,
        loader=val_loader,
        device=device,
        output_path=output_dir / "vae_reconstruction_examples.png",
        max_images=args.num_reconstruction_examples,
    )

    print("\nEvaluation VAE terminee.")
    print(f"Seuil OOD percentile {args.threshold_percentile} : {threshold:.8f}")
    print(f"Scores sauvegardes : {scores_path}")
    print(f"Seuil sauvegarde : {threshold_path}")
    print(f"Figure distribution : {distribution_path}")
    print(f"Figure boxplot : {boxplot_path}")
    print(f"Reconstructions : {reconstruction_path}")

    for split_name in scores_df["split"].unique():
        split_scores = scores_df[scores_df["split"] == split_name]
        rejected = int(split_scores["is_above_threshold"].sum())
        total = int(len(split_scores))
        print(
            f"{split_name}: {rejected}/{total} rejetes "
            f"({rejected / max(total, 1):.2%})"
        )


if __name__ == "__main__":
    main()
