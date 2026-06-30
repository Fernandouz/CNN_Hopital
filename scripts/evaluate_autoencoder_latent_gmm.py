from pathlib import Path
import argparse
import json
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from core.autoencoder import ConvAutoencoder
from core.data_processing import WoundDataset, build_class_mapping, load_splits


class OODImageDataset(Dataset):
    """Dataset simple pour extraire les latents des images hors-domaine."""

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
        description="Evaluate OOD detection with autoencoder latent embeddings and GMM."
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
        help="Percentile validation utilise pour definir le seuil OOD."
    )
    parser.add_argument(
        "--gmm-components",
        type=int,
        default=4,
        help="Nombre de composantes du Gaussian Mixture Model."
    )
    parser.add_argument(
        "--covariance-type",
        type=str,
        default="diag",
        choices=["full", "tied", "diag", "spherical"],
        help="Type de covariance du GMM. 'diag' est robuste avec peu d'images."
    )
    parser.add_argument("--reg-covar", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--ood-dir",
        type=str,
        default="data/ood",
        help="Dossier contenant les images hors-domaine a tester."
    )
    parser.add_argument(
        "--image-extensions",
        type=str,
        default=".jpg,.jpeg,.png,.bmp,.webp",
        help="Extensions image OOD separees par des virgules."
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

    model = ConvAutoencoder(latent_dim=checkpoint["latent_dim"])
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
        return None

    dataset = OODImageDataset(
        image_paths=image_paths,
        transform=transform
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


@torch.no_grad()
def collect_latents(model, loader, device, split_name, has_paths=False):
    rows = []
    latents = []
    sample_index = 0

    for batch in tqdm(loader, desc=f"Extract latents [{split_name}]"):
        if has_paths:
            images, paths = batch
        else:
            images, _ = batch
            paths = [None] * images.size(0)

        images = images.to(device)
        z = model.encode(images).detach().cpu().numpy()
        latents.append(z)

        for image_path in paths:
            row = {
                "split": split_name,
                "sample_index": sample_index,
            }
            if image_path is not None:
                row["image_path"] = image_path
            rows.append(row)
            sample_index += 1

    latent_matrix = np.concatenate(latents, axis=0)
    metadata_df = pd.DataFrame(rows)

    return latent_matrix, metadata_df


def build_scores_df(metadata_df, log_likelihood):
    scores_df = metadata_df.copy()
    scores_df["latent_log_likelihood"] = log_likelihood
    scores_df["latent_anomaly_score"] = -log_likelihood
    return scores_df


def save_score_distribution(scores_df, threshold, output_dir):
    plt.figure(figsize=(9, 5))

    for split_name in scores_df["split"].unique():
        split_scores = scores_df[
            scores_df["split"] == split_name
        ]["latent_anomaly_score"]

        plt.hist(
            split_scores,
            bins=30,
            alpha=0.5,
            label=split_name
        )

    plt.axvline(
        threshold,
        linestyle="--",
        color="black",
        label=f"Seuil OOD = {threshold:.4f}"
    )

    plt.xlabel("Score anomalie latent (- log-vraisemblance GMM)")
    plt.ylabel("Nombre d'images")
    plt.title("Distribution des scores OOD dans l'espace latent")
    plt.legend()
    plt.tight_layout()

    path = output_dir / "latent_gmm_score_distribution.png"
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def save_score_boxplot(scores_df, threshold, output_dir):
    splits = list(scores_df["split"].unique())
    values = [
        scores_df[scores_df["split"] == split]["latent_anomaly_score"].values
        for split in splits
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(values, tick_labels=splits)
    plt.axhline(
        threshold,
        linestyle="--",
        color="black",
        label=f"Seuil OOD = {threshold:.4f}"
    )

    plt.ylabel("Score anomalie latent (- log-vraisemblance GMM)")
    plt.title("Scores latents GMM par split")
    plt.legend()
    plt.tight_layout()

    path = output_dir / "latent_gmm_score_boxplot.png"
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def build_threshold_comparison(scores_df, percentiles):
    val_scores = scores_df[
        scores_df["split"] == "val"
    ]["latent_anomaly_score"].values

    rows = []
    for percentile in percentiles:
        threshold = float(np.percentile(val_scores, percentile))
        row = {
            "threshold_percentile": percentile,
            "threshold": threshold,
        }

        for split_name in ["train", "val", "test", "ood"]:
            split_scores = scores_df[scores_df["split"] == split_name]
            total = len(split_scores)
            rejected = int(split_scores["latent_anomaly_score"].gt(threshold).sum())
            row[f"{split_name}_total"] = total
            row[f"{split_name}_rejected"] = rejected
            row[f"{split_name}_rejection_rate"] = rejected / total if total else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    device = get_device()
    checkpoint_path = PROJECT_ROOT / args.checkpoint

    print(f"Device utilise : {device}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"GMM components : {args.gmm_components}")
    print(f"Covariance type : {args.covariance_type}")

    model, checkpoint = load_autoencoder_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device
    )

    run_name = checkpoint_path.stem.replace("_best", "")
    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "ood"
        / run_name
        / "evaluation_latent_gmm"
    )
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
    ood_loader = build_ood_loader(
        ood_dir=args.ood_dir,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_extensions=parse_image_extensions(args.image_extensions),
    )

    train_latents, train_meta = collect_latents(
        model=model,
        loader=train_loader,
        device=device,
        split_name="train",
    )
    val_latents, val_meta = collect_latents(
        model=model,
        loader=val_loader,
        device=device,
        split_name="val",
    )
    test_latents, test_meta = collect_latents(
        model=model,
        loader=test_loader,
        device=device,
        split_name="test",
    )

    latent_frames = []

    scaler = StandardScaler()
    train_latents_scaled = scaler.fit_transform(train_latents)

    gmm = GaussianMixture(
        n_components=args.gmm_components,
        covariance_type=args.covariance_type,
        reg_covar=args.reg_covar,
        random_state=args.random_state,
    )
    gmm.fit(train_latents_scaled)

    train_scores = build_scores_df(
        metadata_df=train_meta,
        log_likelihood=gmm.score_samples(train_latents_scaled),
    )
    latent_frames.append(train_scores)

    val_scores = build_scores_df(
        metadata_df=val_meta,
        log_likelihood=gmm.score_samples(scaler.transform(val_latents)),
    )
    latent_frames.append(val_scores)

    test_scores = build_scores_df(
        metadata_df=test_meta,
        log_likelihood=gmm.score_samples(scaler.transform(test_latents)),
    )
    latent_frames.append(test_scores)

    if ood_loader is not None:
        ood_latents, ood_meta = collect_latents(
            model=model,
            loader=ood_loader,
            device=device,
            split_name="ood",
            has_paths=True,
        )
        ood_scores = build_scores_df(
            metadata_df=ood_meta,
            log_likelihood=gmm.score_samples(scaler.transform(ood_latents)),
        )
        latent_frames.append(ood_scores)
    else:
        ood_scores = pd.DataFrame()

    scores_df = pd.concat(latent_frames, ignore_index=True)

    threshold = float(
        np.percentile(
            val_scores["latent_anomaly_score"],
            args.threshold_percentile
        )
    )
    scores_df["is_above_threshold"] = (
        scores_df["latent_anomaly_score"] > threshold
    )

    scores_path = output_dir / "latent_gmm_scores.csv"
    scores_df.to_csv(scores_path, index=False)

    comparison_df = build_threshold_comparison(
        scores_df=scores_df,
        percentiles=[99, 97.5, 95, 92.5, 90, 87.5, 85, 80, 75],
    )
    comparison_path = output_dir / "latent_gmm_threshold_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    threshold_data = {
        "method": "autoencoder_latent_gmm_validation_percentile",
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "model_checkpoint": str(checkpoint_path),
        "latent_dim": checkpoint["latent_dim"],
        "img_size": checkpoint.get("img_size", 224),
        "gmm_components": args.gmm_components,
        "covariance_type": args.covariance_type,
        "reg_covar": args.reg_covar,
        "random_state": args.random_state,
        "gmm_converged": bool(gmm.converged_),
        "gmm_n_iter": int(gmm.n_iter_),
        "gmm_lower_bound": float(gmm.lower_bound_),
        "train_score_mean": float(train_scores["latent_anomaly_score"].mean()),
        "train_score_std": float(train_scores["latent_anomaly_score"].std()),
        "val_score_mean": float(val_scores["latent_anomaly_score"].mean()),
        "val_score_std": float(val_scores["latent_anomaly_score"].std()),
        "test_score_mean": float(test_scores["latent_anomaly_score"].mean()),
        "test_score_std": float(test_scores["latent_anomaly_score"].std()),
        "val_rejected_count": int(val_scores["latent_anomaly_score"].gt(threshold).sum()),
        "val_total": int(len(val_scores)),
        "test_rejected_count": int(test_scores["latent_anomaly_score"].gt(threshold).sum()),
        "test_total": int(len(test_scores)),
        "ood_dir": str(PROJECT_ROOT / args.ood_dir),
        "image_extensions": list(parse_image_extensions(args.image_extensions)),
        "ood_total": int(len(ood_scores)),
        "ood_rejected_count": int(ood_scores["latent_anomaly_score"].gt(threshold).sum()) if not ood_scores.empty else 0,
        "ood_score_mean": float(ood_scores["latent_anomaly_score"].mean()) if not ood_scores.empty else None,
        "ood_score_std": float(ood_scores["latent_anomaly_score"].std()) if not ood_scores.empty else None,
    }

    threshold_path = output_dir / "latent_gmm_threshold.json"
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=4)

    dist_path = save_score_distribution(
        scores_df=scores_df,
        threshold=threshold,
        output_dir=output_dir,
    )
    boxplot_path = save_score_boxplot(
        scores_df=scores_df,
        threshold=threshold,
        output_dir=output_dir,
    )

    print("\nEvaluation latent GMM terminee.")
    print(f"Seuil OOD percentile {args.threshold_percentile}: {threshold:.6f}")
    print(
        "OOD rejetes: "
        f"{threshold_data['ood_rejected_count']} / {threshold_data['ood_total']}"
    )
    print(f"Scores sauvegardes : {scores_path}")
    print(f"Seuil sauvegarde : {threshold_path}")
    print(f"Comparaison seuils : {comparison_path}")
    print(f"Figure distribution : {dist_path}")
    print(f"Figure boxplot : {boxplot_path}")


if __name__ == "__main__":
    main()
