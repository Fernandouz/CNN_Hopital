from pathlib import Path
import argparse
import copy
import json
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from core.data_processing import WoundDataset, build_class_mapping, load_splits
from core.vae import ConvVAE, kl_divergence, reconstruction_error, vae_loss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train convolutional variational autoencoder for OOD detection."
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-suffix", type=str, default="")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)

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


def build_vae_dataloaders(split_dir, img_size, batch_size, num_workers):
    train_df, val_df, test_df = load_splits(split_dir)
    class_names, class_to_idx, idx_to_class = build_class_mapping(train_df)
    transform = get_vae_transform(img_size)

    train_dataset = WoundDataset(
        dataframe=train_df,
        transform=transform,
        class_to_idx=class_to_idx,
        project_root=PROJECT_ROOT,
    )
    val_dataset = WoundDataset(
        dataframe=val_df,
        transform=transform,
        class_to_idx=class_to_idx,
        project_root=PROJECT_ROOT,
    )
    test_dataset = WoundDataset(
        dataframe=test_df,
        transform=transform,
        class_to_idx=class_to_idx,
        project_root=PROJECT_ROOT,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    metadata = {
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
    }

    return train_loader, val_loader, test_loader, metadata


def train_one_epoch(model, loader, optimizer, device, beta, max_batches=0):
    model.train()

    totals = {
        "loss": 0.0,
        "reconstruction": 0.0,
        "kl": 0.0,
        "num_images": 0,
    }

    for batch_idx, (images, _) in enumerate(tqdm(loader, desc="Train VAE", leave=False)):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        images = images.to(device)
        optimizer.zero_grad()

        reconstructions, mu, logvar = model(images)
        loss, recon, kl = vae_loss(
            images=images,
            reconstructions=reconstructions,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["reconstruction"] += recon.item() * batch_size
        totals["kl"] += kl.item() * batch_size
        totals["num_images"] += batch_size

    num_images = max(totals["num_images"], 1)
    return {
        "loss": totals["loss"] / num_images,
        "reconstruction": totals["reconstruction"] / num_images,
        "kl": totals["kl"] / num_images,
    }


@torch.no_grad()
def evaluate(model, loader, device, beta, max_batches=0):
    model.eval()

    totals = {
        "loss": 0.0,
        "reconstruction": 0.0,
        "kl": 0.0,
        "num_images": 0,
    }

    all_reconstruction_errors = []
    all_kl_values = []

    for batch_idx, (images, _) in enumerate(tqdm(loader, desc="Eval VAE", leave=False)):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        images = images.to(device)
        reconstructions, mu, logvar = model(images)
        loss, recon, kl = vae_loss(
            images=images,
            reconstructions=reconstructions,
            mu=mu,
            logvar=logvar,
            beta=beta,
        )
        per_image_recon = reconstruction_error(
            images,
            reconstructions,
            reduction="none",
        )
        per_image_kl = kl_divergence(mu, logvar, reduction="none")

        batch_size = images.size(0)
        totals["loss"] += loss.item() * batch_size
        totals["reconstruction"] += recon.item() * batch_size
        totals["kl"] += kl.item() * batch_size
        totals["num_images"] += batch_size

        all_reconstruction_errors.extend(per_image_recon.cpu().numpy().tolist())
        all_kl_values.extend(per_image_kl.cpu().numpy().tolist())

    num_images = max(totals["num_images"], 1)
    return {
        "loss": totals["loss"] / num_images,
        "reconstruction": totals["reconstruction"] / num_images,
        "kl": totals["kl"] / num_images,
        "reconstruction_errors": all_reconstruction_errors,
        "kl_values": all_kl_values,
    }


def save_reconstruction_examples(model, loader, device, output_path, max_images=8):
    model.eval()

    images, _ = next(iter(loader))
    images = images[:max_images].to(device)

    with torch.no_grad():
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


def save_training_curves(history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train total")
    plt.plot(epochs, [row["val_loss"] for row in history], label="Val total")
    plt.plot(
        epochs,
        [row["val_reconstruction"] for row in history],
        label="Val reconstruction",
    )
    plt.plot(epochs, [row["val_kl"] for row in history], label="Val KL")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE training curves")
    plt.legend()
    plt.tight_layout()

    path = output_dir / "vae_training_curves.png"
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def main():
    args = parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("wound-ood-vae")

    run_name = (
        f"conv_vae"
        f"_latent-{args.latent_dim}"
        f"_beta-{args.beta}"
        f"_lr-{args.lr}"
    )

    if args.run_suffix:
        run_name = f"{run_name}_{args.run_suffix}"

    split_dir = PROJECT_ROOT / "data" / "processed" / "splits"
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = PROJECT_ROOT / "reports" / "ood" / run_name
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print(f"Device utilise : {device}")
    print(f"Run name : {run_name}")
    print(f"Beta KL : {args.beta}")

    with mlflow.start_run(run_name=run_name):
        train_loader, val_loader, test_loader, metadata = build_vae_dataloaders(
            split_dir=split_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        model = ConvVAE(latent_dim=args.latent_dim).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        mlflow.log_params({
            "model_type": "conv_vae",
            "latent_dim": args.latent_dim,
            "beta": args.beta,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "loss": "MSE reconstruction + beta * KL",
            "dataset_name": "wound-classification-dataset",
            "dataset_version": "v1",
        })

        mlflow.set_tags({
            "project": "CNN_Hopital",
            "task": "ood-detection",
            "phase": "training",
            "run_type": "vae-training",
            "framework": "pytorch",
            "device": str(device),
        })

        best_val_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                beta=args.beta,
                max_batches=args.max_train_batches,
            )
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                beta=args.beta,
                max_batches=args.max_eval_batches,
            )

            scheduler.step(val_metrics["loss"])
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"train_loss={train_metrics['loss']:.6f} | "
                f"val_loss={val_metrics['loss']:.6f} | "
                f"val_recon={val_metrics['reconstruction']:.6f} | "
                f"val_kl={val_metrics['kl']:.6f} | "
                f"lr={current_lr:.2e}"
            )

            mlflow.log_metric("train_loss", train_metrics["loss"], step=epoch)
            mlflow.log_metric(
                "train_reconstruction_loss",
                train_metrics["reconstruction"],
                step=epoch,
            )
            mlflow.log_metric("train_kl", train_metrics["kl"], step=epoch)
            mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
            mlflow.log_metric(
                "val_reconstruction_loss",
                val_metrics["reconstruction"],
                step=epoch,
            )
            mlflow.log_metric("val_kl", val_metrics["kl"], step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

            history.append({
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_reconstruction": train_metrics["reconstruction"],
                "train_kl": train_metrics["kl"],
                "val_loss": val_metrics["loss"],
                "val_reconstruction": val_metrics["reconstruction"],
                "val_kl": val_metrics["kl"],
                "lr": current_lr,
            })

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                print("Nouveau meilleur VAE sauvegarde en memoire.")
            else:
                epochs_without_improvement += 1
                print(
                    f"Pas d'amelioration : "
                    f"{epochs_without_improvement}/{args.patience}"
                )

            if epochs_without_improvement >= args.patience:
                print("Early stopping declenche.")
                break

        checkpoint = {
            "model_type": "conv_vae",
            "model_state_dict": best_model_state,
            "latent_dim": args.latent_dim,
            "beta": args.beta,
            "img_size": args.img_size,
            "class_names": metadata["class_names"],
            "history": history,
            "best_val_loss": best_val_loss,
        }

        checkpoint_path = model_dir / f"{run_name}_best.pt"
        torch.save(checkpoint, checkpoint_path)

        history_path = reports_dir / "vae_history.json"
        with open(history_path, "w", encoding="utf-8") as file:
            json.dump(history, file, indent=4)

        model.load_state_dict(best_model_state)

        reconstruction_path = reports_dir / "vae_reconstruction_examples.png"
        save_reconstruction_examples(
            model=model,
            loader=val_loader,
            device=device,
            output_path=reconstruction_path,
        )

        curves_path = save_training_curves(
            history=history,
            output_dir=reports_dir,
        )

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("epochs_ran", len(history))
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(history_path), artifact_path="history")
        mlflow.log_artifact(str(reconstruction_path), artifact_path="reconstructions")
        mlflow.log_artifact(str(curves_path), artifact_path="curves")

        print(f"\nVAE sauvegarde : {checkpoint_path}")
        print(f"Rapports sauvegardes : {reports_dir}")
        print(f"Run MLflow enregistre : {run_name}")


if __name__ == "__main__":
    main()
