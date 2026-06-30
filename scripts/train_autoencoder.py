import mlflow
from pathlib import Path
import sys
import argparse
import copy
import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from core.autoencoder import ConvAutoencoder, reconstruction_error
from core.data_processing import load_splits, build_class_mapping, WoundDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train convolutional autoencoder for OOD detection."
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-suffix", type=str, default="")

    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_autoencoder_transforms(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])


def build_autoencoder_dataloaders(split_dir, img_size, batch_size, num_workers):
    train_df, val_df, test_df = load_splits(split_dir)
    class_names, class_to_idx, idx_to_class = build_class_mapping(train_df)

    ae_transform = get_autoencoder_transforms(img_size)

    train_dataset = WoundDataset(
        dataframe=train_df,
        transform=ae_transform,
        class_to_idx=class_to_idx,
        project_root=PROJECT_ROOT,
    )

    val_dataset = WoundDataset(
        dataframe=val_df,
        transform=ae_transform,
        class_to_idx=class_to_idx,
        project_root=PROJECT_ROOT,
    )

    test_dataset = WoundDataset(
        dataframe=test_df,
        transform=ae_transform,
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


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    running_loss = 0.0
    total = 0

    for images, _ in tqdm(loader, desc="Train AE", leave=False):
        images = images.to(device)

        optimizer.zero_grad()

        reconstructions = model(images)
        loss = reconstruction_error(images, reconstructions, reduction="mean")

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    return running_loss / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    running_loss = 0.0
    total = 0

    all_errors = []

    for images, _ in tqdm(loader, desc="Eval AE", leave=False):
        images = images.to(device)

        reconstructions = model(images)
        errors = reconstruction_error(
            images, reconstructions, reduction="none")

        running_loss += errors.sum().item()
        total += images.size(0)

        all_errors.extend(errors.cpu().numpy().tolist())

    return running_loss / total, all_errors


def save_reconstruction_examples(model, loader, device, output_path, max_images=8):
    model.eval()

    images, _ = next(iter(loader))
    images = images[:max_images].to(device)

    with torch.no_grad():
        reconstructions = model(images)

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
            ax.set_ylabel("Reconstruit", fontsize=12)

    plt.suptitle("Exemples de reconstruction — Autoencoder")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_training_curves(history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [h["train_loss"]
             for h in history], label="Train reconstruction loss")
    plt.plot(epochs, [h["val_loss"] for h in history],
             label="Validation reconstruction loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Courbes d'entraînement de l'autoencoder")
    plt.legend()
    plt.tight_layout()

    path = output_dir / "autoencoder_training_curves.png"
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def main():
    args = parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("wound-ood-autoencoder")

    run_name = (
        f"conv_autoencoder"
        f"_latent-{args.latent_dim}"
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

    print(f"Device utilisé : {device}")
    print(f"Run name : {run_name}")

    with mlflow.start_run(run_name=run_name):
        train_loader, val_loader, test_loader, metadata = build_autoencoder_dataloaders(
            split_dir=split_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        model = ConvAutoencoder(latent_dim=args.latent_dim).to(device)

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
            "model_type": "conv_autoencoder",
            "latent_dim": args.latent_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "loss": "MSE reconstruction error",
            "dataset_name": "wound-classification-dataset",
            "dataset_version": "v1",
        })

        mlflow.set_tags({
            "project": "CNN_Hopital",
            "task": "ood-detection",
            "phase": "training",
            "run_type": "autoencoder-training",
            "framework": "pytorch",
            "device": str(device),
        })

        best_val_loss = float("inf")
        best_model_state = None
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")

            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
            )

            val_loss, val_errors = evaluate(
                model=model,
                loader=val_loader,
                device=device,
            )

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | lr={current_lr:.2e}"
            )

            mlflow.log_metric("train_reconstruction_loss",
                              train_loss, step=epoch)
            mlflow.log_metric("val_reconstruction_loss", val_loss, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                print("Nouveau meilleur autoencoder sauvegardé en mémoire.")
            else:
                epochs_without_improvement += 1
                print(
                    f"Pas d'amélioration : {epochs_without_improvement}/{args.patience}")

            if epochs_without_improvement >= args.patience:
                print("Early stopping déclenché.")
                break

        checkpoint = {
            "model_type": "conv_autoencoder",
            "model_state_dict": best_model_state,
            "latent_dim": args.latent_dim,
            "img_size": args.img_size,
            "class_names": metadata["class_names"],
            "history": history,
            "best_val_loss": best_val_loss,
        }

        checkpoint_path = model_dir / f"{run_name}_best.pt"
        torch.save(checkpoint, checkpoint_path)

        history_path = reports_dir / "autoencoder_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

        model.load_state_dict(best_model_state)

        reconstruction_path = reports_dir / "reconstruction_examples.png"
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

        mlflow.log_metric("best_val_reconstruction_loss", best_val_loss)
        mlflow.log_metric("epochs_ran", len(history))

        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(history_path), artifact_path="history")
        mlflow.log_artifact(str(reconstruction_path),
                            artifact_path="reconstructions")
        mlflow.log_artifact(str(curves_path), artifact_path="curves")

        print(f"\nAutoencoder sauvegardé : {checkpoint_path}")
        print(f"Rapports sauvegardés : {reports_dir}")
        print(f"Run MLflow enregistré : {run_name}")


if __name__ == "__main__":
    main()
