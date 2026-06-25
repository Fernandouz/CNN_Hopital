from core.model_utils import (
    build_model,
    freeze_backbone,
    unfreeze_last_blocks,
    count_total_parameters,
    count_trainable_parameters,
)
from core.data_processing import build_dataloaders
from pathlib import Path
import sys
import copy
import json
import argparse

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix

import mlflow
import mlflow.pytorch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def parse_args():
    # Paramètres modifiables depuis la ligne de commande.
    parser = argparse.ArgumentParser(
        description="Train CNN model for wound classification")

    parser.add_argument("--architecture", type=str, default="resnet50",
                        choices=["vgg16", "resnet50", "efficientnet_b0", "mobilenet_v3_large", "custom_cnn"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout-rate", type=float, default=0.3)

    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--fine-tune", action="store_true")

    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--class-weights", action="store_true")

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--run-suffix",
        type=str,
        default="",
        help="Suffixe optionnel ajouté au nom du run et aux fichiers sauvegardés."
    )

    return parser.parse_args()


def get_device():
    # Priorité Mac MPS, puis CUDA, sinon CPU.
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    # Phase entraînement : gradients activés.
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        # Mise à jour des poids.
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc, all_labels, all_preds


def main():
    args = parse_args()

    # Tous les runs CNN sont regroupés dans le même experiment MLflow.
    mlflow.set_experiment("CNN-wound-classif")

    # Nom explicite pour comparer facilement les runs dans MLflow.
    run_name = (
        f"{args.architecture}"
        f"_pretrained-{args.pretrained}"
        f"_freeze-{args.freeze_backbone}"
        f"_finetune-{args.fine_tune}"
        f"_weighted-{args.weighted_sampler}"
        f"_classweights-{args.class_weights}"
        f"_lr-{args.lr}"
    )

    if args.run_suffix:
        run_name = f"{run_name}_{args.run_suffix}"

    split_dir = PROJECT_ROOT / "data" / "processed" / "splits"
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print(f"Device utilisé : {device}")
    print(f"Run name : {run_name}")
    print(f"Architecture : {args.architecture}")
    print(f"Pretrained : {args.pretrained}")
    print(f"Freeze backbone : {args.freeze_backbone}")
    print(f"Fine-tuning partiel : {args.fine_tune}")
    print(f"Weighted sampler : {args.weighted_sampler}")
    print(f"Class weights : {args.class_weights}")

    with mlflow.start_run(run_name=run_name):
        # Chargement des splits, transformations et stratégie de rééquilibrage.
        train_loader, val_loader, test_loader, metadata = build_dataloaders(
            split_dir=split_dir,
            batch_size=args.batch_size,
            img_size=args.img_size,
            use_weighted_sampler=args.weighted_sampler,
            num_workers=args.num_workers,
            project_root=PROJECT_ROOT
        )

        num_classes = len(metadata["class_names"])

        mlflow.log_params({
            "architecture": args.architecture,
            "pretrained": args.pretrained,
            "freeze_backbone": args.freeze_backbone,
            "fine_tune": args.fine_tune,
            "weighted_sampler": args.weighted_sampler,
            "class_weights": args.class_weights,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_size": args.img_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "dropout_rate": args.dropout_rate,
            "patience": args.patience,
            "num_workers": args.num_workers,
            "num_classes": num_classes,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "loss": "CrossEntropyLoss",
            "augmentation": "online_torchvision_transforms",
            "rebalance_strategy": (
                "weighted_sampler" if args.weighted_sampler
                else "class_weights" if args.class_weights
                else "none"
            ),
            "dataset_name": "wound-classification-dataset",
            "dataset_version": "v1",
        })

        # Tags MLflow pour filtrer les expériences.
        mlflow.set_tags({
            "project": "CNN_Hopital",
            "task": "wound-classification",
            "framework": "pytorch",
            "device": str(device),
            "dataset": "wound-classification-dataset",
            "dataset_version": "v1",
            "comment": "CNN training with transfer learning, data augmentation and class rebalancing",
        })

        model = build_model(
            architecture=args.architecture,
            num_classes=num_classes,
            pretrained=args.pretrained,
            dropout_rate=args.dropout_rate
        )

        # Transfer learning : tête seule, ou fine-tuning partiel.
        if args.freeze_backbone:
            model = freeze_backbone(model, args.architecture)

        if args.fine_tune:
            model = freeze_backbone(model, args.architecture)
            model = unfreeze_last_blocks(model, args.architecture)

        model = model.to(device)

        total_params = count_total_parameters(model)
        trainable_params = count_trainable_parameters(model)

        print(f"Paramètres totaux : {total_params:,}")
        print(f"Paramètres entraînables : {trainable_params:,}")

        mlflow.log_metric("total_parameters", total_params)
        mlflow.log_metric("trainable_parameters", trainable_params)

        # Alternative au sampler pondéré : pondérer directement la loss.
        if args.class_weights:
            class_weights = torch.tensor(
                metadata["class_weights"].values,
                dtype=torch.float32
            ).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Réduit le learning rate si la validation stagne.
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_model_state = None
        epochs_without_improvement = 0

        history = []

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")

            # Une époque train puis une évaluation validation.
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device
            )

            val_loss, val_acc, val_labels, val_preds = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device
            )

            precision_macro = precision_score(
                val_labels,
                val_preds,
                average="macro",
                zero_division=0
            )

            recall_macro = recall_score(
                val_labels,
                val_preds,
                average="macro",
                zero_division=0
            )

            f1_macro = f1_score(
                val_labels,
                val_preds,
                average="macro",
                zero_division=0
            )

            scheduler.step(val_loss)

            # LR courant utile pour analyser le scheduler.
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | lr={current_lr:.2e}"
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            mlflow.log_metric("precision_macro", precision_macro, step=epoch)
            mlflow.log_metric("recall_macro", recall_macro, step=epoch)
            mlflow.log_metric("f1_macro", f1_macro, step=epoch)

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "f1_macro": f1_macro,
            })

            # Sauvegarde en mémoire du meilleur modèle selon la val_loss.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
                print("Nouveau meilleur modèle sauvegardé en mémoire.")
            else:
                epochs_without_improvement += 1
                print(
                    f"Pas d'amélioration : {epochs_without_improvement}/{args.patience}"
                )

            # Arrêt anticipé si la validation ne progresse plus.
            if epochs_without_improvement >= args.patience:
                print("Early stopping déclenché.")
                break

        # Checkpoint complet : poids, classes et configuration utile à l'inférence.
        checkpoint = {
            "architecture": args.architecture,
            "model_state_dict": best_model_state,
            "class_names": metadata["class_names"],
            "class_to_idx": metadata["class_to_idx"],
            "idx_to_class": metadata["idx_to_class"],
            "img_size": args.img_size,
            "pretrained": args.pretrained,
            "freeze_backbone": args.freeze_backbone,
            "fine_tune": args.fine_tune,
            "weighted_sampler": args.weighted_sampler,
            "class_weights": args.class_weights,
            "history": history,
            "best_val_loss": best_val_loss,
            "best_val_acc": best_val_acc,
        }

        checkpoint_path = model_dir / f"{run_name}_best.pt"
        torch.save(checkpoint, checkpoint_path)

        history_path = model_dir / f"{run_name}_history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.log_metric("epochs_ran", len(history))

        best_epoch = max(history, key=lambda x: x["val_acc"])

        mlflow.log_metric("final_precision_macro",
                          history[-1]["precision_macro"])
        mlflow.log_metric("final_recall_macro", history[-1]["recall_macro"])
        mlflow.log_metric("final_f1_macro", history[-1]["f1_macro"])

        mlflow.log_metric("best_f1_macro", max(h["f1_macro"] for h in history))
        mlflow.log_metric("best_recall_macro", max(
            h["recall_macro"] for h in history))
        mlflow.log_metric("best_precision_macro", max(
            h["precision_macro"] for h in history))

        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(history_path), artifact_path="history")

        # Désactivé pour éviter un artefact MLflow volumineux pendant les tests.
        # mlflow.pytorch.log_model(
        #     pytorch_model=model,
        #     artifact_path="model"
        # )

        print(f"\nMeilleur modèle sauvegardé : {checkpoint_path}")
        print(f"Historique sauvegardé : {history_path}")
        print(f"Run MLflow enregistré : {run_name}")


if __name__ == "__main__":
    main()
