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
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from core.model_utils import build_model
from core.data_processing import build_dataloaders, IMAGENET_MEAN, IMAGENET_STD


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNN model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint .pt file"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on"
    )

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)

    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    return checkpoint


def build_model_from_checkpoint(checkpoint, device):
    model = build_model(
        architecture=checkpoint["architecture"],
        num_classes=len(checkpoint["class_names"]),
        pretrained=False,
        dropout_rate=0.3
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def denormalize_tensor(image_tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    image_tensor = image_tensor.cpu() * std + mean
    image_tensor = image_tensor.clamp(0, 1)

    return image_tensor


@torch.no_grad()
def collect_predictions(model, loader, device, idx_to_class):
    y_true = []
    y_pred = []
    y_conf = []
    y_top3 = []

    all_images = []

    for images, labels in tqdm(loader, desc="Evaluation"):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)

        confidences, preds = torch.max(probabilities, dim=1)
        top3_probs, top3_indices = torch.topk(probabilities, k=3, dim=1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_conf.extend(confidences.cpu().numpy().tolist())

        for indices, probs in zip(top3_indices.cpu().numpy(), top3_probs.cpu().numpy()):
            y_top3.append([
                {
                    "class": idx_to_class[int(idx)],
                    "probability": float(prob)
                }
                for idx, prob in zip(indices, probs)
            ])

        all_images.extend(images.cpu())

    return y_true, y_pred, y_conf, y_top3, all_images


def save_classification_report(y_true, y_pred, class_names, output_dir):
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_json_path = output_dir / "classification_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)

    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = output_dir / "classification_report.csv"
    report_df.to_csv(report_csv_path)

    return report_dict, report_df


def save_confusion_matrix(y_true, y_pred, class_names, output_dir):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(9, 7))
    plt.imshow(cm)
    plt.title("Matrice de confusion")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.xticks(np.arange(len(class_names)),
               class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)), class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()

    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=300)
    plt.close()

    return cm


def save_training_curves(history, output_dir):
    if not history:
        return None

    history_df = pd.DataFrame(history)

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    plt.plot(history_df["epoch"], history_df["val_loss"],
             label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Courbes de loss")
    plt.legend()
    plt.tight_layout()
    loss_path = output_dir / "training_loss_curve.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_acc"],
             label="Train accuracy")
    plt.plot(history_df["epoch"], history_df["val_acc"],
             label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Courbes d'accuracy")
    plt.legend()
    plt.tight_layout()
    acc_path = output_dir / "training_accuracy_curve.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()

    history_df.to_csv(output_dir / "training_history.csv", index=False)

    return history_df


def save_predictions_csv(y_true, y_pred, y_conf, y_top3, class_names, output_dir):
    rows = []

    for true_idx, pred_idx, conf, top3 in zip(y_true, y_pred, y_conf, y_top3):
        rows.append({
            "true_label": class_names[true_idx],
            "predicted_label": class_names[pred_idx],
            "confidence": conf,
            "correct": true_idx == pred_idx,
            "top3": json.dumps(top3, ensure_ascii=False)
        })

    predictions_df = pd.DataFrame(rows)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    return predictions_df


def save_misclassified_examples(
    images,
    y_true,
    y_pred,
    y_conf,
    class_names,
    output_dir,
    max_examples=12
):
    misclassified_indices = [
        idx for idx, (true_label, pred_label) in enumerate(zip(y_true, y_pred))
        if true_label != pred_label
    ]

    if len(misclassified_indices) == 0:
        return None

    selected_indices = misclassified_indices[:max_examples]

    n_cols = 4
    n_rows = int(np.ceil(len(selected_indices) / n_cols))

    plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    for plot_idx, sample_idx in enumerate(selected_indices):
        image = denormalize_tensor(images[sample_idx]).permute(1, 2, 0).numpy()

        true_name = class_names[y_true[sample_idx]]
        pred_name = class_names[y_pred[sample_idx]]
        conf = y_conf[sample_idx]

        ax = plt.subplot(n_rows, n_cols, plot_idx + 1)
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(
            f"Vrai: {true_name}\nPrédit: {pred_name}\nConf: {conf:.2f}",
            fontsize=9
        )

    plt.tight_layout()
    path = output_dir / "misclassified_examples.png"
    plt.savefig(path, dpi=300)
    plt.close()

    return path


def main():
    args = parse_args()

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    device = get_device()

    print(f"Device utilisé : {device}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Split évalué : {args.split}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    run_name = checkpoint_path.stem.replace("_best", "")

    output_dir = PROJECT_ROOT / "reports" / "evaluation" / run_name / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_from_checkpoint(checkpoint, device)

    split_dir = PROJECT_ROOT / "data" / "processed" / "splits"

    _, val_loader, test_loader, metadata = build_dataloaders(
        split_dir=split_dir,
        batch_size=args.batch_size,
        img_size=checkpoint["img_size"],
        use_weighted_sampler=False,
        num_workers=args.num_workers,
        project_root=PROJECT_ROOT
    )

    loader = val_loader if args.split == "val" else test_loader

    class_names = checkpoint["class_names"]
    idx_to_class = checkpoint["idx_to_class"]

    # Les clés JSON peuvent devenir des strings selon les sauvegardes.
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    y_true, y_pred, y_conf, y_top3, images = collect_predictions(
        model=model,
        loader=loader,
        device=device,
        idx_to_class=idx_to_class
    )

    accuracy = accuracy_score(y_true, y_pred)

    report_dict, report_df = save_classification_report(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_dir=output_dir
    )

    cm = save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        output_dir=output_dir
    )

    history_df = save_training_curves(
        history=checkpoint.get("history", []),
        output_dir=output_dir
    )

    predictions_df = save_predictions_csv(
        y_true=y_true,
        y_pred=y_pred,
        y_conf=y_conf,
        y_top3=y_top3,
        class_names=class_names,
        output_dir=output_dir
    )

    save_misclassified_examples(
        images=images,
        y_true=y_true,
        y_pred=y_pred,
        y_conf=y_conf,
        class_names=class_names,
        output_dir=output_dir
    )

    summary = {
        "run_name": run_name,
        "architecture": checkpoint["architecture"],
        "split": args.split,
        "accuracy": accuracy,
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_precision": report_dict["weighted avg"]["precision"],
        "weighted_recall": report_dict["weighted avg"]["recall"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "best_val_loss": checkpoint.get("best_val_loss"),
        "best_val_acc": checkpoint.get("best_val_acc"),
        "num_samples": len(y_true),
        "num_errors": int((np.array(y_true) != np.array(y_pred)).sum()),
    }

    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nÉvaluation terminée.")
    print(f"Accuracy {args.split}: {accuracy:.4f}")
    print(f"Macro F1 {args.split}: {summary['macro_f1']:.4f}")
    print(f"Résultats sauvegardés dans : {output_dir}")


if __name__ == "__main__":
    main()
