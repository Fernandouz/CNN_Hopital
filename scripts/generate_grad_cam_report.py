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
import pandas as pd
import torch
from PIL import Image

from core.grad_cam import GradCAMExplainer, save_grad_cam_result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM figures for correctly and incorrectly classified wound images."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/resnet50_best.pt",
        help="Chemin du checkpoint CNN.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Split utilise pour choisir les exemples.",
    )
    parser.add_argument(
        "--num-correct",
        type=int,
        default=5,
        help="Nombre de classifications correctes a expliquer.",
    )
    parser.add_argument(
        "--num-errors",
        type=int,
        default=3,
        help="Nombre d'erreurs de classification a expliquer.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/xai",
        help="Dossier racine de sortie.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=[None, "cpu", "cuda", "mps"],
        help="Device PyTorch force. Par defaut, utilise la logique du projet.",
    )
    return parser.parse_args()


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_image_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    if str(path).startswith("../data/"):
        return PROJECT_ROOT / str(path).replace("../data/", "data/")
    return PROJECT_ROOT / path


def get_device(device_name):
    if device_name is None:
        return None
    return torch.device(device_name)


@torch.no_grad()
def collect_split_predictions(explainer, split_df):
    rows = []

    for index, row in split_df.reset_index(drop=True).iterrows():
        image_path = resolve_image_path(row["path"])

        image = Image.open(image_path).convert("RGB")
        input_tensor = explainer.transform(image).unsqueeze(0).to(explainer.device)

        logits = explainer.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)
        confidence, pred_idx = torch.max(probabilities, dim=0)

        pred_idx = int(pred_idx.item())
        true_label = row["class"]
        predicted_label = explainer.class_names[pred_idx]

        rows.append({
            "split_index": int(index),
            "image_path": str(image_path),
            "true_label": true_label,
            "predicted_label": predicted_label,
            "predicted_class_idx": pred_idx,
            "confidence": float(confidence.item()),
            "correct": true_label == predicted_label,
        })

    return pd.DataFrame(rows)


def select_examples(predictions_df, num_correct, num_errors):
    correct_examples = (
        predictions_df[predictions_df["correct"]]
        .sort_values("confidence", ascending=False)
        .head(num_correct)
    )
    error_examples = (
        predictions_df[~predictions_df["correct"]]
        .sort_values("confidence", ascending=False)
        .head(num_errors)
    )

    selected = pd.concat(
        [
            correct_examples.assign(example_type="correct"),
            error_examples.assign(example_type="error"),
        ],
        ignore_index=True,
    )

    return selected


def save_summary_grid(summary_df, output_path):
    if summary_df.empty:
        return None

    n_rows = len(summary_df)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(8, max(3.2 * n_rows, 4)),
        squeeze=False,
    )

    for row_idx, row in summary_df.reset_index(drop=True).iterrows():
        original = Image.open(row["original_path"]).convert("RGB")
        overlay = Image.open(row["overlay_path"]).convert("RGB")

        axes[row_idx, 0].imshow(original)
        axes[row_idx, 0].axis("off")
        axes[row_idx, 0].set_title(
            f"Originale\nVrai: {row['true_label']}",
            fontsize=9,
        )

        axes[row_idx, 1].imshow(overlay)
        axes[row_idx, 1].axis("off")
        axes[row_idx, 1].set_title(
            "Grad-CAM\n"
            f"Pred: {row['predicted_label']} ({row['confidence']:.2f})",
            fontsize=9,
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def clean_generated_examples(examples_dir):
    """Retire les PNG generes par ce script lors d'une execution precedente."""
    generated_suffixes = (
        "_original.png",
        "_heatmap.png",
        "_gradcam_overlay.png",
    )

    for path in examples_dir.glob("*.png"):
        if path.name.endswith(generated_suffixes):
            path.unlink()


def main():
    args = parse_args()

    checkpoint_path = resolve_project_path(args.checkpoint)
    split_path = PROJECT_ROOT / "data" / "processed" / "splits" / f"{args.split}.csv"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Split introuvable : {split_path}")

    run_name = checkpoint_path.stem.replace("_best", "")
    output_dir = resolve_project_path(args.output_dir) / run_name / args.split
    examples_dir = output_dir / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)
    clean_generated_examples(examples_dir)

    explainer = GradCAMExplainer(
        checkpoint_path=checkpoint_path,
        device=get_device(args.device),
        project_root=PROJECT_ROOT,
    )

    split_df = pd.read_csv(split_path)
    predictions_df = collect_split_predictions(
        explainer=explainer,
        split_df=split_df,
    )
    predictions_path = output_dir / "grad_cam_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    selected_df = select_examples(
        predictions_df=predictions_df,
        num_correct=args.num_correct,
        num_errors=args.num_errors,
    )

    summary_rows = []
    for rank, row in selected_df.reset_index(drop=True).iterrows():
        result = explainer.explain_image(
            image_path=row["image_path"],
            target_class=row["predicted_class_idx"],
        )

        stem = (
            f"{rank + 1:02d}_{row['example_type']}_"
            f"{Path(row['image_path']).stem}"
        )
        saved_paths = save_grad_cam_result(
            result=result,
            output_dir=examples_dir,
            stem=stem,
        )

        summary_rows.append({
            **row.to_dict(),
            "architecture": result.architecture,
            "target_class": result.target_class,
            "top_k": json.dumps(result.top_k, ensure_ascii=False),
            **saved_paths,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "grad_cam_examples.csv"
    summary_json_path = output_dir / "grad_cam_examples.json"
    summary_df.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(
        json.dumps(summary_rows, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )

    grid_path = output_dir / "grad_cam_summary_grid.png"
    save_summary_grid(summary_df=summary_df, output_path=grid_path)

    print("\nRapport Grad-CAM genere.")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Split : {args.split}")
    print(f"Predictions : {predictions_path}")
    print(f"Exemples : {summary_csv_path}")
    print(f"Grille : {grid_path}")
    print(f"Dossier images : {examples_dir}")


if __name__ == "__main__":
    main()
