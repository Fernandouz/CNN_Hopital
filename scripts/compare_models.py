from pathlib import Path
import json
import argparse

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "reports" / "evaluation"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare evaluated CNN models."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Split évalué à agréger : val ou test."
    )

    parser.add_argument(
        "--contains",
        type=str,
        default=None,
        help="Filtre optionnel : conserve seulement les runs dont le nom contient cette chaîne."
    )

    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Filtre optionnel : exclut les runs dont le nom contient cette chaîne."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    summary_paths = sorted(EVAL_DIR.glob(
        f"*/{args.split}/evaluation_summary.json"))

    if args.contains:
        summary_paths = [
            path for path in summary_paths
            if args.contains in path.parts[-3]
        ]

    if args.exclude:
        summary_paths = [
            path for path in summary_paths
            if args.exclude not in path.parts[-3]
        ]

    if not summary_paths:
        print(
            f"Aucun fichier evaluation_summary.json trouvé pour le split : {args.split}")
        print(f"Filtre contains : {args.contains}")
        print(f"Filtre exclude : {args.exclude}")
        return

    rows = []

    for summary_path in summary_paths:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        rows.append(summary)

    comparison_df = pd.DataFrame(rows)

    columns_order = [
        "run_name",
        "architecture",
        "split",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "best_val_acc",
        "best_val_loss",
        "num_samples",
        "num_errors",
    ]

    existing_columns = [
        col for col in columns_order
        if col in comparison_df.columns
    ]

    comparison_df = comparison_df[existing_columns]

    comparison_df = comparison_df.sort_values(
        by=["macro_f1", "accuracy"],
        ascending=False
    )

    suffix = f"_{args.contains}" if args.contains else ""
    output_path = EVAL_DIR / f"model_comparison_{args.split}{suffix}.csv"

    comparison_df.to_csv(output_path, index=False)

    print("\nComparaison des modèles :")
    print(comparison_df.to_string(index=False))

    print(f"\nTable sauvegardée dans : {output_path}")


if __name__ == "__main__":
    main()
