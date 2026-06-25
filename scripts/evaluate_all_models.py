from pathlib import Path
import subprocess
import sys
import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all saved CNN checkpoints."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Split à évaluer : val ou test."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoints = sorted(MODELS_DIR.glob("*_best.pt"))

    if not checkpoints:
        print(f"Aucun checkpoint trouvé dans : {MODELS_DIR}")
        return

    print(f"{len(checkpoints)} checkpoint(s) trouvé(s).")
    print(f"Split évalué : {args.split}")
    print()

    for checkpoint in checkpoints:
        relative_checkpoint = checkpoint.relative_to(PROJECT_ROOT)

        print("=" * 80)
        print(f"Évaluation du modèle : {relative_checkpoint}")
        print("=" * 80)

        command = [
            sys.executable,
            "-m",
            "scripts.evaluate_cnn",
            "--checkpoint",
            str(relative_checkpoint),
            "--split",
            args.split,
        ]

        result = subprocess.run(command)

        if result.returncode != 0:
            print(f"Erreur pendant l'évaluation de : {relative_checkpoint}")
        else:
            print(f"Évaluation terminée pour : {relative_checkpoint}")

        print()

    print("Toutes les évaluations sont terminées.")


if __name__ == "__main__":
    main()
