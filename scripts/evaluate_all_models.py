from pathlib import Path
import subprocess
import sys
import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved CNN checkpoints."
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Split à évaluer : val ou test."
    )

    parser.add_argument(
        "--contains",
        type=str,
        default=None,
        help="Filtre optionnel : évalue uniquement les checkpoints dont le nom contient cette chaîne."
    )

    parser.add_argument(
        "--exclude",
        type=str,
        default=None,
        help="Filtre optionnel : exclut les checkpoints dont le nom contient cette chaîne."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size utilisé pendant l'évaluation."
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Nombre de workers DataLoader."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoints = sorted(MODELS_DIR.glob("*_best.pt"))

    if args.contains:
        checkpoints = [
            ckpt for ckpt in checkpoints
            if args.contains in ckpt.name
        ]

    if args.exclude:
        checkpoints = [
            ckpt for ckpt in checkpoints
            if args.exclude not in ckpt.name
        ]

    if not checkpoints:
        print(f"Aucun checkpoint trouvé dans : {MODELS_DIR}")
        print(f"Filtre contains : {args.contains}")
        print(f"Filtre exclude : {args.exclude}")
        return

    print(f"{len(checkpoints)} checkpoint(s) trouvé(s).")
    print(f"Split évalué : {args.split}")
    print(f"Filtre contains : {args.contains}")
    print(f"Filtre exclude : {args.exclude}")
    print()

    failed = []

    for checkpoint in checkpoints:
        relative_checkpoint = checkpoint.relative_to(PROJECT_ROOT)

        print("=" * 100)
        print(f"Évaluation du modèle : {relative_checkpoint}")
        print("=" * 100)

        command = [
            sys.executable,
            "-m",
            "scripts.evaluate_cnn",
            "--checkpoint",
            str(relative_checkpoint),
            "--split",
            args.split,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
        ]

        result = subprocess.run(command, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            print(f"Erreur pendant l'évaluation de : {relative_checkpoint}")
            failed.append(str(relative_checkpoint))
        else:
            print(f"Évaluation terminée pour : {relative_checkpoint}")

        print()

    print("=" * 100)
    print("Résumé des évaluations")
    print("=" * 100)

    if failed:
        print("Évaluations en erreur :")
        for item in failed:
            print(f"- {item}")
    else:
        print("Toutes les évaluations sont terminées correctement.")


if __name__ == "__main__":
    main()
