from pathlib import Path
import subprocess
import sys
import argparse
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch final training runs for selected CNN models."
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--run-suffix", type=str, default=None)

    parser.add_argument(
        "--include-custom",
        action="store_true",
        help="Inclure le custom_cnn from scratch."
    )

    return parser.parse_args()


def run_command(command):
    print("=" * 100)
    print("Commande lancée :")
    print(" ".join(command))
    print("=" * 100)

    result = subprocess.run(command, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print("Erreur pendant l'entraînement.")
        return False

    print("Entraînement terminé avec succès.")
    return True


def main():
    args = parse_args()

    if args.run_suffix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        run_suffix = f"final_{args.epochs}epochs_{timestamp}"
    else:
        run_suffix = args.run_suffix

    experiments = [
        {
            "name": "vgg16_frozen",
            "architecture": "vgg16",
            "pretrained": True,
            "freeze_backbone": True,
            "fine_tune": False,
            "weighted_sampler": True,
            "class_weights": False,
            "lr": "1e-4",
        },
        {
            "name": "resnet50_frozen",
            "architecture": "resnet50",
            "pretrained": True,
            "freeze_backbone": True,
            "fine_tune": False,
            "weighted_sampler": True,
            "class_weights": False,
            "lr": "1e-4",
        },
        {
            "name": "efficientnet_b0_frozen",
            "architecture": "efficientnet_b0",
            "pretrained": True,
            "freeze_backbone": True,
            "fine_tune": False,
            "weighted_sampler": True,
            "class_weights": False,
            "lr": "1e-4",
        },
        {
            "name": "mobilenet_v3_large_frozen",
            "architecture": "mobilenet_v3_large",
            "pretrained": True,
            "freeze_backbone": True,
            "fine_tune": False,
            "weighted_sampler": True,
            "class_weights": False,
            "lr": "1e-4",
        },
        {
            "name": "resnet50_finetune",
            "architecture": "resnet50",
            "pretrained": True,
            "freeze_backbone": False,
            "fine_tune": True,
            "weighted_sampler": True,
            "class_weights": False,
            "lr": "1e-5",
        },
    ]

    if args.include_custom:
        experiments.append(
            {
                "name": "custom_cnn_from_scratch",
                "architecture": "custom_cnn",
                "pretrained": False,
                "freeze_backbone": False,
                "fine_tune": False,
                "weighted_sampler": True,
                "class_weights": False,
                "lr": "1e-4",
            }
        )

    print(f"Nombre d'expériences à lancer : {len(experiments)}")
    print(f"Epochs max : {args.epochs}")
    print(f"Patience early stopping : {args.patience}")
    print(f"Run suffix : {run_suffix}")
    print()

    failed_runs = []

    for exp in experiments:
        command = [
            sys.executable,
            "-m",
            "scripts.train_cnn",
            "--architecture",
            exp["architecture"],
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            exp["lr"],
            "--patience",
            str(args.patience),
            "--run-suffix",
            f"{exp['name']}_{run_suffix}",
        ]

        if exp["pretrained"]:
            command.append("--pretrained")

        if exp["freeze_backbone"]:
            command.append("--freeze-backbone")

        if exp["fine_tune"]:
            command.append("--fine-tune")

        if exp["weighted_sampler"]:
            command.append("--weighted-sampler")

        if exp["class_weights"]:
            command.append("--class-weights")

        success = run_command(command)

        if not success:
            failed_runs.append(exp["name"])

    print("\nRésumé des entraînements")
    print("-" * 80)

    if failed_runs:
        print("Runs en erreur :")
        for run in failed_runs:
            print(f"- {run}")
    else:
        print("Tous les entraînements se sont terminés correctement.")


if __name__ == "__main__":
    main()
