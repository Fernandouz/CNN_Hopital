from pathlib import Path
import sys
import argparse
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from core.inference import predict_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict wound class for a single image."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Chemin du checkpoint .pt."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Chemin de l'image à prédire."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Nombre de prédictions retournées."
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Afficher le résultat au format JSON."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    image_path = PROJECT_ROOT / args.image

    result = predict_image(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        top_k=args.top_k
    )

    if args.json:
        print(json.dumps(result, indent=4, ensure_ascii=False))
        return

    print("\nPrédiction image unique")
    print("-" * 40)
    print(f"Image : {result['image_path']}")
    print(f"Architecture : {result['architecture']}")
    print(f"Classe prédite : {result['predicted_class']}")
    print(f"Confiance : {result['confidence']:.4f}")

    print("\nTop prédictions :")
    for pred in result["top_k"]:
        print(
            f"{pred['rank']}. {pred['class']} "
            f"— {pred['probability']:.4f}"
        )


if __name__ == "__main__":
    main()
