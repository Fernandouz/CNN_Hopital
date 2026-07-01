from pathlib import Path
import sys
import argparse
import json
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

from core.pipeline import analyze_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Complete pipeline: OOD detection + CNN classification."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Chemin de l'image à analyser."
    )

    parser.add_argument(
        "--cnn-checkpoint",
        type=str,
        required=True,
        help="Checkpoint du meilleur modèle CNN."
    )

    parser.add_argument(
        "--autoencoder-checkpoint",
        type=str,
        required=True,
        help="Checkpoint de l'autoencoder OOD."
    )

    parser.add_argument(
        "--ood-threshold",
        type=str,
        required=True,
        help="Fichier JSON contenant le seuil OOD."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Nombre de classes retournées par le CNN."
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Afficher la sortie complète au format JSON."
    )

    parser.add_argument(
        "--include-grad-cam",
        action="store_true",
        help="Générer une explication Grad-CAM si l'image est acceptée par le filtre OOD."
    )

    parser.add_argument(
        "--grad-cam-output-dir",
        type=str,
        default="reports/xai/pipeline",
        help="Dossier de sortie des images Grad-CAM."
    )

    parser.add_argument(
        "--grad-cam-target-class",
        type=str,
        default=None,
        help="Classe cible Grad-CAM. Par défaut, utilise la classe prédite."
    )

    parser.add_argument(
        "--grad-cam-alpha",
        type=float,
        default=0.45,
        help="Poids visuel de la heatmap dans la superposition."
    )

    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main():
    args = parse_args()

    result = analyze_image(
        image_path=resolve_path(args.image),
        cnn_checkpoint_path=resolve_path(args.cnn_checkpoint),
        autoencoder_checkpoint_path=resolve_path(args.autoencoder_checkpoint),
        ood_threshold_path=resolve_path(args.ood_threshold),
        top_k=args.top_k,
        include_grad_cam=args.include_grad_cam,
        grad_cam_output_dir=resolve_path(args.grad_cam_output_dir),
        grad_cam_target_class=args.grad_cam_target_class,
        grad_cam_alpha=args.grad_cam_alpha,
    )

    if args.json:
        print(json.dumps(result, indent=4, ensure_ascii=False))
        return

    print("\nAnalyse complète de l'image")
    print("=" * 50)

    print(f"Statut : {result['status']}")
    print(f"Raison : {result['reason']}")

    print("\nDétection OOD")
    print("-" * 50)
    print(f"Décision : {result['ood']['decision']}")
    print(f"Score anomalie : {result['ood']['anomaly_score']:.8f}")
    print(f"Seuil OOD : {result['ood']['threshold']:.8f}")

    if result["status"] == "rejected":
        print("\nAucune classification CNN n'est effectuée.")
        return

    prediction = result["prediction"]

    print("\nClassification CNN")
    print("-" * 50)
    print(f"Architecture : {prediction['architecture']}")
    print(f"Classe prédite : {prediction['predicted_class']}")
    print(f"Confiance : {prediction['confidence']:.4f}")

    print("\nTop prédictions :")
    for pred in prediction["top_k"]:
        print(
            f"{pred['rank']}. {pred['class']} "
            f"— {pred['probability']:.4f}"
        )

    if result["grad_cam"] is not None:
        grad_cam = result["grad_cam"]
        print("\nExplicabilité Grad-CAM")
        print("-" * 50)
        print(f"Classe cible : {grad_cam['target_class']}")
        print(f"Overlay : {grad_cam['overlay_path']}")


if __name__ == "__main__":
    main()
