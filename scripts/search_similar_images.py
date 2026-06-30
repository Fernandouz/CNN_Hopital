from pathlib import Path
import argparse
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from core.image_similarity import (  # noqa: E402
    DEFAULT_COLLECTION_NAME,
    save_similarity_figure,
    search_similar_images,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search similar wound images from the ChromaDB index."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Chemin de l'image requete.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/resnet50_best.pt",
        help="Checkpoint CNN utilise pour extraire l'embedding requete.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="data/processed/chroma",
        help="Dossier de persistance ChromaDB.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help="Nom de la collection ChromaDB.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Nombre de voisins similaires a retourner.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Afficher les resultats au format JSON.",
    )
    parser.add_argument(
        "--exclude-query",
        action="store_true",
        help="Exclure l'image requete si elle est deja presente dans l'index.",
    )
    parser.add_argument(
        "--save-figure",
        type=str,
        default=None,
        help="Chemin PNG pour sauvegarder une visualisation query + top-K.",
    )

    return parser.parse_args()


def resolve_project_path(path):
    path = Path(path)

    if path.is_absolute():
        return path

    path_text = path.as_posix()
    if path_text.startswith("../data/"):
        return PROJECT_ROOT / path_text.replace("../data/", "data/", 1)

    return PROJECT_ROOT / path


def main():
    args = parse_args()

    image_path = resolve_project_path(args.image)
    checkpoint_path = resolve_project_path(args.checkpoint)
    persist_dir = resolve_project_path(args.persist_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    results = search_similar_images(
        query_image_path=image_path,
        checkpoint_path=checkpoint_path,
        persist_dir=persist_dir,
        collection_name=args.collection_name,
        top_k=args.top_k,
        exclude_image_path=image_path if args.exclude_query else None,
    )

    figure_path = None
    if args.save_figure:
        figure_path = resolve_project_path(args.save_figure)
        save_similarity_figure(
            query_image_path=image_path,
            results=results,
            output_path=figure_path,
            title=f"Recherche par similarite - {image_path.name}",
        )

    output = {
        "query_image": str(image_path),
        "checkpoint": str(checkpoint_path),
        "collection_name": args.collection_name,
        "top_k": args.top_k,
        "exclude_query": args.exclude_query,
        "figure_path": str(figure_path) if figure_path else None,
        "results": results,
    }

    if args.json:
        print(json.dumps(output, indent=4, ensure_ascii=False))
        return

    print("\nRecherche d'images similaires")
    print("-" * 40)
    print(f"Image requete : {image_path}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Collection : {args.collection_name}")
    print(f"Top-K : {args.top_k}")
    print(f"Exclusion requete : {args.exclude_query}")

    print("\nVoisins trouves :")
    for item in results:
        print(
            f"{item['rank']}. classe={item['class']} | "
            f"score={item['score']:.4f} | "
            f"distance={item['distance']:.4f} | "
            f"{item['image_path']}"
        )

    if figure_path:
        print(f"\nFigure sauvegardee : {figure_path}")


if __name__ == "__main__":
    main()
