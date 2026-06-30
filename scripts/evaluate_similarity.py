from pathlib import Path
import argparse
import json
import os
import random
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

from core.image_similarity import (  # noqa: E402
    CNNEmbeddingExtractor,
    DEFAULT_COLLECTION_NAME,
    get_chroma_collection,
    save_similarity_figure,
    search_chroma_by_embedding,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate visual similarity retrieval on indexed images."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/resnet50_best.pt",
        help="Checkpoint CNN utilise pour extraire les embeddings requetes.",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default="data/processed/splits",
        help="Dossier contenant les fichiers de split CSV.",
    )
    parser.add_argument(
        "--query-splits",
        type=str,
        default="test",
        help="Splits utilises comme requetes, separes par des virgules.",
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
        help="Nombre de voisins evalues apres exclusion de la requete.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Nombre maximum de requetes. 0 signifie toutes les images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour l'echantillonnage des requetes.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/similarity/evaluation",
        help="Dossier de sortie des rapports et figures.",
    )
    parser.add_argument(
        "--num-figures",
        type=int,
        default=4,
        help="Nombre de visualisations query + top-K a sauvegarder.",
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


def load_queries(split_dir, split_names):
    dataframes = []

    for split_name in split_names:
        split_path = split_dir / f"{split_name}.csv"

        if not split_path.exists():
            raise FileNotFoundError(f"Split introuvable : {split_path}")

        dataframe = pd.read_csv(split_path)
        missing_columns = {"path", "class"} - set(dataframe.columns)

        if missing_columns:
            raise ValueError(
                f"Colonnes manquantes dans {split_path} : "
                f"{sorted(missing_columns)}"
            )

        dataframe = dataframe.copy()
        dataframe["split"] = split_name
        dataframe["resolved_path"] = dataframe["path"].apply(resolve_project_path)
        dataframes.append(dataframe)

    queries = pd.concat(dataframes, ignore_index=True)

    missing_files = [
        str(path)
        for path in queries["resolved_path"]
        if not Path(path).exists()
    ]

    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(
            "Certaines images de requete sont introuvables :\n"
            f"{preview}"
        )

    return queries


def evaluate_query(extractor, collection, image_path, true_class, top_k):
    embedding = extractor.extract_image(image_path)
    results = search_chroma_by_embedding(
        query_embedding=embedding,
        collection=collection,
        top_k=top_k,
        exclude_image_path=image_path,
    )

    same_class_flags = [
        result["class"] == true_class
        for result in results
    ]
    first_match_rank = None

    for rank, is_match in enumerate(same_class_flags, start=1):
        if is_match:
            first_match_rank = rank
            break

    retrieved_count = len(results)
    precision_at_k = (
        sum(same_class_flags) / retrieved_count
        if retrieved_count > 0
        else 0.0
    )
    hit_at_k = any(same_class_flags)
    top1_match = same_class_flags[0] if same_class_flags else False
    reciprocal_rank = 1.0 / first_match_rank if first_match_rank else 0.0
    top1_class = results[0]["class"] if results else None
    top1_score = results[0]["score"] if results else None

    return {
        "results": results,
        "metrics": {
            "retrieved_count": retrieved_count,
            "top1_match": bool(top1_match),
            "hit_at_k": bool(hit_at_k),
            "precision_at_k": float(precision_at_k),
            "reciprocal_rank": float(reciprocal_rank),
            "first_match_rank": first_match_rank,
            "top1_class": top1_class,
            "top1_score": top1_score,
        },
    }


def summarize_details(details, top_k):
    detail_df = pd.DataFrame(details)
    class_df = (
        detail_df
        .groupby("true_class")
        .agg(
            num_queries=("query_image", "count"),
            top1_accuracy=("top1_match", "mean"),
            hit_at_k=("hit_at_k", "mean"),
            precision_at_k=("precision_at_k", "mean"),
            mrr=("reciprocal_rank", "mean"),
        )
        .reset_index()
        .sort_values("true_class")
    )

    summary = {
        "num_queries": int(len(detail_df)),
        "top_k": int(top_k),
        "top1_accuracy": float(detail_df["top1_match"].mean()),
        "hit_at_k": float(detail_df["hit_at_k"].mean()),
        "mean_precision_at_k": float(detail_df["precision_at_k"].mean()),
        "mrr": float(detail_df["reciprocal_rank"].mean()),
        "mean_retrieved_count": float(detail_df["retrieved_count"].mean()),
        "per_class": class_df.to_dict(orient="records"),
    }

    return summary, detail_df, class_df


def save_outputs(output_dir, summary, detail_df, class_df):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "similarity_evaluation_summary.json"
    details_path = output_dir / "similarity_evaluation_details.csv"
    per_class_path = output_dir / "similarity_evaluation_by_class.csv"

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    detail_df.to_csv(details_path, index=False)
    class_df.to_csv(per_class_path, index=False)

    return summary_path, details_path, per_class_path


def main():
    args = parse_args()

    checkpoint_path = resolve_project_path(args.checkpoint)
    split_dir = resolve_project_path(args.split_dir)
    persist_dir = resolve_project_path(args.persist_dir)
    output_dir = resolve_project_path(args.output_dir)
    split_names = [
        split.strip()
        for split in args.query_splits.split(",")
        if split.strip()
    ]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    queries = load_queries(split_dir=split_dir, split_names=split_names)

    if args.max_queries > 0 and args.max_queries < len(queries):
        queries = (
            queries
            .sample(n=args.max_queries, random_state=args.seed)
            .reset_index(drop=True)
        )

    random.seed(args.seed)
    start_time = time.time()

    extractor = CNNEmbeddingExtractor(checkpoint_path=checkpoint_path)
    collection = get_chroma_collection(
        persist_dir=persist_dir,
        collection_name=args.collection_name,
    )

    print("Evaluation de la recherche par similarite")
    print("-" * 45)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Collection : {args.collection_name}")
    print(f"Splits requetes : {', '.join(split_names)}")
    print(f"Nombre de requetes : {len(queries)}")
    print(f"Top-K : {args.top_k}")

    details = []
    example_payloads = []

    for index, row in queries.iterrows():
        image_path = Path(row["resolved_path"])
        true_class = row["class"]
        evaluation = evaluate_query(
            extractor=extractor,
            collection=collection,
            image_path=image_path,
            true_class=true_class,
            top_k=args.top_k,
        )
        metrics = evaluation["metrics"]
        results = evaluation["results"]

        details.append({
            "query_image": str(image_path),
            "split": row["split"],
            "true_class": true_class,
            "retrieved_count": metrics["retrieved_count"],
            "top1_class": metrics["top1_class"],
            "top1_score": metrics["top1_score"],
            "top1_match": metrics["top1_match"],
            "hit_at_k": metrics["hit_at_k"],
            "precision_at_k": metrics["precision_at_k"],
            "reciprocal_rank": metrics["reciprocal_rank"],
            "first_match_rank": metrics["first_match_rank"],
        })

        if len(example_payloads) < args.num_figures:
            example_payloads.append({
                "image_path": image_path,
                "true_class": true_class,
                "results": results,
                "index": index,
            })

    summary, detail_df, class_df = summarize_details(
        details=details,
        top_k=args.top_k,
    )
    summary["duration_seconds"] = round(time.time() - start_time, 3)
    summary["checkpoint"] = str(checkpoint_path)
    summary["collection_name"] = args.collection_name
    summary["query_splits"] = split_names

    figures_dir = output_dir / "figures"
    figure_paths = []

    for payload in example_payloads:
        image_path = payload["image_path"]
        safe_stem = image_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
        figure_path = figures_dir / f"query_topk_{payload['index']}_{safe_stem}.png"
        save_similarity_figure(
            query_image_path=image_path,
            results=payload["results"],
            output_path=figure_path,
            title=f"Classe requete : {payload['true_class']}",
        )
        figure_paths.append(str(figure_path))

    summary["figure_paths"] = figure_paths

    summary_path, details_path, per_class_path = save_outputs(
        output_dir=output_dir,
        summary=summary,
        detail_df=detail_df,
        class_df=class_df,
    )

    print("\nResultats")
    print(f"Top-1 accuracy : {summary['top1_accuracy']:.4f}")
    print(f"Hit@{args.top_k} : {summary['hit_at_k']:.4f}")
    print(f"Precision@{args.top_k} moyenne : {summary['mean_precision_at_k']:.4f}")
    print(f"MRR : {summary['mrr']:.4f}")
    print(f"Resume JSON : {summary_path}")
    print(f"Details CSV : {details_path}")
    print(f"Par classe CSV : {per_class_path}")

    if figure_paths:
        print("Figures :")
        for figure_path in figure_paths:
            print(f"- {figure_path}")


if __name__ == "__main__":
    main()
