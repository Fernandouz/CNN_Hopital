
from pathlib import Path
import argparse
import json
import sys
import time

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from core.image_similarity import (  # noqa: E402
    DEFAULT_COLLECTION_NAME,
    index_images_in_chroma,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a ChromaDB similarity index from CNN embeddings."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/resnet50_best.pt",
        help="Chemin du checkpoint CNN utilise pour extraire les embeddings.",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default="data/processed/splits",
        help="Dossier contenant train.csv, val.csv et test.csv.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Splits a indexer, separes par des virgules.",
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
        "--batch-size",
        type=int,
        default=32,
        help="Taille de batch pour l'extraction des embeddings.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="reports/similarity/index_summary.json",
        help="Chemin du resume JSON genere.",
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

    path_text = path.as_posix()
    if path_text.startswith("../data/"):
        return PROJECT_ROOT / path_text.replace("../data/", "data/", 1)

    return PROJECT_ROOT / path


def load_split_dataframe(split_dir, split_name):
    split_path = split_dir / f"{split_name}.csv"

    if not split_path.exists():
        raise FileNotFoundError(f"Split introuvable : {split_path}")

    dataframe = pd.read_csv(split_path)
    required_columns = {"path", "class"}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes dans {split_path} : "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()
    dataframe["split"] = split_name
    dataframe["resolved_path"] = dataframe["path"].apply(resolve_image_path)

    return dataframe


def build_index_dataframe(split_dir, split_names):
    dataframes = [
        load_split_dataframe(split_dir, split_name)
        for split_name in split_names
    ]
    dataframe = pd.concat(dataframes, ignore_index=True)

    missing_files = [
        str(path)
        for path in dataframe["resolved_path"]
        if not Path(path).exists()
    ]

    if missing_files:
        preview = "\n".join(missing_files[:10])
        raise FileNotFoundError(
            "Certaines images referencees dans les splits sont introuvables :\n"
            f"{preview}"
        )

    return dataframe


def write_summary(summary_path, summary):
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)


def main():
    args = parse_args()

    checkpoint_path = resolve_project_path(args.checkpoint)
    split_dir = resolve_project_path(args.split_dir)
    persist_dir = resolve_project_path(args.persist_dir)
    summary_path = resolve_project_path(args.summary_path)
    split_names = [
        split.strip()
        for split in args.splits.split(",")
        if split.strip()
    ]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    start_time = time.time()
    dataframe = build_index_dataframe(
        split_dir=split_dir,
        split_names=split_names,
    )

    print("Construction de l'index de similarite")
    print("-" * 40)
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Splits : {', '.join(split_names)}")
    print(f"Images : {len(dataframe)}")
    print(f"Collection : {args.collection_name}")
    print(f"Persistance : {persist_dir}")

    collection = index_images_in_chroma(
        image_paths=dataframe["resolved_path"].tolist(),
        labels=dataframe["class"].tolist(),
        checkpoint_path=checkpoint_path,
        persist_dir=persist_dir,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
    )

    duration_seconds = time.time() - start_time
    class_counts = dataframe["class"].value_counts().sort_index().to_dict()
    split_counts = dataframe["split"].value_counts().sort_index().to_dict()

    summary = {
        "checkpoint": str(checkpoint_path),
        "architecture": checkpoint.get("architecture"),
        "img_size": checkpoint.get("img_size"),
        "collection_name": args.collection_name,
        "persist_dir": str(persist_dir),
        "num_images_indexed": int(len(dataframe)),
        "collection_count": int(collection.count()),
        "splits": split_names,
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "class_counts": {key: int(value) for key, value in class_counts.items()},
        "batch_size": int(args.batch_size),
        "duration_seconds": round(duration_seconds, 3),
        "summary_path": str(summary_path),
    }

    write_summary(summary_path=summary_path, summary=summary)

    print("\nIndex genere avec succes")
    print(f"Images indexees : {summary['num_images_indexed']}")
    print(f"Documents dans ChromaDB : {summary['collection_count']}")
    print(f"Resume : {summary_path}")


if __name__ == "__main__":
    main()
