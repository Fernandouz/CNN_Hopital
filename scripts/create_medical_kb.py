"""
Script d'indexation de la base de connaissances médicales dans ChromaDB.

Ce script doit être exécuté une seule fois (ou après mise à jour du JSONL)
avant de lancer le pipeline RAG ou l'interface Streamlit.

Usage :
    python scripts/create_medical_kb.py
    python scripts/create_medical_kb.py --kb-path data/base_connaissances_medicales.jsonl
    python scripts/create_medical_kb.py --reset  # Supprime et recrée la collection
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


DEFAULT_KB_PATH = PROJECT_ROOT / "data" / "base_connaissances_medicales.jsonl"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_kb"
COLLECTION_NAME = "medical_knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Indexe la base de connaissances médicales dans ChromaDB"
    )
    parser.add_argument(
        "--kb-path",
        type=str,
        default=str(DEFAULT_KB_PATH),
        help="Chemin vers le fichier JSONL de la base de connaissances"
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default=str(DEFAULT_CHROMA_DIR),
        help="Répertoire de stockage persistant ChromaDB"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help="Nom de la collection ChromaDB"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL,
        help="Modèle sentence-transformers pour les embeddings"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Supprime et recrée la collection (utile après modification du JSONL)"
    )
    return parser.parse_args()


def load_knowledge_base(jsonl_path: Path) -> list[dict]:
    """Charge le fichier JSONL ligne par ligne."""
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Base de connaissances introuvable : {jsonl_path}\n"
            "Vérifier que le fichier existe ou le créer via scripts/create_medical_kb.py"
        )

    documents = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                documents.append(doc)
            except json.JSONDecodeError as e:
                print(f"[Avertissement] Ligne {line_num} ignorée (JSON invalide) : {e}")

    print(f"[KB] {len(documents)} documents chargés depuis {jsonl_path}")
    return documents


def build_chromadb_collection(
    documents: list[dict],
    chroma_dir: Path,
    collection_name: str,
    embedding_model_name: str,
    reset: bool = False,
) -> chromadb.Collection:
    """
    Indexe les documents dans ChromaDB avec des embeddings sentence-transformers.

    Args:
        documents: Liste de dicts depuis le JSONL.
        chroma_dir: Répertoire ChromaDB persistant.
        collection_name: Nom de la collection.
        embedding_model_name: Modèle d'embedding.
        reset: Si True, supprime la collection existante avant indexation.

    Returns:
        Collection ChromaDB avec les documents indexés.
    """
    chroma_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    # Supprime la collection existante si demandé.
    if reset:
        try:
            client.delete_collection(collection_name)
            print(f"[ChromaDB] Collection '{collection_name}' supprimée.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Si la collection est déjà peuplée et pas de reset, on arrête.
    existing_count = collection.count()
    if existing_count > 0 and not reset:
        print(
            f"[ChromaDB] Collection '{collection_name}' déjà indexée "
            f"({existing_count} documents). Utiliser --reset pour forcer la réindexation."
        )
        return collection

    print(f"[Embeddings] Chargement du modèle '{embedding_model_name}'...")
    model = SentenceTransformer(embedding_model_name)

    # Préparation des données pour ChromaDB.
    ids = []
    embeddings = []
    metadatas = []
    doc_texts = []

    for doc in documents:
        doc_id = doc.get("id", f"doc_{len(ids)}")

        # Le texte indexé combine titre + contenu pour une meilleure recherche.
        text_to_embed = f"{doc.get('titre', '')} {doc.get('contenu', '')}"

        # Métadonnées stockées avec chaque document.
        metadata = {
            "type_plaie": doc.get("type_plaie", ""),
            "titre": doc.get("titre", ""),
            "source": doc.get("source", ""),
            "disclaimer": doc.get("disclaimer", ""),
        }

        ids.append(doc_id)
        doc_texts.append(doc.get("contenu", ""))
        metadatas.append(metadata)

    print(f"[Embeddings] Calcul des embeddings pour {len(doc_texts)} documents...")
    embeddings_matrix = model.encode(
        [f"{m['titre']} {t}" for m, t in zip(metadatas, doc_texts)],
        show_progress_bar=True,
        batch_size=32,
    )
    embeddings = embeddings_matrix.tolist()

    # Indexation dans ChromaDB.
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=doc_texts,
        metadatas=metadatas,
    )

    print(f"[ChromaDB] {len(ids)} documents indexés dans la collection '{collection_name}'.")
    return collection


def verify_collection(collection: chromadb.Collection) -> None:
    """Vérifie l'indexation en effectuant quelques requêtes test."""
    print("\n[Vérification] Test de la collection...")

    test_queries = [
        ("Burns", "brûlure traitement premiers secours"),
        ("Abrasions", "écorchure nettoyage cicatrisation"),
        ("Stab_wound", "plaie pénétrante urgence"),
    ]

    for wound_class, query in test_queries:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDING_MODEL)
        embedding = model.encode(query).tolist()

        results = collection.query(
            query_embeddings=[embedding],
            n_results=2,
            where={"type_plaie": wound_class},
            include=["metadatas", "distances"],
        )

        docs_found = len(results["metadatas"][0])
        if docs_found > 0:
            best_doc = results["metadatas"][0][0]
            best_dist = results["distances"][0][0]
            print(
                f"  OK '{wound_class}' -> '{best_doc['titre']}' "
                f"(distance cosinus: {best_dist:.4f})"
            )
        else:
            print(f"  AVERTISSEMENT : aucun résultat pour '{wound_class}'")


def print_collection_stats(collection: chromadb.Collection) -> None:
    """Affiche des statistiques sur la collection."""
    total = collection.count()
    print(f"\n[Stats] Collection '{collection.name}' : {total} documents au total")

    # Compte par type de plaie.
    all_results = collection.get(include=["metadatas"])
    type_counts: dict[str, int] = {}
    for meta in all_results["metadatas"]:
        tp = meta.get("type_plaie", "inconnu")
        type_counts[tp] = type_counts.get(tp, 0) + 1

    print("[Stats] Répartition par type de plaie :")
    for tp, count in sorted(type_counts.items()):
        print(f"  - {tp}: {count} document(s)")


def main():
    args = parse_args()

    kb_path = Path(args.kb_path)
    chroma_dir = Path(args.chroma_dir)

    print(f"Base de connaissances : {kb_path}")
    print(f"Répertoire ChromaDB   : {chroma_dir}")
    print(f"Collection            : {args.collection}")
    print(f"Modèle d'embedding    : {args.embedding_model}")
    print(f"Reset                 : {args.reset}")
    print()

    documents = load_knowledge_base(kb_path)

    collection = build_chromadb_collection(
        documents=documents,
        chroma_dir=chroma_dir,
        collection_name=args.collection,
        embedding_model_name=args.embedding_model,
        reset=args.reset,
    )

    print_collection_stats(collection)
    verify_collection(collection)

    print("\n[OK] Base de connaissances prête. Vous pouvez lancer le pipeline RAG.")
    print("     Lancer l'interface : streamlit run app-streamlit/Home.py")


if __name__ == "__main__":
    main()
