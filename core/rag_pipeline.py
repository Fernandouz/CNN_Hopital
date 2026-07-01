"""
Pipeline RAG (Retrieval-Augmented Generation) pour les recommandations médicales.

Étapes du pipeline :
    1. Diagnostic CNN → classe de plaie + score de confiance
    2. Recherche dans la base de connaissances ChromaDB
    3. Construction du prompt augmenté
    4. Appel au LLM (Ollama ou HuggingFace)
    5. Traçabilité Langfuse
    6. Retour de la recommandation structurée

Usage rapide :
    from core.rag_pipeline import build_rag_pipeline, run_rag

    pipeline = build_rag_pipeline()
    result = run_rag(pipeline, wound_class="Burns", confidence=0.91, top3=[...])
    print(result["recommendation"])
"""

import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from core.ollama_client import call_llm
from core.langfuse_client import get_langfuse_client, create_rag_trace


# Chemin de la base ChromaDB persistante (relatif à la racine du projet).
DEFAULT_CHROMA_DIR = Path(__file__).resolve().parents[1] / "data" / "chroma_kb"
DEFAULT_COLLECTION_NAME = "medical_knowledge_base"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = (
    "Tu es un assistant médical éducatif pour les professionnels de santé. "
    "Tu fournis des informations générales sur les soins des plaies basées uniquement "
    "sur les documents médicaux fournis dans le contexte. "
    "Tu ne poses jamais de diagnostic définitif et tu rappelles toujours que ces "
    "informations ne remplacent pas l'avis d'un professionnel de santé qualifié. "
    "Réponds en français, de façon structurée et professionnelle."
)

RAG_PROMPT_TEMPLATE = """
## Diagnostic du modèle CNN

- **Type de plaie détecté** : {wound_class}
- **Score de confiance** : {confidence_pct}%
- **Top-3 prédictions** :
{top3_formatted}

## Contexte médical (base de connaissances)

{medical_context}

## Demande

Sur la base du diagnostic ci-dessus et des protocoles médicaux fournis, génère une
recommandation de prise en charge structurée incluant :

1. **Évaluation initiale** — points clés à vérifier
2. **Soins immédiats recommandés** — étapes pratiques
3. **Surveillance** — signes d'alerte à surveiller
4. **Orientation** — quand consulter un professionnel / urgences
5. **Limites et disclaimer** — rappel que ceci est une aide éducative

⚠️ DISCLAIMER OBLIGATOIRE : Rappelle explicitement à la fin que cette recommandation
est générée par un modèle d'IA à des fins éducatives uniquement et ne remplace pas
l'avis d'un professionnel de santé qualifié.
"""


class RAGPipeline:
    """
    Pipeline RAG complet pour les recommandations de soins de plaies.

    Attributes:
        chroma_client: Client ChromaDB.
        collection: Collection ChromaDB de la base de connaissances.
        embedding_model: Modèle sentence-transformers pour les requêtes.
        langfuse: Client Langfuse pour la traçabilité.
        ollama_model: Modèle Ollama à utiliser.
    """

    def __init__(
        self,
        chroma_dir: Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        ollama_model: str = "llama3.2",
    ):
        self.ollama_model = ollama_model

        # Connexion à ChromaDB persistant.
        self.chroma_client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Modèle d'embedding pour encoder les requêtes.
        self.embedding_model = SentenceTransformer(embedding_model_name)

        # Client Langfuse (no-op si non configuré).
        self.langfuse = get_langfuse_client()

    def search_knowledge_base(
        self,
        wound_class: str,
        n_results: int = 3,
    ) -> list[dict]:
        """
        Recherche les documents les plus pertinents pour un type de plaie.

        Args:
            wound_class: Classe de plaie (ex: "Burns", "Abrasions").
            n_results: Nombre de documents à retourner.

        Returns:
            Liste de dicts avec "titre", "contenu", "source", "distance".
        """
        query_text = f"protocole soin traitement {wound_class} plaie"
        query_embedding = self.embedding_model.encode(query_text).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),
            where={"type_plaie": wound_class},
            include=["documents", "metadatas", "distances"],
        )

        # Fallback sans filtre par classe si aucun résultat trouvé.
        if not results["documents"][0]:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )

        documents = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            documents.append({
                "titre": meta.get("titre", ""),
                "contenu": doc,
                "source": meta.get("source", ""),
                "type_plaie": meta.get("type_plaie", ""),
                "distance": round(dist, 4),
            })

        return documents

    def build_prompt(
        self,
        wound_class: str,
        confidence: float,
        top3: list[dict],
        documents: list[dict],
    ) -> str:
        """
        Construit le prompt augmenté avec le contexte médical retrouvé.

        Args:
            wound_class: Classe prédite.
            confidence: Score de confiance (0-1).
            top3: Liste des top-3 prédictions [{"class": ..., "probability": ...}].
            documents: Documents médicaux retrouvés par la recherche.

        Returns:
            Prompt complet prêt à être envoyé au LLM.
        """
        confidence_pct = round(confidence * 100, 1)

        top3_lines = "\n".join(
            f"  {i+1}. {item['class']} ({round(item['probability'] * 100, 1)}%)"
            for i, item in enumerate(top3)
        )

        medical_context_parts = []
        for i, doc in enumerate(documents, 1):
            medical_context_parts.append(
                f"### Document {i} : {doc['titre']}\n"
                f"{doc['contenu']}\n"
                f"*Source : {doc['source']}*"
            )
        medical_context = "\n\n".join(medical_context_parts)

        if not medical_context:
            medical_context = "Aucun document spécifique trouvé pour ce type de plaie."

        return RAG_PROMPT_TEMPLATE.format(
            wound_class=wound_class,
            confidence_pct=confidence_pct,
            top3_formatted=top3_lines,
            medical_context=medical_context,
        )

    def run(
        self,
        wound_class: str,
        confidence: float,
        top3: Optional[list[dict]] = None,
        n_docs: int = 3,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Exécute le pipeline RAG complet.

        Args:
            wound_class: Classe de plaie prédite par le CNN.
            confidence: Score de confiance du CNN (0-1).
            top3: Top-3 prédictions [{"class": ..., "probability": ...}].
            n_docs: Nombre de documents à retrouver.
            session_id: Identifiant de session pour Langfuse.

        Returns:
            dict avec :
                - "recommendation" : texte de la recommandation LLM
                - "prompt" : prompt augmenté envoyé au LLM
                - "documents" : documents retrouvés
                - "model" : modèle LLM utilisé
                - "backend" : "ollama" ou "huggingface"
                - "duration_ms" : durée de l'appel LLM
        """
        if top3 is None:
            top3 = [{"class": wound_class, "probability": confidence}]

        # Étape 1 : Recherche dans la base de connaissances.
        documents = self.search_knowledge_base(wound_class, n_results=n_docs)

        # Étape 2 : Construction du prompt augmenté.
        prompt = self.build_prompt(
            wound_class=wound_class,
            confidence=confidence,
            top3=top3,
            documents=documents,
        )

        # Étape 3 : Appel au LLM.
        llm_result = call_llm(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            ollama_model=self.ollama_model,
        )

        recommendation = llm_result["response"]

        # Étape 4 : Traçabilité Langfuse.
        create_rag_trace(
            langfuse=self.langfuse,
            wound_class=wound_class,
            confidence=confidence,
            prompt=prompt,
            llm_response=recommendation,
            model=llm_result["model"],
            duration_ms=llm_result["duration_ms"],
            prompt_tokens=llm_result.get("prompt_tokens", 0),
            completion_tokens=llm_result.get("completion_tokens", 0),
            n_docs_retrieved=len(documents),
            backend=llm_result.get("backend", "unknown"),
            session_id=session_id,
        )

        return {
            "recommendation": recommendation,
            "prompt": prompt,
            "documents": documents,
            "model": llm_result["model"],
            "backend": llm_result.get("backend", "unknown"),
            "duration_ms": llm_result["duration_ms"],
        }


def build_rag_pipeline(
    chroma_dir: Optional[Path] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ollama_model: str = "llama3.2",
) -> RAGPipeline:
    """
    Factory pour créer et retourner un pipeline RAG prêt à l'emploi.

    La base ChromaDB doit avoir été indexée au préalable via
    scripts/create_medical_kb.py.
    """
    if chroma_dir is None:
        chroma_dir = DEFAULT_CHROMA_DIR

    return RAGPipeline(
        chroma_dir=chroma_dir,
        collection_name=collection_name,
        embedding_model_name=embedding_model_name,
        ollama_model=ollama_model,
    )


def run_rag(
    pipeline: RAGPipeline,
    wound_class: str,
    confidence: float,
    top3: Optional[list[dict]] = None,
    n_docs: int = 3,
    session_id: Optional[str] = None,
) -> dict:
    """Wrapper fonctionnel pour appeler pipeline.run()."""
    return pipeline.run(
        wound_class=wound_class,
        confidence=confidence,
        top3=top3,
        n_docs=n_docs,
        session_id=session_id,
    )
