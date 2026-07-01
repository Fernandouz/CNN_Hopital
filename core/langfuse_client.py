"""
Client Langfuse pour la traçabilité des appels LLM du pipeline RAG.

Langfuse trace chaque appel LLM avec : prompt, réponse, latence, tokens,
classe prédite et score de confiance.

Configuration via variables d'environnement (fichier .env) :
    LANGFUSE_PUBLIC_KEY=pk-lf-...
    LANGFUSE_SECRET_KEY=sk-lf-...
    LANGFUSE_HOST=https://cloud.langfuse.com  (ou URL locale)

Si les clés ne sont pas configurées, le client retourne un objet no-op
qui ne lève pas d'erreur : le pipeline RAG fonctionne sans traçabilité.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class _NoOpTrace:
    """Objet trace factice utilisé quand Langfuse n'est pas configuré."""

    def __init__(self, *args, **kwargs):
        pass

    def generation(self, *args, **kwargs):
        return _NoOpGeneration()

    def update(self, *args, **kwargs):
        pass


class _NoOpGeneration:
    """Objet génération factice."""

    def update(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass


class _NoOpLangfuse:
    """Client Langfuse factice utilisé quand les clés ne sont pas configurées."""

    def trace(self, *args, **kwargs):
        return _NoOpTrace()

    def flush(self):
        pass


def get_langfuse_client():
    """
    Retourne un client Langfuse initialisé ou un client no-op si non configuré.

    Usage:
        langfuse = get_langfuse_client()
        trace = langfuse.trace(name="rag-call", ...)
        ...
        langfuse.flush()
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        print(
            "[Langfuse] Clés non configurées — traçabilité désactivée. "
            "Définir LANGFUSE_PUBLIC_KEY et LANGFUSE_SECRET_KEY dans .env."
        )
        return _NoOpLangfuse()

    try:
        from langfuse import Langfuse  # type: ignore

        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        print(f"[Langfuse] Client initialisé (host={host})")
        return client
    except ImportError:
        print(
            "[Langfuse] Package 'langfuse' non installé — traçabilité désactivée. "
            "Installez avec : pip install langfuse"
        )
        return _NoOpLangfuse()
    except Exception as exc:
        print(f"[Langfuse] Erreur d'initialisation : {exc} — traçabilité désactivée.")
        return _NoOpLangfuse()


def create_rag_trace(
    langfuse,
    wound_class: str,
    confidence: float,
    prompt: str,
    llm_response: str,
    model: str,
    duration_ms: int,
    prompt_tokens: int,
    completion_tokens: int,
    n_docs_retrieved: int,
    backend: str = "ollama",
    session_id: Optional[str] = None,
) -> None:
    """
    Crée une trace Langfuse complète pour un appel RAG.

    Args:
        langfuse: Client Langfuse (réel ou no-op).
        wound_class: Classe de plaie prédite par le CNN.
        confidence: Score de confiance du CNN (0-1).
        prompt: Prompt augmenté envoyé au LLM.
        llm_response: Réponse générée par le LLM.
        model: Nom du modèle LLM utilisé.
        duration_ms: Durée de l'appel LLM en millisecondes.
        prompt_tokens: Nombre de tokens du prompt.
        completion_tokens: Nombre de tokens de la réponse.
        n_docs_retrieved: Nombre de documents retrouvés par la recherche RAG.
        backend: "ollama" ou "huggingface".
        session_id: Identifiant de session optionnel.
    """
    trace = langfuse.trace(
        name="rag-medical-recommendation",
        metadata={
            "wound_class": wound_class,
            "confidence": round(confidence, 4),
            "n_docs_retrieved": n_docs_retrieved,
            "backend": backend,
        },
        session_id=session_id,
    )

    trace.generation(
        name="llm-recommendation",
        model=model,
        input=prompt,
        output=llm_response,
        usage={
            "input": prompt_tokens,
            "output": completion_tokens,
            "unit": "TOKENS",
        },
        metadata={
            "duration_ms": duration_ms,
            "backend": backend,
        },
    )

    langfuse.flush()
