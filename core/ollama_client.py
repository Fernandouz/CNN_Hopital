"""
Client LLM pour le pipeline RAG médical.

Supporte Ollama (local) en priorité et HuggingFace comme fallback.
"""

import json
import time
from typing import Optional

import requests


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_TIMEOUT = 120  # secondes


def is_ollama_available(base_url: str = OLLAMA_BASE_URL) -> bool:
    """Vérifie si le serveur Ollama est accessible."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.Timeout:
        return False


def list_ollama_models(base_url: str = OLLAMA_BASE_URL) -> list[str]:
    """Retourne la liste des modèles disponibles dans Ollama."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def call_ollama(
    prompt: str,
    model: str = DEFAULT_OLLAMA_MODEL,
    system_prompt: Optional[str] = None,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    """
    Appelle l'API Ollama et retourne la réponse.

    Returns:
        dict avec les clés :
            - "response" : texte généré
            - "model" : modèle utilisé
            - "duration_ms" : durée en ms
            - "prompt_tokens" : estimation tokens (optionnel)
            - "completion_tokens" : estimation tokens (optionnel)
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1024,
        }
    }

    if system_prompt:
        payload["system"] = system_prompt

    start_time = time.time()
    response = requests.post(
        f"{base_url}/api/generate",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    duration_ms = int((time.time() - start_time) * 1000)

    data = response.json()

    return {
        "response": data.get("response", ""),
        "model": model,
        "duration_ms": duration_ms,
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
    }


def call_huggingface_local(
    prompt: str,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 512,
) -> dict:
    """
    Fallback : appelle un modèle HuggingFace local (pipeline transformers).

    Nécessite : pip install transformers accelerate
    Le modèle est chargé en CPU (pas de CUDA requis).
    Premier appel : téléchargement du modèle (~3 Go pour Qwen2.5-1.5B).
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "transformers et accelerate sont requis. "
            "Installez avec : pip install transformers accelerate"
        ) from exc

    print(f"[HuggingFace] Chargement du modèle '{model_name}' (premier appel = téléchargement)...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # float32 pour CPU (float16 nécessite CUDA)
        # device_map et low_cpu_mem_usage désactivés : incompatibles avec CPU-only
        # et certaines versions de PyTorch (erreur __path__._path).
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    pipe = pipeline(
        "text-generation",
        model=model_hf,
        tokenizer=tokenizer,
        device=-1,  # -1 = CPU
    )

    start_time = time.time()
    output = pipe(messages, max_new_tokens=max_new_tokens)
    duration_ms = int((time.time() - start_time) * 1000)

    generated = output[0]["generated_text"]
    # La pipeline retourne les messages + la réponse, extraire uniquement la réponse.
    if isinstance(generated, list):
        text = generated[-1].get("content", "")
    else:
        text = str(generated)

    return {
        "response": text,
        "model": model_name,
        "duration_ms": duration_ms,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    hf_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    base_url: str = OLLAMA_BASE_URL,
) -> dict:
    """
    Point d'entrée unifié : essaie Ollama, puis HuggingFace si indisponible.

    Returns:
        dict avec "response", "model", "duration_ms", "backend",
        "prompt_tokens", "completion_tokens"
    """
    if is_ollama_available(base_url):
        result = call_ollama(
            prompt=prompt,
            model=ollama_model,
            system_prompt=system_prompt,
            base_url=base_url,
        )
        result["backend"] = "ollama"
        return result

    # Fallback HuggingFace local.
    result = call_huggingface_local(
        prompt=prompt,
        model_name=hf_model,
        system_prompt=system_prompt,
    )
    result["backend"] = "huggingface"
    return result
