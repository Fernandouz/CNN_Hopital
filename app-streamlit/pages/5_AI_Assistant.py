"""Page Streamlit Assistant IA médical avec RAG."""

import sys
import uuid
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from core.config import CLASS_NAMES, PROJECT_ROOT
from core.ollama_client import is_ollama_available, list_ollama_models
from core.streamlit_ui import apply_custom_style, render_sidebar, page_header
from core.ui_metrics import metric_card


KB_PATH = PROJECT_ROOT / "data" / "base_connaissances_medicales.jsonl"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_kb"
DISCLAIMER = (
    "Cette application est réalisée dans un cadre pédagogique. Elle ne constitue "
    "pas un dispositif médical et ne remplace pas l'avis d'un professionnel de santé."
)

WOUND_LABELS_FR = {
    "Abrasions": "Abrasion",
    "Bruises": "Contusion / hématome",
    "Burns": "Brûlure",
    "Cut": "Coupure",
    "Ingrown_nails": "Ongle incarné",
    "Laceration": "Lacération",
    "Stab_wound": "Plaie par arme blanche",
}


def get_prediction_from_session() -> dict | None:
    """Convertit le résultat de la page Prédiction au format attendu par le RAG."""
    page_state = st.session_state.get("prediction_page_state")
    if not page_state:
        return None

    result = page_state.get("result", {})
    if result.get("ood", {}).get("is_ood"):
        return None

    classification = result.get("classification")
    if not classification:
        return None

    top3 = [
        {"class": item["classe"], "probability": item["score"]}
        for item in classification.get("top3", [])
    ]

    return {
        "class": classification["classe_predite"],
        "confidence": classification["confiance"],
        "top3": top3
        or [
            {
                "class": classification["classe_predite"],
                "probability": classification["confiance"],
            }
        ],
        "source": "prediction_page",
    }


def build_manual_prediction(wound_class: str, confidence: float) -> dict:
    """Construit un diagnostic manuel compatible avec le pipeline RAG."""
    return {
        "class": wound_class,
        "confidence": confidence,
        "top3": [{"class": wound_class, "probability": confidence}],
        "source": "manual",
    }


def format_label(wound_class: str) -> str:
    """Nom lisible d'une classe de plaie."""
    return WOUND_LABELS_FR.get(wound_class, wound_class)


def missing_dependency_message(exc: Exception) -> str:
    """Message court quand les dépendances LLM ne sont pas installées."""
    return (
        "Dépendances RAG/LLM manquantes ou indisponibles. Installez-les avec : "
        "`pip install -r requirements_llm.txt`.\n\n"
        f"Détail technique : {exc}"
    )


@st.cache_resource(show_spinner=False)
def load_rag_pipeline(ollama_model: str):
    """Charge le pipeline RAG en différant les imports lourds."""
    if not KB_PATH.exists():
        return (
            None,
            "Base de connaissances introuvable. Vérifiez le fichier "
            "`data/base_connaissances_medicales.jsonl`.",
        )

    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        return (
            None,
            "Base ChromaDB non indexée. Lancez `python scripts/create_medical_kb.py` "
            "depuis la racine du projet.",
        )

    # Après indexation, le modèle d'embedding est dans le cache local.
    # Cela évite un appel réseau HuggingFace à chaque chargement de la page.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        from core.rag_pipeline import build_rag_pipeline
    except Exception as exc:
        return None, missing_dependency_message(exc)

    try:
        return build_rag_pipeline(ollama_model=ollama_model), None
    except Exception as exc:
        return None, f"Erreur pendant l'initialisation du pipeline RAG : {exc}"


def run_recommendation(pipeline, prediction: dict, n_docs: int) -> dict:
    """Exécute le RAG via l'API publique de la partie 5."""
    from core.rag_pipeline import run_rag

    return run_rag(
        pipeline=pipeline,
        wound_class=prediction["class"],
        confidence=prediction["confidence"],
        top3=prediction["top3"],
        n_docs=n_docs,
        session_id=str(uuid.uuid4()),
    )


def render_prediction_summary(prediction: dict):
    """Affiche le diagnostic utilisé comme entrée RAG."""
    cols = st.columns(3)
    with cols[0]:
        metric_card("Classe retenue", format_label(prediction["class"]))
    with cols[1]:
        metric_card("Confiance", f"{prediction['confidence'] * 100:.1f} %")
    with cols[2]:
        source = "Page Prédiction" if prediction["source"] == "prediction_page" else "Saisie manuelle"
        metric_card("Source", source)

    st.write("Top prédictions transmises au RAG")
    for item in prediction["top3"]:
        st.progress(
            float(item["probability"]),
            text=f"{format_label(item['class'])} - {item['probability'] * 100:.1f} %",
        )


def render_documents(documents: list[dict]):
    """Affiche les documents retrouvés dans ChromaDB."""
    with st.expander(f"Documents médicaux retrouvés ({len(documents)})", expanded=False):
        for index, doc in enumerate(documents, start=1):
            st.markdown(f"**{index}. {doc.get('titre', 'Document')}**")
            st.caption(
                f"Type : {format_label(doc.get('type_plaie', ''))} | "
                f"Source : {doc.get('source', 'non renseignée')} | "
                f"Distance : {doc.get('distance', '-')}"
            )
            st.write(doc.get("contenu", ""))
            if index < len(documents):
                st.divider()


def render_rag_button(
    pipeline,
    prediction: dict,
    n_docs: int,
    button_key: str,
    ollama_available: bool,
):
    """Affiche le bouton unique qui déclenche l'analyse RAG."""
    if st.button("Lancer l'analyse RAG", type="primary", key=button_key):
        with st.spinner("Recherche documentaire et génération LLM en cours..."):
            try:
                result = run_recommendation(pipeline, prediction, n_docs=n_docs)
                st.session_state["ai_assistant_result"] = result
            except Exception as exc:
                st.error(f"Erreur pendant la génération RAG : {exc}")
                if not ollama_available:
                    st.info(
                        "Ollama n'est pas joignable. Lancez `ollama serve` après "
                        "`ollama pull llama3.2`, ou installez le fallback HuggingFace."
                    )


st.set_page_config(
    page_title="Assistant IA",
    page_icon=":material/smart_toy:",
    layout="wide",
)
apply_custom_style()
render_sidebar()
page_header(
    "Assistant IA médical",
    subtitle="RAG pédagogique basé sur la base de connaissances médicale du projet",
    icon="smart_toy",
)

st.markdown(
    f"""
    <div style="
        margin-bottom:1rem;
        background-color:#142420;
        border:1px solid #1D9E75;
        border-left:5px solid #1D9E75;
        border-radius:8px;
        padding:0.9rem 1rem;
        color:#EAF6F1;
        line-height:1.5;
    ">
        {DISCLAIMER}
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Assistant RAG")

    ollama_available = is_ollama_available()
    if ollama_available:
        st.success("Ollama disponible")
        models = list_ollama_models()
        ollama_model = st.selectbox(
            "Modèle Ollama",
            options=models or ["llama3.2"],
        )
    else:
        st.warning("Ollama non détecté")
        ollama_model = st.text_input("Modèle Ollama cible", value="llama3.2")
        st.caption("Le pipeline tentera le fallback HuggingFace si les dépendances sont installées.")

    n_docs = st.slider("Documents récupérés", min_value=1, max_value=5, value=3)


pipeline, pipeline_error = load_rag_pipeline(ollama_model)

if pipeline_error:
    st.error(pipeline_error)
    st.info(
        "Étapes attendues : installer `requirements_llm.txt`, vérifier "
        "`data/base_connaissances_medicales.jsonl`, puis indexer ChromaDB."
    )
    st.stop()


session_prediction = get_prediction_from_session()

tab_prediction, tab_manual = st.tabs(["Diagnostic courant", "Diagnostic manuel"])

with tab_prediction:
    if session_prediction is None:
        st.info(
            "Aucun diagnostic CNN valide n'est disponible en session. "
            "Lancez une prédiction depuis la page Prédiction ou utilisez l'onglet manuel."
        )
    else:
        st.subheader("Diagnostic issu de la page Prédiction")
        render_prediction_summary(session_prediction)
        render_rag_button(
            pipeline=pipeline,
            prediction=session_prediction,
            n_docs=n_docs,
            button_key="run_rag_current_prediction",
            ollama_available=ollama_available,
        )

with tab_manual:
    st.subheader("Saisie manuelle")
    manual_class = st.selectbox(
        "Type de plaie",
        options=CLASS_NAMES,
        format_func=format_label,
    )
    manual_confidence = st.slider(
        "Confiance simulée",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.01,
    )
    manual_prediction = build_manual_prediction(manual_class, manual_confidence)
    render_prediction_summary(manual_prediction)
    render_rag_button(
        pipeline=pipeline,
        prediction=manual_prediction,
        n_docs=n_docs,
        button_key="run_rag_manual_prediction",
        ollama_available=ollama_available,
    )


result = st.session_state.get("ai_assistant_result")
if result:
    st.divider()
    st.subheader("Recommandation générée")
    st.success(
        f"Générée via {result.get('backend', 'inconnu')} "
        f"({result.get('model', 'modèle inconnu')}) en {result.get('duration_ms', 0)} ms."
    )
    st.markdown(result["recommendation"])
    render_documents(result.get("documents", []))

    with st.expander("Détails techniques"):
        st.text_area("Prompt augmenté envoyé au LLM", result.get("prompt", ""), height=280)


st.divider()
st.caption("Partie 5 : Assistant LLM/RAG avec ChromaDB, Ollama/HuggingFace et Langfuse optionnel.")
