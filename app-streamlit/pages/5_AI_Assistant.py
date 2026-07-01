"""app-streamlit/pages/5_AI_Assistant.py — Partie 5, optionnelle."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from core.streamlit_ui import apply_custom_style, render_sidebar, page_header

st.set_page_config(page_title="Assistant IA", page_icon=":material/smart_toy:", layout="wide")
apply_custom_style()
render_sidebar()
page_header("Assistant IA — Recommandations de traitement", subtitle="Pipeline RAG (Partie 5, optionnelle)", icon="smart_toy")

st.warning("Ces recommandations sont fournies à titre indicatif et pédagogique uniquement. Elles ne remplacent pas l'avis d'un professionnel de santé.")
st.info(
    "Page liée à la Partie 5 (optionnelle, non encore implémentée). Nécessite : "
    "base de connaissances indexée, Ollama lancé localement, et le pipeline RAG complet."
)

if "historique" not in st.session_state:
    st.session_state["historique"] = []

st.subheader("Historique des consultations")
if not st.session_state["historique"]:
    st.caption("Aucune consultation pour le moment.")
else:
    for entry in st.session_state["historique"]:
        st.markdown(f"**{entry['classe']}** ({entry['confiance']}%) — {entry['recommandation'][:200]}...")
