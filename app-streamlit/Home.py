"""app-streamlit/Home.py — Exercice 3.1, Page 1."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from core.config import CLASS_NAMES, SPLIT_DIR
from core.streamlit_ui import apply_custom_style, render_sidebar, page_header

st.set_page_config(page_title="Plateforme d'analyse de plaies", page_icon=":material/water_drop:", layout="wide")
apply_custom_style()
render_sidebar()

page_header(
    "Plateforme d'analyse d'imagerie médicale des plaies",
    subtitle="Classification CNN · Détection hors-domaine · Recherche par similarité",
    icon="water_drop",
)

st.warning("Outil d'aide au diagnostic à visée pédagogique — ne remplace pas l'avis d'un professionnel de santé.")

st.subheader("Statistiques clés")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Classes de plaies", len(CLASS_NAMES))
with col2:
    try:
        test_df = pd.read_csv(SPLIT_DIR / "test.csv")
        train_df = pd.read_csv(SPLIT_DIR / "train.csv")
        val_df = pd.read_csv(SPLIT_DIR / "val.csv")
        st.metric("Images (train+val+test)", len(train_df) + len(val_df) + len(test_df))
    except FileNotFoundError:
        st.metric("Images", "—")
with col3:
    st.metric("Accuracy (test, meilleur modèle)", "89.2 %", help="ResNet50 fine-tuné — macro-F1 test = 0.890, 7 erreurs/65. Détail dans MLflow.")

st.divider()
st.subheader("Navigation")

nav_items = [
    ("folder_open", "Explorer le dataset", "Distribution des classes, exemples d'images", "pages/1_Dataset_Explorer.py"),
    ("search", "Faire une prédiction", "Upload une image et obtenez un diagnostic", "pages/3_Prediction.py"),
    ("model_training", "Entraînement", "Configuration et suivi des modèles", "pages/2_Training.py"),
]

cols = st.columns(3)
for col, (icon, title, desc, target) in zip(cols, nav_items):
    with col:
        with st.container(border=True):
            st.markdown(f":material/{icon}: **{title}**")
            st.caption(desc)
            st.page_link(target, label="Ouvrir")

st.divider()
st.subheader("Classes reconnues")
st.write(" · ".join(CLASS_NAMES))
