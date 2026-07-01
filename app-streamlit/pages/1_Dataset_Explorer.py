"""app-streamlit/pages/1_Dataset_Explorer.py — Exercice 3.1, Page 2."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from PIL import Image

from core.data_processing import build_dataframe, class_distribution
from core.config import CLASS_NAMES
from core.streamlit_ui import apply_custom_style, render_sidebar, page_header

st.set_page_config(page_title="Exploration du dataset", page_icon=":material/folder_open:", layout="wide")
apply_custom_style()
render_sidebar()
page_header("Exploration du dataset", subtitle="Distribution des classes et exemples d'images", icon="folder_open")

with st.spinner("Chargement de l'inventaire des images..."):
    df = build_dataframe()

if df.empty:
    st.error("Aucune image trouvée dans data/raw/. Vérifiez le téléchargement du dataset (voir README).")
else:
    st.caption(f"{len(df)} images réparties sur {df['classe'].nunique()} classes.")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Distribution des classes")
        st.bar_chart(class_distribution(df), color="#1D9E75")
    with col_right:
        st.subheader("Statistiques de résolution")
        st.dataframe(df[["width", "height", "file_size_kb"]].describe(), width="stretch")

    st.divider()
    st.subheader("Exemples par classe")
    selected_class = st.selectbox("Classe", CLASS_NAMES)
    samples = df[df["classe"] == selected_class].sample(min(4, len(df[df["classe"] == selected_class])))
    cols = st.columns(len(samples)) if len(samples) > 0 else []
    for col, (_, row) in zip(cols, samples.iterrows()):
        with col:
            st.image(Image.open(row["filepath"]), width="stretch")
