"""Composants visuels partages par l'application Streamlit."""

import streamlit as st


BG = "#0E1614"
SURFACE = "#142420"
SIDEBAR_BG = "#070B0A"
BORDER = "#2D4A40"
ACCENT = "#1D9E75"
ACCENT_DARK = "#16805F"
TEXT = "#EAF6F1"
TEXT_MUTED = "#8FB3A6"


def apply_custom_style():
    """Applique le theme sombre clinique de l'application."""
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        .stApp {{
            background: {BG};
        }}
        .block-container {{
            padding-top: 2.2rem;
            max-width: 1120px;
        }}
        h1, h2, h3 {{
            color: {TEXT};
            font-weight: 600;
            letter-spacing: 0;
        }}
        .subtitle {{
            color: {TEXT_MUTED};
            font-size: 0.95rem;
            margin-top: -0.5rem;
            margin-bottom: 0.5rem;
        }}
        hr {{
            border-color: {BORDER};
            opacity: 1;
            margin: 1.6rem 0;
        }}
        .stButton > button, .stDownloadButton > button {{
            background-color: {ACCENT};
            color: #06231A;
            border: none;
            border-radius: 8px;
            padding: 0.55rem 1.1rem;
            font-weight: 600;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background-color: {ACCENT_DARK};
            color: #06231A;
        }}
        [data-testid="stFileUploaderDropzone"],
        div[data-testid="stVerticalBlockBorderWrapper"],
        div[data-testid="stMetric"],
        div[data-testid="metric-container"] {{
            background-color: {SURFACE} !important;
            border-color: {BORDER} !important;
            border-radius: 8px !important;
        }}
        div[data-testid="stMetricValue"] * {{
            color: {ACCENT} !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {SIDEBAR_BG} !important;
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] * {{
            color: {TEXT};
        }}
        [data-testid="stSidebarNav"] a {{
            border-radius: 8px;
        }}
        [data-testid="stSidebarNav"] a:hover,
        [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background-color: {SURFACE} !important;
        }}
        div[data-testid="stAlert"] {{
            border-radius: 8px;
        }}
        .main, .main * {{
            color: {TEXT};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Affiche la barre laterale commune."""
    with st.sidebar:
        st.markdown(":material/water_drop: **Wound AI Platform**")
        st.caption("Projet pedagogique d'analyse de plaies")
        st.divider()
        st.caption(
            "Cette application est realisee dans un cadre pedagogique. "
            "Elle ne constitue pas un dispositif medical et ne remplace pas "
            "l'avis d'un professionnel de sante."
        )
        st.divider()
        st.caption(":material/model_training: ResNet50 / EfficientNet-B0")
        st.caption(":material/category: 7 classes de plaies cutanees")


def page_header(title: str, subtitle: str = "", icon: str = ""):
    """Rend un titre de page homogene."""
    label = f":material/{icon}: {title}" if icon else title
    st.markdown(f"# {label}")
    if subtitle:
        st.markdown(f"<p class='subtitle'>{subtitle}</p>", unsafe_allow_html=True)
    st.divider()
