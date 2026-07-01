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
        .metric-card {{
            background-color: {SURFACE};
            border: 1px solid {BORDER};
            border-radius: 8px;
            padding: 1rem 1.1rem;
            min-height: 104px;
        }}
        .metric-card-label {{
            color: {TEXT_MUTED};
            font-size: 0.86rem;
            line-height: 1.25;
            margin-bottom: 0.45rem;
        }}
        .metric-card-value {{
            color: {ACCENT};
            font-size: 1.9rem;
            font-weight: 700;
            line-height: 1.15;
        }}
        .metric-card-help {{
            color: {TEXT_MUTED};
            font-size: 0.78rem;
            line-height: 1.3;
            margin-top: 0.45rem;
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
        div[data-testid="stMetric"],
        div[data-testid="metric-container"] {{
            padding: 0.9rem 1rem !important;
        }}
        div[data-testid="stMetricValue"] * {{
            color: {ACCENT} !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {SIDEBAR_BG} !important;
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] > div {{
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        section[data-testid="stSidebar"] * {{
            color: {TEXT};
        }}
        .sidebar-brand {{
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }}
        .sidebar-brand-subtitle {{
            color: {TEXT_MUTED};
            font-size: 0.82rem;
            line-height: 1.35;
        }}
        .sidebar-footer {{
            margin-top: auto;
            padding-top: 1rem;
            color: {TEXT_MUTED};
            font-size: 0.78rem;
            line-height: 1.35;
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
        st.markdown(
            """
            <div class="sidebar-brand">Wound AI Platform</div>
            <div class="sidebar-brand-subtitle">Projet pédagogique d'analyse de plaies</div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption(":material/model_training: ResNet50 / EfficientNet-B0")
        st.caption(":material/category: 7 classes de plaies cutanees")


def page_header(title: str, subtitle: str = "", icon: str = ""):
    """Rend un titre de page homogene."""
    label = f":material/{icon}: {title}" if icon else title
    st.markdown(f"# {label}")
    if subtitle:
        st.markdown(
            f"<p class='subtitle'>{subtitle}</p>", unsafe_allow_html=True)
    st.divider()


def metric_card(label: str, value, help_text: str = ""):
    """Affiche une métrique dans un encart stable avec padding."""
    help_html = f"<div class='metric-card-help'>{help_text}</div>" if help_text else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-card-label">{label}</div>
            <div class="metric-card-value">{value}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
