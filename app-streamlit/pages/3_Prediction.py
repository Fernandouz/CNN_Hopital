"""app-streamlit/pages/3_Prediction.py — Exercice 3.1, Page 4."""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from PIL import Image

from core.config import AVAILABLE_CHECKPOINTS, DEFAULT_ARCHITECTURE
from core.grad_cam import GradCAMExplainer
from core.predict_pipeline import load_pipeline_resources, run_full_pipeline
from core.streamlit_ui import apply_custom_style, render_sidebar, page_header


os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


st.set_page_config(page_title="Prédiction",
                   page_icon=":material/search:", layout="wide")
apply_custom_style()
render_sidebar()
page_header("Prédiction & analyse",
            subtitle="Upload une image pour obtenir un diagnostic", icon="search")

st.warning("Outil d'aide au diagnostic à visée pédagogique uniquement. Ne remplace pas l'avis d'un professionnel de santé.")

st.markdown(
    """
    <style>
    div[data-testid="stMetric"],
    div[data-testid="metric-container"] {
        padding: 22px 24px !important;
        min-height: 116px;
    }
    div[data-testid="stMetricLabel"] * {
        font-size: 0.9rem !important;
    }
    div[data-testid="stMetricValue"] * {
        line-height: 1.15 !important;
        overflow-wrap: anywhere;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ARCHITECTURE_LABELS = {
    "resnet50": "ResNet50 fine-tuné (recommandé, meilleur macro-F1)",
    "efficientnet_b0": "EfficientNet-B0 (léger et performant)",
    "mobilenet_v3_large": "MobileNetV3-Large (léger, alternative)",
}

available_architectures = [
    architecture
    for architecture, checkpoint_path in AVAILABLE_CHECKPOINTS.items()
    if checkpoint_path.exists() and architecture in ARCHITECTURE_LABELS
]

if not available_architectures:
    st.error("Aucun checkpoint de classification disponible dans models/.")
    st.stop()

selected_label = st.selectbox(
    "Modèle de classification",
    options=[ARCHITECTURE_LABELS[a] for a in available_architectures],
    index=available_architectures.index(DEFAULT_ARCHITECTURE)
    if DEFAULT_ARCHITECTURE in available_architectures
    else 0,
)
selected_architecture = [
    k for k, v in ARCHITECTURE_LABELS.items() if v == selected_label][0]


@st.cache_resource
def get_resources(architecture: str):
    with st.spinner(f"Chargement du modèle {architecture} (première utilisation seulement)..."):
        return load_pipeline_resources(architecture=architecture)


@st.cache_resource
def get_grad_cam_explainer(checkpoint_path: str):
    return GradCAMExplainer(checkpoint_path=checkpoint_path)


if "prediction_page_state" not in st.session_state:
    st.session_state["prediction_page_state"] = None


def compute_grad_cam_overlay(resources, image_path, classification):
    """Genere l'overlay Grad-CAM pour la classe predite."""
    explainer = get_grad_cam_explainer(str(resources["checkpoint_path"]))
    grad_cam_result = explainer.explain_image(
        image_path=str(image_path),
        target_class=classification["classe_predite"],
        top_k=3,
        alpha=0.45,
    )
    return {
        "overlay_rgb": grad_cam_result.overlay_rgb,
        "target_class": grad_cam_result.target_class,
    }


def render_prediction_panel(classification, ood):
    """Affiche le diagnostic avec des composants Streamlit natifs."""
    with st.container(border=True, height=620):
        st.success(
            "Image valide - "
            f"Score d'anomalie {ood['score']:.4f} "
            f"(seuil {ood['threshold']:.4f})"
        )
        st.subheader("Diagnostic")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Classe prédite", classification["classe_predite"])
        with c2:
            st.metric(
                "Confiance", f"{classification['confiance'] * 100:.1f} %")

        st.subheader("Top-3 prédictions")
        for pred in classification["top3"]:
            st.progress(
                pred["score"],
                text=f"{pred['classe']} - {pred['score'] * 100:.1f} %",
            )

        st.caption(
            "La heatmap Grad-CAM affichée "
            "à gauche indique les zones ayant le plus contribué à cette prédiction."
        )


uploaded_file = st.file_uploader("Image de plaie", type=["jpg", "jpeg", "png"])

if uploaded_file:
    tmp_path = Path("data/processed/_tmp_upload.jpg")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(uploaded_file).convert("RGB")
    image.save(tmp_path)

    with st.spinner("Analyse en cours..."):
        resources = get_resources(selected_architecture)
        result = run_full_pipeline(resources, str(tmp_path), k_similar=5)

    grad_cam = None
    if not result["ood"]["is_ood"] and result["classification"] is not None:
        try:
            with st.spinner("Génération Grad-CAM..."):
                grad_cam = compute_grad_cam_overlay(
                    resources=resources,
                    image_path=tmp_path,
                    classification=result["classification"],
                )
        except ImportError:
            grad_cam = {
                "error": "Grad-CAM indisponible : installez la dépendance `grad-cam`."
            }
        except Exception as exc:
            grad_cam = {"error": f"Grad-CAM non généré pour cette image : {exc}"}

    st.session_state["prediction_page_state"] = {
        "image": image.copy(),
        "result": result,
        "grad_cam": grad_cam,
        "architecture": selected_architecture,
    }

page_state = st.session_state.get("prediction_page_state")

if page_state is not None:
    image = page_state["image"]
    result = page_state["result"]
    grad_cam = page_state.get("grad_cam")
    previous_architecture = page_state.get("architecture", selected_architecture)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption="Image uploadée", width="stretch")

        if grad_cam:
            st.caption("Overlay Grad-CAM")
            if "error" in grad_cam:
                st.warning(grad_cam["error"])
            else:
                st.image(
                    grad_cam["overlay_rgb"],
                    caption=f"Heatmap pour la classe {grad_cam['target_class']}",
                    width="stretch",
                )

    with col2:
        st.caption(f"Résultat conservé pour le modèle : {previous_architecture}")
        ood = result["ood"]
        if ood["is_ood"]:
            st.error(
                f"Image non reconnue par le système.\n\n"
                f"Type de plaie inconnu ou image hors domaine "
                f"(score d'anomalie = {ood['score']:.1f}, seuil = {ood['threshold']:.1f})."
            )
            st.caption(
                "Cette image ne ressemble pas suffisamment aux types de plaies connus "
                "par le système. Vérifiez qu'il s'agit bien d'une photo de plaie cutanée, "
                "ou consultez directement un professionnel de santé."
            )
        elif not ood.get("enabled", True):
            st.warning(ood["decision"])
        else:
            render_prediction_panel(
                classification=result["classification"],
                ood=ood,
            )

if page_state and not page_state["result"]["ood"]["is_ood"]:
    st.divider()
    st.subheader("Cas historiques similaires")
    similar_cases = page_state["result"]["similar_cases"]
    if not similar_cases:
        st.info("Aucun cas similaire disponible pour cette image.")
        st.stop()

    cols = st.columns(len(similar_cases))
    for col, case in zip(cols, similar_cases):
        with col:
            try:
                st.image(Image.open(case["filepath"]), width="stretch")
            except Exception:
                st.caption("Image indisponible")
            st.caption(
                f"{case['classe']} — similarité {case['similarite']:.2f}")
