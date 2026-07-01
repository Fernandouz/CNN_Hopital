"""Page Streamlit d'exploration du dataset."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from core.data_processing import build_dataframe
from core.config import CLASS_NAMES, DATA_RAW_DIR, IMG_SIZE, SEED
from core.streamlit_ui import apply_custom_style, render_sidebar, page_header
from core.ui_metrics import metric_card


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@st.cache_data(show_spinner=False)
def find_invalid_images(raw_dir: Path) -> pd.DataFrame:
    """Retourne les fichiers image illisibles, comme dans le notebook EDA."""
    records = []
    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        return pd.DataFrame(columns=["path", "class", "error"])

    class_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    for class_dir in class_dirs:
        for file_path in class_dir.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                with Image.open(file_path) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError) as exc:
                records.append(
                    {
                        "path": str(file_path),
                        "class": class_dir.name,
                        "error": str(exc),
                    }
                )

    return pd.DataFrame(records)


def build_class_distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    """Construit le tableau count/percentage du notebook."""
    distribution = (
        df["class"]
        .value_counts()
        .rename_axis("classe")
        .reset_index(name="nombre_images")
    )
    distribution["pourcentage"] = (
        distribution["nombre_images"] / distribution["nombre_images"].sum() * 100
    ).round(2)
    return distribution


def imbalance_message(imbalance_ratio: float) -> str:
    """Produit le diagnostic textuel sur le déséquilibre de classes."""
    if imbalance_ratio < 1.5:
        return "Le dataset peut être considéré comme globalement équilibré."
    if imbalance_ratio < 3:
        return "Le dataset présente un déséquilibre modéré entre les classes."
    return "Le dataset présente un déséquilibre important entre les classes."


def build_image_format_table(df: pd.DataFrame) -> pd.DataFrame:
    """Regroupe les images par format utile au prétraitement CNN."""
    formats = df.copy()
    formats["format"] = "carré"
    formats.loc[formats["width"] > formats["height"], "format"] = "paysage"
    formats.loc[formats["width"] < formats["height"], "format"] = "portrait"
    formats["pixels"] = formats["width"] * formats["height"]
    formats["dimension_min"] = formats[["width", "height"]].min(axis=1)
    formats["sous_224px"] = formats["dimension_min"] < IMG_SIZE

    table = (
        formats.groupby("format")
        .agg(
            nombre_images=("path", "count"),
            largeur_mediane=("width", "median"),
            hauteur_mediane=("height", "median"),
            pixels_medians=("pixels", "median"),
            images_sous_224px=("sous_224px", "sum"),
        )
        .reset_index()
    )
    table["pourcentage"] = (table["nombre_images"] / len(df) * 100).round(2)
    integer_columns = [
        "largeur_mediane",
        "hauteur_mediane",
        "pixels_medians",
        "images_sous_224px",
    ]
    table[integer_columns] = table[integer_columns].round(0).astype(int)
    return table


def build_stratified_split_preview(df: pd.DataFrame):
    """Reproduit le split 70/15/15 du notebook sans écrire de CSV."""
    if df.empty or df["class"].value_counts().min() < 2:
        return None, None

    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            stratify=df["class"],
            random_state=SEED,
        )
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            stratify=temp_df["class"],
            random_state=SEED,
        )
    except ValueError:
        return None, None

    split_distribution = pd.DataFrame(
        {
            "train": train_df["class"].value_counts(),
            "validation": val_df["class"].value_counts(),
            "test": test_df["class"].value_counts(),
        }
    ).fillna(0).astype(int)
    split_distribution["total"] = split_distribution.sum(axis=1)

    split_percentages = (
        split_distribution[["train", "validation", "test"]]
        .div(split_distribution["total"], axis=0)
        * 100
    ).round(2)

    return split_distribution.sort_index(), split_percentages.sort_index()


st.set_page_config(
    page_title="Exploration du dataset",
    page_icon=":material/folder_open:",
    layout="wide",
)
apply_custom_style()
render_sidebar()
page_header(
    "Exploration du dataset",
    subtitle="Statistiques, déséquilibre de classes, résolutions et exemples d'images",
    icon="folder_open",
)

with st.spinner("Chargement de l'inventaire des images..."):
    df = build_dataframe()
    df_invalid = find_invalid_images(DATA_RAW_DIR)

if df.empty:
    st.error("Aucune image trouvée dans data/raw/. Vérifiez le téléchargement du dataset (voir README).")
else:
    distribution = build_class_distribution_table(df)
    image_format_table = build_image_format_table(df)
    min_count = int(distribution["nombre_images"].min())
    max_count = int(distribution["nombre_images"].max())
    imbalance_ratio = round(max_count / min_count, 2) if min_count else 0
    summary = {
        "total_images": len(df),
        "num_classes": df["class"].nunique(),
        "min_images_per_class": min_count,
        "max_images_per_class": max_count,
        "imbalance_ratio": imbalance_ratio,
        "median_width": int(df["width"].median()),
        "median_height": int(df["height"].median()),
        "num_invalid_files": len(df_invalid),
    }

    st.caption(f"Dataset source : `{DATA_RAW_DIR}`")

    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card("Images valides", summary["total_images"])
    with metric_cols[1]:
        metric_card("Classes", summary["num_classes"])
    with metric_cols[2]:
        metric_card("Ratio max/min", summary["imbalance_ratio"])
    with metric_cols[3]:
        metric_card("Fichiers illisibles", summary["num_invalid_files"])

    st.markdown(
        f"""
        <div style="
            margin-top:1rem;
            background-color:#142420;
            border:1px solid #1D9E75;
            border-left:5px solid #1D9E75;
            border-radius:8px;
            padding:0.9rem 1rem;
            color:#EAF6F1;
            line-height:1.5;
        ">
            {imbalance_message(imbalance_ratio)} Le nombre d'images par classe varie
            de {min_count} à {max_count}. Les images ont une résolution médiane de
            {summary['median_width']} x {summary['median_height']} pixels et seront
            redimensionnées en 224 x 224 pour les CNN pré-entraînés.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Distribution des classes")
        st.bar_chart(
            distribution,
            x="classe",
            y="nombre_images",
            color="#1D9E75",
            height=360,
        )
        st.dataframe(distribution, width="stretch", hide_index=True)
    with col_right:
        st.subheader("Formats avant redimensionnement")
        st.bar_chart(
            image_format_table,
            x="format",
            y="nombre_images",
            color="#1D9E75",
            height=360,
        )
        st.dataframe(image_format_table, width="stretch", hide_index=True)
        st.caption(
            f"Cette vue indique si le redimensionnement en {IMG_SIZE} x {IMG_SIZE} "
            "risque surtout de compresser des images portrait, paysage ou carrées."
        )

    st.divider()
    st.subheader("Exemples par classe")
    available_classes = sorted(df["class"].unique())
    default_classes = [class_name for class_name in CLASS_NAMES if class_name in available_classes]
    selected_class = st.selectbox("Classe", default_classes or available_classes)
    samples = df[df["classe"] == selected_class].sample(
        min(4, len(df[df["classe"] == selected_class])),
        random_state=SEED,
    )
    cols = st.columns(len(samples)) if len(samples) > 0 else []
    for col, (_, row) in zip(cols, samples.iterrows()):
        with col:
            st.image(Image.open(row["filepath"]), width="stretch")
            st.caption(f"{row['filename']} - {row['width']} x {row['height']}")

    st.divider()
    st.subheader("Aperçu du split stratifié 70 / 15 / 15")
    split_distribution, split_percentages = build_stratified_split_preview(df)
    if split_distribution is None:
        st.warning(
            "Le split stratifié ne peut pas être calculé avec la distribution actuelle "
            "des classes. Vérifiez qu'il y a assez d'images par classe."
        )
    else:
        split_cols = st.columns([1, 1])
        with split_cols[0]:
            st.write("Nombre d'images par classe")
            st.dataframe(split_distribution, width="stretch")
        with split_cols[1]:
            st.write("Pourcentages par classe")
            st.dataframe(split_percentages, width="stretch")

        st.caption(
            "Cet aperçu reprend la logique du notebook. Les CSV réels de split sont "
            "générés par le pipeline de préparation afin de rester reproductibles."
        )
