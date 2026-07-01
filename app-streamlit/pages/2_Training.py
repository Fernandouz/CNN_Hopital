"""Page Streamlit de lancement des entrainements CNN."""

import atexit
import html
import os
import signal
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from core.config import BATCH_SIZE, IMG_SIZE, LEARNING_RATE, NUM_EPOCHS, PROJECT_ROOT
import core.model_utils as model_utils
from core.streamlit_ui import apply_custom_style, page_header, render_sidebar


SUPPORTED_ARCHITECTURES = getattr(
    model_utils,
    "SUPPORTED_ARCHITECTURES",
    ["custom_cnn", "vgg16", "resnet50", "efficientnet_b0", "mobilenet_v3_large"],
)

MODEL_LABELS = {
    "resnet50": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v3_large": "MobileNetV3-Large",
    "vgg16": "VGG16",
    "custom_cnn": "Custom CNN from scratch",
}
MLFLOW_PORT = 5001
MLFLOW_HOST = "127.0.0.1"
MLFLOW_URL = f"http://{MLFLOW_HOST}:{MLFLOW_PORT}"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_BACKEND_URI = f"sqlite:///{MLFLOW_DB_PATH}"
MLFLOW_ALLOWED_HOSTS = "*"
TRAINING_LOG_MAX_LINES = 45
TRAINING_LOG_HEIGHT = 300
CNN_EXPERIMENT_NAME = "wound-classification-app"
PROJECT_PYTHON = (
    PROJECT_ROOT / ".venv" / "bin" / "python"
    if (PROJECT_ROOT / ".venv" / "bin" / "python").exists()
    else Path(sys.executable)
)
if "_MLFLOW_PROCESSES" not in globals():
    _MLFLOW_PROCESSES = []
if "_MLFLOW_CLEANUP_REGISTERED" not in globals():
    _MLFLOW_CLEANUP_REGISTERED = False


st.set_page_config(
    page_title="Entrainement",
    page_icon=":material/model_training:",
    layout="wide",
)
apply_custom_style()
render_sidebar()
page_header(
    "Entrainement du modele",
    subtitle="Lancement de python -m scripts.train_cnn avec suivi MLflow",
    icon="model_training",
)

st.warning(
    "Un entrainement lance depuis Streamlit bloque la page jusqu'a la fin du run. "
    "Les checkpoints sont sauvegardes dans models/ et le run est trace dans MLflow."
)

st.markdown(
    f"""
    <style>
    .training-log-box {{
        max-height: {TRAINING_LOG_HEIGHT}px;
        overflow-y: auto;
        padding: 12px 14px;
        border: 1px solid #2D4A40;
        border-radius: 8px;
        background: #070B0A;
        color: #EAF6F1;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.82rem;
        line-height: 1.45;
        white-space: pre-wrap;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def build_training_command(params):
    """Construit la commande Python correspondant aux parametres UI."""
    command = [
        str(PROJECT_PYTHON),
        "-m",
        "scripts.train_cnn",
        "--architecture",
        params["architecture"],
        "--epochs",
        str(params["epochs"]),
        "--batch-size",
        str(params["batch_size"]),
        "--img-size",
        str(params["img_size"]),
        "--lr",
        str(params["learning_rate"]),
        "--weight-decay",
        str(params["weight_decay"]),
        "--dropout-rate",
        str(params["dropout_rate"]),
        "--patience",
        str(params["patience"]),
        "--num-workers",
        str(params["num_workers"]),
    ]

    if params["pretrained"]:
        command.append("--pretrained")
    if params["freeze_backbone"]:
        command.append("--freeze-backbone")
    if params["fine_tune"]:
        command.append("--fine-tune")
    if params["weighted_sampler"]:
        command.append("--weighted-sampler")
    if params["class_weights"]:
        command.append("--class-weights")
    if params["run_suffix"].strip():
        command.extend(["--run-suffix", params["run_suffix"].strip()])

    return command


def shell_preview(command):
    """Affiche une commande copiable compatible shell."""
    return " ".join(shlex.quote(part) for part in command)


def run_training(command):
    """Execute le script d'entrainement et affiche les logs dans Streamlit."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["MLFLOW_TRACKING_URI"] = MLFLOW_BACKEND_URI
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

    log_placeholder = st.empty()
    lines = []

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip())
        visible_logs = html.escape("\n".join(lines[-TRAINING_LOG_MAX_LINES:]))
        log_placeholder.markdown(
            f"<div class='training-log-box'>{visible_logs}</div>",
            unsafe_allow_html=True,
        )

    return_code = process.wait()
    return return_code, "\n".join(lines)


def terminate_process(process):
    """Termine un process lance par l'app, avec son groupe si disponible."""
    if process is None or process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError:
        process.terminate()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            process.kill()
        process.wait(timeout=5)


def cleanup_mlflow_processes():
    """Arrete les serveurs MLflow lances par cette app quand Streamlit quitte."""
    for process in list(_MLFLOW_PROCESSES):
        terminate_process(process)
    _MLFLOW_PROCESSES.clear()


def register_mlflow_cleanup():
    """Enregistre une seule fois le nettoyage automatique a la fermeture."""
    global _MLFLOW_CLEANUP_REGISTERED
    if not _MLFLOW_CLEANUP_REGISTERED:
        atexit.register(cleanup_mlflow_processes)
        _MLFLOW_CLEANUP_REGISTERED = True


def start_mlflow_ui():
    """Lance MLflow UI en arriere-plan si l'utilisateur le demande."""
    register_mlflow_cleanup()

    if st.session_state.get("mlflow_process") is not None:
        process = st.session_state["mlflow_process"]
        if process.poll() is None:
            return process

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

    log_path = PROJECT_ROOT / ".tmp" / "mlflow-ui.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")

    process = subprocess.Popen(
        [
            str(PROJECT_PYTHON),
            "-m",
            "mlflow",
            "ui",
            "--host",
            MLFLOW_HOST,
            "--backend-store-uri",
            MLFLOW_BACKEND_URI,
            "--port",
            str(MLFLOW_PORT),
            "--allowed-hosts",
            MLFLOW_ALLOWED_HOSTS,
            "--cors-allowed-origins",
            "*",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    _MLFLOW_PROCESSES.append(process)
    st.session_state["mlflow_process"] = process
    st.session_state["mlflow_log_path"] = str(log_path)
    return process


def stop_mlflow_ui():
    """Arrete l'instance MLflow UI lancee par cette session Streamlit."""
    process = st.session_state.get("mlflow_process")
    terminate_process(process)
    if process in _MLFLOW_PROCESSES:
        _MLFLOW_PROCESSES.remove(process)
    st.session_state["mlflow_process"] = None


def render_mlflow_access():
    """Affiche les actions pour consulter MLflow depuis Streamlit."""
    with st.container(border=True):
        st.subheader("MLflow")
        st.caption(
            "Les runs d'entrainement sont traces dans `mlflow.db`. "
            "Ouvrez l'interface locale pour suivre les métriques, paramètres et artefacts."
        )

        if "mlflow_log_path" in st.session_state:
            st.caption(f"Logs MLflow UI : `{st.session_state['mlflow_log_path']}`")

        experiments = load_mlflow_experiments_summary()
        if experiments.empty:
            st.warning(
                "Aucune expérience MLflow détectée dans la base configurée. "
                "Vérifiez qu'un run a bien démarré et que `mlflow.db` existe."
            )
        else:
            cnn_experiment = experiments[
                experiments["nom"] == CNN_EXPERIMENT_NAME
            ]
            if not cnn_experiment.empty:
                experiment_id = str(cnn_experiment.iloc[0]["id"])
                st.link_button(
                    "Ouvrir l'expérience CNN",
                    f"{MLFLOW_URL}/#/experiments/{experiment_id}",
                )

            st.caption("Expériences détectées dans la même base MLflow :")
            st.dataframe(experiments, width="stretch", hide_index=True)


def load_mlflow_experiments_summary():
    """Lit les expériences depuis le même backend que l'UI MLflow."""
    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_BACKEND_URI)
        client = mlflow.tracking.MlflowClient()
        rows = []

        for experiment in client.search_experiments(view_type=3):
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=10000,
            )
            rows.append(
                {
                    "id": experiment.experiment_id,
                    "nom": experiment.name,
                    "statut": experiment.lifecycle_stage,
                    "runs": len(runs),
                    "artefacts": experiment.artifact_location,
                }
            )

        return pd.DataFrame(rows).sort_values(
            by=["runs", "nom"],
            ascending=[False, True],
        )
    except Exception as exc:
        st.caption(f"Résumé MLflow indisponible : {exc}")
        return pd.DataFrame()


with st.container(border=True):
    st.subheader("Configuration du run")

    col1, col2, col3 = st.columns(3)
    with col1:
        architecture = st.selectbox(
            "Architecture",
            options=SUPPORTED_ARCHITECTURES,
            index=SUPPORTED_ARCHITECTURES.index("resnet50"),
            format_func=lambda value: MODEL_LABELS.get(value, value),
        )
    with col2:
        epochs = st.number_input(
            "Epochs",
            min_value=1,
            max_value=300,
            value=NUM_EPOCHS,
            step=1,
        )
    with col3:
        patience = st.number_input(
            "Patience early stopping",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=128,
            value=BATCH_SIZE,
            step=1,
        )
    with col2:
        img_size = st.number_input(
            "Taille image",
            min_value=64,
            max_value=512,
            value=IMG_SIZE,
            step=32,
        )
    with col3:
        num_workers = st.number_input(
            "Num workers",
            min_value=0,
            max_value=8,
            value=0,
            step=1,
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        learning_rate = st.select_slider(
            "Learning rate",
            options=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
            value=LEARNING_RATE,
            format_func=lambda value: f"{value:.0e}",
        )
    with col2:
        weight_decay = st.select_slider(
            "Weight decay",
            options=[0.0, 1e-6, 1e-5, 1e-4, 1e-3],
            value=1e-4,
            format_func=lambda value: f"{value:.0e}" if value else "0",
        )
    with col3:
        dropout_rate = st.slider(
            "Dropout",
            min_value=0.0,
            max_value=0.8,
            value=0.3,
            step=0.05,
        )

    custom_cnn_selected = architecture == "custom_cnn"

    col1, col2, col3 = st.columns(3)
    with col1:
        pretrained = st.checkbox(
            "Poids ImageNet",
            value=not custom_cnn_selected,
            disabled=custom_cnn_selected,
        )
    with col2:
        freeze_backbone = st.checkbox(
            "Geler le backbone",
            value=not custom_cnn_selected,
            disabled=custom_cnn_selected,
        )
    with col3:
        fine_tune = st.checkbox(
            "Fine-tuning dernier bloc",
            value=False,
            disabled=custom_cnn_selected,
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        weighted_sampler = st.checkbox("WeightedRandomSampler", value=True)
    with col2:
        class_weights = st.checkbox("Class weights dans la loss", value=False)
    with col3:
        run_suffix = st.text_input(
            "Suffixe du run",
            value="streamlit_config",
            help="Ajoute un suffixe au nom du run MLflow et aux fichiers sauvegardes.",
        )

if architecture == "custom_cnn":
    pretrained = False
    freeze_backbone = False
    fine_tune = False
    st.info(
        "Custom CNN est entraine from scratch : poids ImageNet, freeze backbone "
        "et fine-tuning sont desactives."
    )

params = {
    "architecture": architecture,
    "epochs": int(epochs),
    "batch_size": int(batch_size),
    "img_size": int(img_size),
    "learning_rate": float(learning_rate),
    "weight_decay": float(weight_decay),
    "dropout_rate": float(dropout_rate),
    "pretrained": bool(pretrained),
    "freeze_backbone": bool(freeze_backbone),
    "fine_tune": bool(fine_tune),
    "weighted_sampler": bool(weighted_sampler),
    "class_weights": bool(class_weights),
    "patience": int(patience),
    "num_workers": int(num_workers),
    "run_suffix": run_suffix,
}

command = build_training_command(params)

st.subheader("Commande executee")
st.code(shell_preview(command), language="bash")

start_training = st.button(
    "Lancer l'entrainement",
    type="primary",
    disabled=st.session_state.get("training_running", False),
)

if start_training:
    st.session_state["training_running"] = True
    st.session_state["last_training_command"] = shell_preview(command)

    with st.status("Entrainement en cours...", expanded=True) as status:
        mlflow_process = start_mlflow_ui()
        if mlflow_process.poll() is None:
            st.info(f"MLflow UI démarre sur {MLFLOW_URL}.")
        else:
            st.warning(
                "MLflow UI n'a pas pu démarrer automatiquement. "
                "Consultez `.tmp/mlflow-ui.log`."
            )
        return_code, logs = run_training(command)
        st.session_state["last_training_logs"] = logs
        st.session_state["last_training_return_code"] = return_code

        if return_code == 0:
            status.update(label="Entrainement termine", state="complete")
            st.success("Run termine avec succes. Consultez models/ et MLflow.")
            experiments = load_mlflow_experiments_summary()
            cnn_experiment = experiments[
                experiments["nom"] == CNN_EXPERIMENT_NAME
            ]
            if not cnn_experiment.empty:
                experiment_id = str(cnn_experiment.iloc[0]["id"])
                st.link_button(
                    "Ouvrir l'expérience CNN dans MLflow",
                    f"{MLFLOW_URL}/#/experiments/{experiment_id}",
                )
            else:
                st.link_button("Ouvrir MLflow", MLFLOW_URL)
        else:
            status.update(label="Entrainement termine avec erreur", state="error")
            st.error(f"Le script a retourne le code {return_code}.")

    st.session_state["training_running"] = False

if "last_training_return_code" in st.session_state:
    st.divider()
    st.subheader("Dernier run lance depuis l'app")
    st.caption(f"Code retour : {st.session_state['last_training_return_code']}")
    st.code(st.session_state.get("last_training_command", ""), language="bash")

st.divider()
render_mlflow_access()

st.divider()
st.caption(
    "Commande equivalente : "
    f"`mlflow ui --host 127.0.0.1 --backend-store-uri {MLFLOW_BACKEND_URI} "
    "--port 5001 --allowed-hosts '*' --cors-allowed-origins '*'`"
)
