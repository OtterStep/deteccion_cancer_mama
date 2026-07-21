import json
from pathlib import Path

import streamlit as st
import pandas as pd

from config import (
    DATA_DIR, MODELS_DIR, TABULAR_DIR, RESULTS_DIR, RESULTS_FIGURES_DIR, JSON_RESULTS_PATH,
)
from modules.i18n import get_translation


lang = st.session_state.get("language", "es")

MODEL_CATEGORIES = [
    ("CNN (EfficientNet)", ["cnn_efficientnet_", "best_cnn_efficientnet"], [".keras", ".h5"]),
    ("Ensemble (CNN+Clínico)", ["ensemble_"], [".keras"]),
    ("Hybrid CNN-RF Extractor", ["extractor_hibrid_rf_cnn", "hybrid_cnn_rf_extractor_"], [".keras"]),
    ("Hybrid CNN-RF Classifier", ["classifier_hibrid_rf_cnn", "hybrid_cnn_rf_classifier_"], [".pkl"]),
    ("Tabular (RF)", ["tabular_rf_", "rf_tabular_"], [".pkl"]),
    ("Tabular (XGBoost)", ["tabular_xgboost_"], [".pkl"]),
]


def classify_model(filename: str):
    for label, prefixes, extensions in MODEL_CATEGORIES:
        ext = Path(filename).suffix.lower()
        if ext not in extensions:
            continue
        for prefix in prefixes:
            if filename.startswith(prefix):
                return label
    return "Otro" if lang == "es" else "Other"


def get_model_type(filename: str) -> str:
    label = classify_model(filename)
    if "CNN" in label and "Extractor" in label:
        return "Hybrid"
    if "CNN" in label:
        return "CNN"
    if "Ensemble" in label:
        return "Ensemble"
    if "Hybrid" in label or "Classifier" in label:
        return "Hybrid"
    if "XGBoost" in label:
        return "Tabular"
    if "RF" in label or "Random" in label:
        return "Tabular"
    return "Unknown"


def load_json_results():
    if JSON_RESULTS_PATH.exists():
        with open(JSON_RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


st.title(get_translation(lang, "dashboard.title"))
st.markdown(get_translation(lang, "dashboard.welcome"))

st.info(
    f"**{get_translation(lang, 'dashboard.base_path')}:** `{DATA_DIR.parent}`  \n"
    "✅ Datos locales"
)

# ── Métricas principales ────────────────────────────────
data = load_json_results()

col1, col2, col3, col4 = st.columns(4)

csv_files = list(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
total_csv = len(csv_files)
col1.metric(get_translation(lang, "dashboard.csv_files"), total_csv)

model_files = []
if MODELS_DIR.exists():
    for f in MODELS_DIR.glob("*"):
        if f.suffix in (".keras", ".h5", ".pkl"):
            model_files.append(f)

model_types = {}
for f in model_files:
    t = get_model_type(f.name)
    model_types.setdefault(t, set()).add(f.name)

trained = data.get("fase2_modelado", {}).get("modelos_entrenados", [])
trained_count = len(trained) if trained else 0
unique_types = len(model_types)
total_models = len(model_files)
label = (
    f"{trained_count} entrenados ({total_models} descargados, {unique_types} tipos)"
    if trained_count
    else f"{total_models} ({unique_types} {get_translation(lang, 'dashboard.total_models_sub')})"
)
col2.metric(
    get_translation(lang, "dashboard.total_models"),
    label,
)

if data:
    wis = data.get("fase1_eda", {}).get("distribucion_wisconsin", {})
    total_wis = wis.get("B", 0) + wis.get("M", 0)
    malignant = wis.get("M", 0)
    benign = wis.get("B", 0)
    col3.metric(get_translation(lang, "dashboard.wisconsin_cases"), total_wis)
    col4.metric(get_translation(lang, "dashboard.malignant_benign"), f"{malignant} / {benign}")

    img_idx = data.get("fase1_eda", {}).get("image_index", {})
    if img_idx:
        st.info(
            f"**Índice de imágenes CBIS-DDSM:** {img_idx.get('total_imagenes_indexadas', 0)} imágenes totales "
            f"({img_idx.get('malignas', 0)} malignas, {img_idx.get('benignas', 0)} benignas) — "
            f"{img_idx.get('calcificaciones', 0)} calcificaciones, {img_idx.get('masas', 0)} masas."
        )

    mejor = data.get("ranking_modelos", {}).get("mejor_modelo", "")
    if mejor:
        tabla = data.get("ranking_modelos", {}).get("tabla", [])
        if tabla:
            modelo_map = {
                "tabular_xgboost": "XGBoost",
                "tabular_rf": "Random Forest",
                "cnn_efficientnet": "CNN",
                "hybrid_cnn_rf": "CNN+RF",
                "hybrid_cnn_xgb": "CNN+XGBoost",
            }
            modelo_nombre = modelo_map.get(mejor, mejor)
            score = tabla[0].get("Score compuesto", 0)
            auc = tabla[0].get("auc", 0)
            acc = tabla[0].get("accuracy", 0)
            st.success(
                f"Mejor modelo: {modelo_nombre} - "
                f"Score Compuesto: {score:.4f} | AUC: {auc:.4f} | Accuracy: {acc:.4f}"
            )
else:
    wis_path = TABULAR_DIR / "data.csv"
    if wis_path.exists():
        df = pd.read_csv(wis_path)
        total_cases = len(df)
        malignant = int((df["diagnosis"] == "M").sum()) if "diagnosis" in df.columns else 0
        benign = total_cases - malignant
        col3.metric(get_translation(lang, "dashboard.wisconsin_cases"), total_cases)
        col4.metric(get_translation(lang, "dashboard.malignant_benign"), f"{malignant} / {benign}")
    else:
        col3.metric(get_translation(lang, "dashboard.wisconsin_cases"), get_translation(lang, "dashboard.no_data"))
        col4.metric(get_translation(lang, "dashboard.malignant_benign"), get_translation(lang, "dashboard.no_data"))

# ── Archivos CSV ────────────────────────────────────────
st.markdown("---")
st.subheader(get_translation(lang, "dashboard.data_files"))

if DATA_DIR.exists():
    data_rows = []
    for f in sorted(csv_files):
        size_kb = f.stat().st_size / 1024
        data_rows.append({
            get_translation(lang, "dashboard.model"): f.name,
            "Tamaño (KB)" if lang == "es" else "Size (KB)": f"{size_kb:.1f}",
        })
    st.table(pd.DataFrame(data_rows))
else:
    st.warning(f"{get_translation(lang, 'dashboard.path_not_found')}: `{DATA_DIR}`")

# ── Modelos ─────────────────────────────────────────────
st.subheader(get_translation(lang, "dashboard.trained_models"))

if model_files:
    mdl_rows = []
    for f in sorted(model_files, key=lambda x: (get_model_type(x.name), x.name)):
        size_mb = f.stat().st_size / (1024 * 1024)
        mdl_rows.append({
            get_translation(lang, "dashboard.type"): classify_model(f.name),
            get_translation(lang, "dashboard.model"): f.name,
            get_translation(lang, "dashboard.size_mb"): f"{size_mb:.2f}",
        })
    st.table(pd.DataFrame(mdl_rows).sort_values(get_translation(lang, "dashboard.type")))
else:
    st.info("No hay modelos descargados localmente. Se descargarán desde HuggingFace Hub al configurar las credenciales.")

with st.expander(get_translation(lang, "dashboard.summary_by_type")):
    type_rows = []
    for cat_label, _, _ in MODEL_CATEGORIES:
        count = sum(1 for f in model_files if classify_model(f.name) == cat_label)
        if count > 0:
            type_rows.append({
                get_translation(lang, "dashboard.type"): cat_label,
                get_translation(lang, "dashboard.quantity"): count,
            })
    if type_rows:
        st.table(pd.DataFrame(type_rows).set_index(get_translation(lang, "dashboard.type")))

# ── Workflow ────────────────────────────────────────────
st.markdown("---")
st.subheader(get_translation(lang, "dashboard.recommended_workflow"))
workflow_steps = get_translation(lang, "dashboard.workflow_steps")
st.markdown("\n".join([f"{i+1}. {step}" for i, step in enumerate(workflow_steps)]))
