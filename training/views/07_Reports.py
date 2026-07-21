import json
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

from config import RESULTS_DIR, JSON_RESULTS_PATH, DATA_DIR, TABULAR_DIR, RANDOM_STATE
from modules.eda import BreastCancerEDA
from modules.models import (
    prepare_wisconsin_data, train_xgboost, train_random_forest, evaluate_model,
)
from modules.cross_validation import run_cross_validation
from modules.hyperparameter_tuning import run_grid_search, XGBOOST_GRID
from modules.statistical_tests import run_all_tests
from modules.visualization import (
    generate_confusion_matrix_fig, generate_roc_curve_fig,
    generate_metrics_bar_fig, generate_cv_bar_fig,
)
from modules.reports import generate_pdf
from modules.i18n import get_translation


lang = st.session_state.get("language", "es")


@st.cache_resource
def load_eda():
    eda = BreastCancerEDA(
        csv_path=str(DATA_DIR),
        tabular_path=str(TABULAR_DIR),
        results_path=str(RESULTS_DIR),
    )
    eda.load_all_data()
    return eda


def load_json_results():
    if JSON_RESULTS_PATH.exists():
        with open(JSON_RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


st.title(get_translation(lang, "reports.title"))
st.markdown(get_translation(lang, "reports.description"))

if not DATA_DIR.exists():
    st.error(f"Directorio de datos no encontrado: `{DATA_DIR}`")
    st.stop()

try:
    eda = load_eda()
except Exception as e:
    st.error(f"{get_translation(lang, 'reports.error_loading_data')}: {e}")
    st.stop()

if eda.wisconsin_data is None:
    st.error(get_translation(lang, "reports.wisconsin_not_available"))
    st.stop()

st.subheader(get_translation(lang, "reports.generate_data"))

if st.button(get_translation(lang, "reports.run_pipeline"), type="primary", use_container_width=True):
    status = st.status(get_translation(lang, "reports.running_pipeline"))

    with status:
        st.write(get_translation(lang, "reports.preparing_data"))
        X_train, X_test, y_train, y_test = prepare_wisconsin_data(
            eda.wisconsin_data, random_state=RANDOM_STATE
        )
        wis_df = eda.wisconsin_data.drop(["id", "Unnamed: 32"], axis=1, errors="ignore").dropna()
        X_all = wis_df.drop("diagnosis", axis=1)
        y_all = (wis_df["diagnosis"] == "M").astype(int)

        st.write(get_translation(lang, "reports.training_models"))
        xgb_model, xgb_time = train_xgboost(X_train, y_train, X_test, y_test)
        rf_model, rf_time = train_random_forest(X_train, y_train)
        xgb_metrics = evaluate_model(xgb_model, X_test, y_test)
        xgb_metrics["training_time_s"] = xgb_time
        rf_metrics = evaluate_model(rf_model, X_test, y_test)
        rf_metrics["training_time_s"] = rf_time
        results = {"XGBoost": xgb_metrics, "Random Forest": rf_metrics}

        # Cargar métricas precomputadas del JSON para los modelos de imagen
        json_data = load_json_results()
        if json_data:
            metricas_json = json_data.get("fase2_modelado", {}).get("metricas_test", {})
            for key, m in metricas_json.items():
                if key in ("cnn_efficientnet", "hybrid_cnn_rf", "hybrid_cnn_xgb"):
                    model_label = {"cnn_efficientnet": "CNN", "hybrid_cnn_rf": "Hybrid CNN-RF", "hybrid_cnn_xgb": "Hybrid CNN-XGBoost"}.get(key, key)
                    results[model_label] = m

        st.write(get_translation(lang, "reports.running_cv"))
        cv_xgb = run_cross_validation(X_all, y_all, "XGBoost", n_splits=5, random_state=RANDOM_STATE)
        cv_rf = run_cross_validation(X_all, y_all, "Random Forest", n_splits=5, random_state=RANDOM_STATE)

        st.write(get_translation(lang, "reports.searching_params"))
        tuning = run_grid_search("XGBoost", X_all, y_all, XGBOOST_GRID, n_folds=3, random_state=RANDOM_STATE)

        st.write(get_translation(lang, "reports.running_stats"))
        models_data = {
            "XGBoost": {"y_pred": xgb_metrics["y_pred"], "y_proba": xgb_metrics["y_proba"]},
            "Random Forest": {"y_pred": rf_metrics["y_pred"], "y_proba": rf_metrics["y_proba"]},
        }
        stats_df = run_all_tests(models_data, y_test)

        st.write(get_translation(lang, "reports.generating_figures"))
        figures = {}
        for name in results:
            cm = results[name].get("confusion_matrix", {})
            if cm:
                figures[f"cm_{name}"] = generate_confusion_matrix_fig(cm, title=f"{name} - Confusion Matrix")
        probas_dict = {}
        if "XGBoost" in results:
            probas_dict["XGBoost"] = results["XGBoost"].get("y_proba", [])
        if "Random Forest" in results:
            probas_dict["Random Forest"] = results["Random Forest"].get("y_proba", [])
        if probas_dict and y_test is not None:
            figures["roc"] = generate_roc_curve_fig(y_test, probas_dict)
        if results:
            figures["metrics_bar"] = generate_metrics_bar_fig(results)
        for cv in [cv_xgb, cv_rf]:
            if cv and "fold_results" in cv:
                figures[f"cv_{cv['model_name']}"] = generate_cv_bar_fig(cv["fold_results"])

        eda_summary = eda.get_data_summary()

        st.write(get_translation(lang, "reports.generating_reports"))
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        pdf_path = str(RESULTS_DIR / f"reporte_modelos_{timestamp}.pdf")
        generate_pdf(results, [cv_xgb, cv_rf], tuning, stats_df, eda_summary, figures, pdf_path)
        st.success(get_translation(lang, "reports.pdf_success", path=pdf_path))

        st.balloons()

    st.subheader(get_translation(lang, "reports.generated_reports"))

    for label, path in [
        ("PDF", pdf_path)
    ]:
        p = Path(path)
        if p.exists():
            size_kb = p.stat().st_size / 1024
            with open(p, "rb") as f:
                st.download_button(
                    label=get_translation(lang, "reports.download", label=label, size=size_kb),
                    data=f.read(),
                    file_name=p.name,
                    mime="application/octet-stream",
                    use_container_width=True,
                )

    st.info(get_translation(lang, "reports.saved_info", results_dir=RESULTS_DIR))
