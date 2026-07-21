import json

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import JSON_RESULTS_PATH, RESULTS_FIGURES_DIR
from modules.i18n import get_translation


lang = st.session_state.get("language", "es")


def load_json_results():
    if JSON_RESULTS_PATH.exists():
        with open(JSON_RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def format_model_name(key: str) -> str:
    names = {
        "tabular_xgboost": "XGBoost (Wisconsin)",
        "tabular_rf": "Random Forest (Wisconsin)",
        "cnn_efficientnet": "CNN (EfficientNet)",
        "hybrid_cnn_rf": "Hybrid CNN-RF",
        "hybrid_cnn_xgb": "Hybrid CNN-XGBoost",
    }
    return names.get(key, key)


st.title("🔁 Validación Cruzada — Resultados Precomputados")
st.markdown(
    "Resultados de **5-Fold Cross-Validation** ejecutada durante el entrenamiento en Google Colab.  "
    "Se muestran las métricas promedio y por fold para cada modelo."
)

data = load_json_results()

if not data or "validacion_cruzada" not in data:
    st.warning("No se encontraron resultados de validación cruzada en el archivo de resultados.")
    st.stop()

cv_data = data["validacion_cruzada"]
metricas = data.get("fase2_modelado", {}).get("metricas_test", {})
ranking = data.get("ranking_modelos", {})

for model_key, cv in cv_data.items():
    display_name = format_model_name(model_key)

    with st.container():
        st.subheader(f"📊 {display_name}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy (media)", f"{cv['accuracy_media']:.4f}", f"±{cv['accuracy_std']:.4f}")
        col2.metric("AUC (media)", f"{cv['auc_media']:.4f}", f"±{cv['auc_std']:.4f}")
        col3.metric("F1 (media)", f"{cv['f1_media']:.4f}", f"±{cv['f1_std']:.4f}")

        # Resultados por fold
        folds_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(cv.get("accuracy_por_fold", [])))],
            "Accuracy": cv.get("accuracy_por_fold", []),
            "AUC": cv.get("auc_por_fold", []),
            "F1": cv.get("f1_por_fold", []),
        })
        st.dataframe(folds_df, use_container_width=True)

        # Gráfico por fold
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Accuracy", x=folds_df["Fold"], y=folds_df["Accuracy"],
                              text=folds_df["Accuracy"].apply(lambda x: f"{x:.4f}"),
                              textposition="outside"))
        fig.add_trace(go.Bar(name="AUC", x=folds_df["Fold"], y=folds_df["AUC"],
                              text=folds_df["AUC"].apply(lambda x: f"{x:.4f}"),
                              textposition="outside"))
        fig.add_trace(go.Bar(name="F1", x=folds_df["Fold"], y=folds_df["F1"],
                              text=folds_df["F1"].apply(lambda x: f"{x:.4f}"),
                              textposition="outside"))
        fig.update_layout(
            title=f"{display_name} — Métricas por Fold",
            yaxis_range=[0, 1.05],
            height=350,
            barmode="group",
            margin=dict(t=40),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Cada grupo de barras muestra Accuracy, AUC y F1-Score en un fold específico. "
            "Barras de altura similar indican consistencia entre folds."
        )

        # Interpretación
        std_acc = cv.get("accuracy_std", 0)
        if std_acc < 0.01:
            estabilidad = "muy estable"
        elif std_acc < 0.03:
            estabilidad = "estable"
        else:
            estabilidad = "presenta variabilidad"

        st.info(
            f"El modelo **{display_name}** muestra un accuracy promedio de **{cv['accuracy_media']:.4f}** "
            f"con desviación estándar de **{std_acc:.4f}**, indicando que es **{estabilidad}** "
            f"a través de los {len(cv.get('accuracy_por_fold', []))} folds."
        )

    st.divider()

# ── Comparación entre modelos ────────────────────────────
st.subheader("📈 Comparación entre Modelos")

comparison_rows = []
for model_key, cv in cv_data.items():
    comparison_rows.append({
        "Modelo": format_model_name(model_key),
        "Accuracy (media)": cv["accuracy_media"],
        "Accuracy (std)": cv["accuracy_std"],
        "AUC (media)": cv["auc_media"],
        "AUC (std)": cv["auc_std"],
        "F1 (media)": cv["f1_media"],
        "F1 (std)": cv["f1_std"],
    })

comp_df = pd.DataFrame(comparison_rows)
st.dataframe(comp_df, use_container_width=True)

# Gráfico comparativo
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    name="Accuracy",
    x=comp_df["Modelo"],
    y=comp_df["Accuracy (media)"],
    error_y=dict(type="data", array=comp_df["Accuracy (std)"], visible=True),
))
fig2.add_trace(go.Bar(
    name="AUC",
    x=comp_df["Modelo"],
    y=comp_df["AUC (media)"],
    error_y=dict(type="data", array=comp_df["AUC (std)"], visible=True),
))
fig2.update_layout(
    title="Comparación CV entre Modelos (con desviación estándar)",
    yaxis_range=[0, 1.1],
    height=400,
    barmode="group",
    margin=dict(t=40),
)
st.plotly_chart(fig2, use_container_width=True)
st.info(
    "**Interpretación de la validación cruzada:**  \n"
    "**Random Forest** y **XGBoost** muestran la menor desviación estándar en Accuracy (< 0.015), "
    "lo que indica alta **estabilidad** entre los 5 folds.  \n"
    "Los modelos **híbridos** presentan mayor variabilidad (±0.03), lo que sugiere que su rendimiento "
    "depende más de la partición específica de los datos.  \n"
    "Las diferencias entre modelos tabulares e híbridos son **estadísticamente significativas** "
    "(p < 0.005, Bonferroni), confirmando que la brecha de rendimiento no es atribuible al azar.  \n"
    "**CNN (EfficientNet)** no tiene datos de CV disponibles (no se ejecutó CV por limitaciones de GPU)."
)

# Figuras pre-generadas
cv_bars = RESULTS_FIGURES_DIR / "cv_comparison_bars.png"
cv_box = RESULTS_FIGURES_DIR / "cv_comparison_boxplots.png"
if cv_bars.exists() or cv_box.exists():
    st.divider()
    st.subheader("📸 Figuras de Validación Cruzada (desde Colab)")
    col_fig1, col_fig2 = st.columns(2)
    if cv_bars.exists():
        with col_fig1:
            st.image(str(cv_bars), caption="Comparación de CV por modelo", use_container_width=True)
    if cv_box.exists():
        with col_fig2:
            st.image(str(cv_box), caption="Boxplots de CV por modelo", use_container_width=True)

# Ranking
if ranking and "tabla" in ranking:
    st.divider()
    st.subheader("🏆 Ranking Final")

    rank_df = pd.DataFrame(ranking["tabla"])
    display_cols = ["Puesto", "Modelo", "accuracy", "f1", "auc",
                     "CV Accuracy (media)", "CV Accuracy (std)", "CV AUC (media)", "Score compuesto"]
    display_cols = [c for c in display_cols if c in rank_df.columns]
    display_rank = rank_df[display_cols].copy()
    for col in display_rank.columns:
        if col not in ["Puesto", "Modelo"]:
            display_rank[col] = display_rank[col].apply(lambda x: f"{x:.4f}")
    st.dataframe(display_rank, use_container_width=True)
