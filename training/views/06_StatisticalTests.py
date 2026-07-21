import json

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from config import JSON_RESULTS_PATH
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


data = load_json_results()

st.title(get_translation(lang, "statistical_tests.title"))
st.markdown(get_translation(lang, "statistical_tests.description"))

if not data:
    st.warning("No se encontró el archivo de resultados. Ejecuta el pipeline de entrenamiento primero.")
    st.stop()

# ──────────────────────────────────────────────
# 1. MODELOS DEL SISTEMA
# ──────────────────────────────────────────────
st.subheader(get_translation(lang, "statistical_tests.models_section_title"))
st.markdown(get_translation(lang, "statistical_tests.models_section_desc"))

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"**{get_translation(lang, 'statistical_tests.classic_models_title')}**")
    classic_df = pd.DataFrame([
        [get_translation(lang, "statistical_tests.model_cnn"),
         get_translation(lang, "statistical_tests.model_cnn_data"),
         get_translation(lang, "statistical_tests.model_cnn_training"),
         get_translation(lang, "statistical_tests.model_cnn_purpose")],
        [get_translation(lang, "statistical_tests.model_xgb"),
         get_translation(lang, "statistical_tests.model_xgb_data"),
         get_translation(lang, "statistical_tests.model_xgb_training"),
         get_translation(lang, "statistical_tests.model_xgb_purpose")],
        [get_translation(lang, "statistical_tests.model_rf"),
         get_translation(lang, "statistical_tests.model_rf_data"),
         get_translation(lang, "statistical_tests.model_rf_training"),
         get_translation(lang, "statistical_tests.model_rf_purpose")],
    ], columns=[
        get_translation(lang, "statistical_tests.model_col"),
        get_translation(lang, "statistical_tests.data_col"),
        get_translation(lang, "statistical_tests.training_col"),
        get_translation(lang, "statistical_tests.purpose_col"),
    ])
    st.table(classic_df)

with col2:
    st.markdown(f"**{get_translation(lang, 'statistical_tests.hybrid_models_title')}**")
    hybrid_df = pd.DataFrame([
        [get_translation(lang, "statistical_tests.model_ensemble"),
         get_translation(lang, "statistical_tests.model_ensemble_data"),
         get_translation(lang, "statistical_tests.model_ensemble_training"),
         get_translation(lang, "statistical_tests.model_ensemble_purpose")],
        [get_translation(lang, "statistical_tests.model_hybrid"),
         get_translation(lang, "statistical_tests.model_hybrid_data"),
         get_translation(lang, "statistical_tests.model_hybrid_training"),
         get_translation(lang, "statistical_tests.model_hybrid_purpose")],
    ], columns=[
        get_translation(lang, "statistical_tests.model_col"),
        get_translation(lang, "statistical_tests.data_col"),
        get_translation(lang, "statistical_tests.training_col"),
        get_translation(lang, "statistical_tests.purpose_col"),
    ])
    st.table(hybrid_df)

st.divider()

# ──────────────────────────────────────────────
# 2. MÉTRICAS COMPLETAS (TODOS LOS MODELOS)
# ──────────────────────────────────────────────
st.subheader(get_translation(lang, "statistical_tests.metrics_title"))

metricas = data.get("fase2_modelado", {}).get("metricas_test", {})

if metricas:
    metrics_rows = []
    for model_key, m in metricas.items():
        metrics_rows.append({
            get_translation(lang, "statistical_tests.model_col"): format_model_name(model_key),
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1']:.4f}",
            "AUC-ROC": f"{m['auc']:.4f}",
        })
    st.table(pd.DataFrame(metrics_rows))

    # Mejor modelo
    best_key = max(metricas, key=lambda k: metricas[k]["auc"] * 0.4 + metricas[k]["f1"] * 0.3 + metricas[k]["accuracy"] * 0.3)
    best = metricas[best_key]
    st.success(get_translation(lang, "statistical_tests.best_model_result").format(
        model=format_model_name(best_key),
        auc=f"{best['auc']:.4f}",
        accuracy=f"{best['accuracy']:.4f}",
        f1=f"{best['f1']:.4f}",
    ))

st.divider()

# ──────────────────────────────────────────────
# 3. VISTA UNIFICADA
# ──────────────────────────────────────────────
st.subheader(get_translation(lang, "statistical_tests.unified_view_title"))
st.markdown(get_translation(lang, "statistical_tests.unified_view_desc"))

if metricas:
    unified_rows = []
    for model_key, m in metricas.items():
        score = m["auc"] * 0.4 + m["f1"] * 0.3 + m["accuracy"] * 0.3
        group = "Tabular" if model_key in ("tabular_xgboost", "tabular_rf") else "Imagen"
        unified_rows.append({
            get_translation(lang, "statistical_tests.model_col"): format_model_name(model_key),
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1-Score": m["f1"],
            "AUC-ROC": m["auc"],
            get_translation(lang, "statistical_tests.score_col"): round(score, 4),
            "Grupo": group,
        })

    unified_df = pd.DataFrame(unified_rows)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC", get_translation(lang, "statistical_tests.score_col")]
    display_cols = [get_translation(lang, "statistical_tests.model_col")] + metric_cols + ["Grupo"]
    display_unified = unified_df[display_cols]

    styled = display_unified.style
    for col in metric_cols:
        col_max = unified_df[col].max()
        styled = styled.map(
            lambda v, mx=col_max: "background-color: #90EE90; font-weight: bold"
                if isinstance(v, (int, float)) and v == mx else "",
            subset=[col],
        )
    st.dataframe(styled, use_container_width=True)
    st.caption(get_translation(lang, "statistical_tests.score_formula"))

    # Gráfico
    chart_df = unified_df.melt(
        id_vars=[get_translation(lang, "statistical_tests.model_col"), "Grupo"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"],
        var_name="Métrica",
        value_name="Valor (%)",
    )
    chart_df["Valor (%)"] = (chart_df["Valor (%)"] * 100).round(1)

    fig = px.bar(
        chart_df,
        x="Métrica",
        y="Valor (%)",
        color=get_translation(lang, "statistical_tests.model_col"),
        barmode="group",
        text="Valor (%)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="outside", textfont_size=10)
    fig.update_layout(
        yaxis_range=[0, 105],
        height=450,
        margin=dict(t=30, b=30),
        legend_title_text=get_translation(lang, "statistical_tests.model_col"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        "**Interpretación de la comparativa porcentual:**  \n"
        "En escala porcentual se aprecia claramente la **brecha** entre modelos tabulares "
        "(>95% en todas las métricas) y modelos de imagen (~60% en Accuracy).  \n"
        "**Random Forest** lidera con el Score Compuesto más alto, seguido muy de cerca por "
        "**XGBoost** (diferencia no significativa, p = 1.0 en McNemar).  \n"
        "Los modelos **híbridos (CNN+RF y CNN+XGBoost)** muestran un rendimiento intermedio "
        "con Accuracy ~62%, superando a CNN pura pero lejos de los tabulares.  \n"
        "**CNN (EfficientNet)** tiene el rendimiento más bajo (AUC = 0.56), lo cual es esperable "
        "dado que clasifica la imagen completa sin segmentación de la región de interés."
    )

st.divider()

# ──────────────────────────────────────────────
# 4. PRUEBAS POR PARES (desde JSON)
# ──────────────────────────────────────────────
pruebas = data.get("pruebas_estadisticas", {})

st.subheader("📊 Pruebas Estadísticas por Pares")
st.markdown("Resultados de **McNemar**, **Wilcoxon**, **DeLong** y **t-pareada** precomputados desde Google Colab.")

resumen = pruebas.get("resumen_pares", [])
if resumen:
    pairs_df = pd.DataFrame(resumen)
    st.dataframe(pairs_df, use_container_width=True)

    # Heatmap de p-valores (McNemar)
    mcnemar = pruebas.get("mcnemar_pvalues", {})
    if mcnemar:
        st.subheader(get_translation(lang, "statistical_tests.heatmap"))
        model_names = list(mcnemar.keys())
        n = len(model_names)
        heatmap_data = np.full((n, n), np.nan)
        for i, a in enumerate(model_names):
            for j, b in enumerate(model_names):
                if a == b:
                    heatmap_data[i][j] = 1.0
                elif b in mcnemar.get(a, {}):
                    v = mcnemar[a][b]
                    heatmap_data[i][j] = v if v is not None else 1.0
                elif a in mcnemar.get(b, {}):
                    v = mcnemar[b][a]
                    heatmap_data[i][j] = v if v is not None else 1.0

        fig_heat = px.imshow(
            heatmap_data,
            x=[format_model_name(m) for m in model_names],
            y=[format_model_name(m) for m in model_names],
            color_continuous_scale="RdBu_r",
            zmin=0, zmax=1,
            text_auto=".4f",
            aspect="auto",
            height=500,
            title="McNemar p-valores",
        )
        fig_heat.update_layout(margin=dict(t=40))
        st.plotly_chart(fig_heat, use_container_width=True)
        st.info(
            "**Interpretación del mapa de calor (McNemar):**  \n"
            "Las celdas en tonos **azules/verdes** (p < 0.05) indican diferencias estadísticamente "
            "significativas entre las predicciones de dos modelos.  \n"
            "Se observa que **CNN (EfficientNet)** difiere significativamente de los modelos "
            "híbridos (p < 0.005 con Bonferroni), consistente con su menor rendimiento.  \n"
            "**Random Forest** y **XGBoost** no muestran diferencias significativas entre sí "
            "(p = 1.0), lo que sugiere que ambos modelos tabulares tienen rendimiento comparable "
            "en este dataset.  \n"
            "Las celdas en gris claro indican valores > 0.05 (diferencias no significativas).  \n"
            "**Nota:** Los pares entre grupos (tabular vs imagen) no tienen p-valor porque operan "
            "sobre conjuntos de datos diferentes (Wisconsin vs mamografías)."
        )

    # Interpretación general
    if resumen:
        total = len(resumen)
        sig_count = sum(1 for r in resumen if r.get("Significativo (Bonferroni)", False))
        if sig_count == 0:
            st.info(get_translation(lang, "statistical_tests.no_significant", total=total))
        elif sig_count == total:
            st.warning(get_translation(lang, "statistical_tests.all_significant", total=total))
        else:
            st.info(get_translation(lang, "statistical_tests.some_significant", sig_count=sig_count, total=total))

st.divider()

# ──────────────────────────────────────────────
# 5. RANKING
# ──────────────────────────────────────────────
ranking = data.get("ranking_modelos", {})
if ranking and "tabla" in ranking:
    st.subheader("🏆 Ranking de Modelos")

    rank_df = pd.DataFrame(ranking["tabla"])
    display_cols = [c for c in ["Puesto", "Modelo", "accuracy", "precision", "recall", "f1", "auc",
                                  "Score compuesto", "CV Accuracy (media)", "CV AUC (media)"] if c in rank_df.columns]
    display_rank = rank_df[display_cols].copy()
    for col in display_rank.columns:
        if col not in ["Puesto", "Modelo"]:
            display_rank[col] = display_rank[col].apply(lambda x: f"{x:.4f}")
    st.dataframe(display_rank, use_container_width=True)

    mejor = ranking.get("mejor_modelo", "")
    if mejor:
        st.success(
            f"{get_translation(lang, 'statistical_tests.best_model_overall')}: **{format_model_name(mejor)}**"
        )

st.divider()

# ──────────────────────────────────────────────
# 6. CONCLUSIONES
# ──────────────────────────────────────────────
st.subheader(get_translation(lang, "statistical_tests.conclusions_title"))

ranking_data = data.get("ranking_modelos", {})
best_model_key = ranking_data.get("mejor_modelo", "")
best_model_name = format_model_name(best_model_key) if best_model_key else "N/A"

st.success(
    f"**Conclusión general:** El mejor modelo del sistema es **{best_model_name}**, "
    f"con un Score Compuesto de **{ranking_data['tabla'][0]['Score compuesto']:.4f}** "
    f"(AUC = {ranking_data['tabla'][0]['auc']:.4f}, F1 = {ranking_data['tabla'][0]['f1']:.4f}).  \n"
    "Los modelos tabulares superan ampliamente a los basados en imagen, lo que sugiere que "
    "las 30 características morfológicas del núcleo celular (Wisconsin) tienen mayor poder "
    "discriminativo que las mamografías completas sin segmentación.  \n"
    "Para producción, se recomienda usar **Random Forest** o **XGBoost** como modelo principal, "
    "y reservar los modelos de imagen para casos donde no se disponga de datos tabulares."
)

conc_col1, conc_col2 = st.columns(2)

with conc_col1:
    st.markdown(f"**{get_translation(lang, 'statistical_tests.classic_models_title')}**")
    st.markdown(get_translation(lang, "statistical_tests.conclusion_cnn"))
    st.markdown("---")
    st.markdown(get_translation(lang, "statistical_tests.conclusion_xgb"))
    st.markdown("---")
    st.markdown(get_translation(lang, "statistical_tests.conclusion_rf"))

with conc_col2:
    st.markdown(f"**{get_translation(lang, 'statistical_tests.hybrid_models_title')}**")
    st.markdown(get_translation(lang, "statistical_tests.conclusion_ensemble"))
    st.markdown("---")
    st.markdown(get_translation(lang, "statistical_tests.conclusion_hybrid"))
