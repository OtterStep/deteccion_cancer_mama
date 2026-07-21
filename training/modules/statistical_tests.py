import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency, wilcoxon, friedmanchisquare, norm, studentized_range
from sklearn.metrics import confusion_matrix, roc_auc_score


def mcnemar_test(y_true, y_pred_model1, y_pred_model2):
    """
    Prueba de McNemar para comparar dos modelos.
    H0: No hay diferencia en las proporciones de error entre modelos.
    """
    cm = confusion_matrix(
        (y_pred_model1 != y_true).astype(int),
        (y_pred_model2 != y_true).astype(int),
    )
    if cm.shape != (2, 2):
        return {"statistic": None, "p_value": None, "error": "Matriz no 2x2"}

    b = int(cm[0, 1])
    c = int(cm[1, 0])

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "note": "Modelos idénticos"}

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(statistic, 1)

    return {"statistic": round(float(statistic), 4), "p_value": round(float(p_value), 4)}


def wilcoxon_test(y_true, y_proba_model1, y_proba_model2):
    """
    Prueba de Wilcoxon Signed-Rank para comparar probabilidades de dos modelos.
    H0: Las distribuciones de probabilidades son iguales.
    """
    diff = y_proba_model1 - y_proba_model2
    diff = diff[diff != 0]
    if len(diff) < 2:
        return {"statistic": None, "p_value": None, "note": "Diferencias insuficientes"}

    try:
        stat, p = wilcoxon(diff, alternative="two-sided")
        return {"statistic": round(float(stat), 4), "p_value": round(float(p), 4)}
    except ValueError as e:
        return {"statistic": None, "p_value": None, "error": str(e)}


def interpret_mcnemar(p_value, alpha=0.05):
    if p_value is None:
        return "No se pudo calcular."
    if p_value < alpha:
        return f"**Diferencias significativas** (p = {p_value:.4f} < {alpha}): Los modelos tienen rendimiento diferente."
    return f"**Sin diferencias significativas** (p = {p_value:.4f} ≥ {alpha}): Los modelos tienen rendimiento similar."


def interpret_wilcoxon(p_value, alpha=0.05):
    if p_value is None:
        return "No se pudo calcular."
    if p_value < alpha:
        return f"**Diferencias significativas** (p = {p_value:.4f} < {alpha}): Las distribuciones de probabilidad difieren."
    return f"**Sin diferencias significativas** (p = {p_value:.4f} ≥ {alpha}): Las distribuciones de probabilidad son similares."


def run_all_tests(models_data: dict, y_true) -> pd.DataFrame:
    """
    models_data: {name: {"y_pred": ..., "y_proba": ...}}
    y_true: valores reales
    """
    model_names = list(models_data.keys())
    rows = []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            n1, n2 = model_names[i], model_names[j]
            m1 = models_data[n1]
            m2 = models_data[n2]

            mcnemar = mcnemar_test(y_true, m1["y_pred"], m2["y_pred"])
            wilc = wilcoxon_test(y_true, m1["y_proba"], m2["y_proba"])

            rows.append({
                "Modelo A": n1,
                "Modelo B": n2,
                "McNemar χ²": mcnemar.get("statistic", "-"),
                "McNemar p-valor": mcnemar.get("p_value", "-"),
                "McNemar Interpretación": interpret_mcnemar(mcnemar.get("p_value")),
                "Wilcoxon W": wilc.get("statistic", "-"),
                "Wilcoxon p-valor": wilc.get("p_value", "-"),
                "Wilcoxon Interpretación": interpret_wilcoxon(wilc.get("p_value")),
            })

    return pd.DataFrame(rows)


def friedman_test(*probas):
    """
    Friedman test para comparar múltiples modelos.
    H0: Todos los modelos tienen la misma distribución de probabilidades.
    probas: arrays de probabilidades (uno por modelo).
    """
    try:
        stat, p = friedmanchisquare(*probas)
        return {"statistic": round(float(stat), 4), "p_value": round(float(p), 4)}
    except Exception as e:
        return {"statistic": None, "p_value": None, "error": str(e)}


def nemenyi_posthoc(*probas, alpha=0.05):
    """
    Prueba post-hoc de Nemenyi después de Friedman.
    Compara todos los pares usando la diferencia de rangos promedio.
    """
    n_models = len(probas)
    n_samples = len(probas[0])

    # Ranking por muestra (de menor probabilidad = rank 1)
    ranks = np.zeros((n_samples, n_models))
    for i in range(n_samples):
        ranks[i] = np.argsort(np.argsort([p[i] for p in probas])) + 1

    mean_ranks = ranks.mean(axis=0)
    model_names = [f"Modelo {i+1}" for i in range(n_models)]

    q_alpha = studentized_range.ppf(1 - alpha, n_models, 1e6)

    rows = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            diff = abs(mean_ranks[i] - mean_ranks[j])
            se = np.sqrt(n_models * (n_models + 1) / (6 * n_samples))
            q_stat = diff / se
            p_value = 1 - studentized_range.cdf(q_stat, n_models, 1e6)

            rows.append({
                "Modelo A": model_names[i],
                "Modelo B": model_names[j],
                "Diferencia de rangos": round(diff, 4),
                "Estadístico Q": round(q_stat, 4),
                "p-valor (Nemenyi)": round(float(p_value), 4),
                "Significativo": "Sí" if p_value < alpha else "No",
            })

    return pd.DataFrame(rows), mean_ranks


def delong_test(y_true, y_proba_model1, y_proba_model2):
    """
    Test de DeLong para comparar dos curvas ROC (AUC).
    H0: Los AUC de ambos modelos son iguales.
    Implementación basada en DeLong et al. (1988).
    """
    auc1 = roc_auc_score(y_true, y_proba_model1)
    auc2 = roc_auc_score(y_true, y_proba_model2)

    n1 = np.sum(y_true == 1)
    n2 = np.sum(y_true == 0)
    n = n1 + n2

    # Ordenar por probabilidad
    order = np.argsort(y_proba_model1)
    y_true_s = y_true[order]
    y_proba1_s = y_proba_model1[order]

    # Componentes de la matriz de covarianza
    v10 = np.zeros(n)
    v01 = np.zeros(n)
    for i in range(n):
        if y_true_s[i] == 1:
            v10[i] = np.sum(y_proba1_s[:i] <= y_proba1_s[i]) + np.sum(y_proba1_s[i+1:] < y_proba1_s[i])
        else:
            v01[i] = np.sum(y_proba1_s[:i] >= y_proba1_s[i]) + np.sum(y_proba1_s[i+1:] > y_proba1_s[i])

    v10 = v10 / n1 - auc1
    v01 = v01 / n2 - auc1

    # Varianza
    s2 = np.var(v10) / n1 + np.var(v01) / n2

    order2 = np.argsort(y_proba_model2)
    y_true_s2 = y_true[order2]
    y_proba2_s = y_proba_model2[order2]

    v10_2 = np.zeros(n)
    v01_2 = np.zeros(n)
    for i in range(n):
        if y_true_s2[i] == 1:
            v10_2[i] = np.sum(y_proba2_s[:i] <= y_proba2_s[i]) + np.sum(y_proba2_s[i+1:] < y_proba2_s[i])
        else:
            v01_2[i] = np.sum(y_proba2_s[:i] >= y_proba2_s[i]) + np.sum(y_proba2_s[i+1:] > y_proba2_s[i])

    v10_2 = v10_2 / n1 - auc2
    v01_2 = v01_2 / n2 - auc2

    s2_2 = np.var(v10_2) / n1 + np.var(v01_2) / n2

    # Covarianza entre los dos modelos
    cov = np.cov(np.concatenate([v10, v01]), np.concatenate([v10_2, v01_2]))[0, 1]

    # Estadístico Z (proteger contra varianza negativa por redondeo)
    var_diff = max(s2 + s2_2 - 2 * cov, 1e-10)
    se = np.sqrt(var_diff)
    if se == 0:
        return {"auc1": auc1, "auc2": auc2, "z_statistic": None, "p_value": 1.0, "note": "Errores idénticos"}

    z = (auc1 - auc2) / se
    p = 2 * (1 - norm.cdf(abs(z)))

    return {
        "auc1": round(float(auc1), 4),
        "auc2": round(float(auc2), 4),
        "z_statistic": round(float(z), 4),
        "p_value": round(float(p), 4),
    }


def interpret_friedman(p_value, alpha=0.05):
    if p_value is None:
        return "No se pudo calcular."
    if p_value < alpha:
        return f"**Diferencias significativas** (p = {p_value:.4f} < {alpha}): Al menos un modelo difiere de los demás."
    return f"**Sin diferencias significativas** (p = {p_value:.4f} ≥ {alpha}): Los modelos tienen rendimiento similar."


def interpret_delong(p_value, alpha=0.05):
    if p_value is None:
        return "No se pudo calcular."
    if p_value < alpha:
        return f"**Diferencias significativas** (p = {p_value:.4f} < {alpha}): Los AUC son diferentes."
    return f"**Sin diferencias significativas** (p = {p_value:.4f} ≥ {alpha}): Los AUC son comparables."


def plot_pvalue_heatmap(test_results: pd.DataFrame) -> go.Figure:
    models = sorted(set(test_results["Modelo A"].tolist() + test_results["Modelo B"].tolist()))
    n = len(models)
    p_matrix = np.ones((n, n))

    for _, row in test_results.iterrows():
        i = models.index(row["Modelo A"])
        j = models.index(row["Modelo B"])
        try:
            p = float(row["McNemar p-valor"])
            p_matrix[i, j] = p
            p_matrix[j, i] = p
        except (ValueError, TypeError):
            pass

    fig = go.Figure(data=go.Heatmap(
        z=p_matrix,
        x=models,
        y=models,
        text=[[f"{v:.4f}" if v < 1 else "1.0" for v in row] for row in p_matrix],
        texttemplate="%{text}",
        colorscale="RdYlGn_r",
        zmin=0, zmax=1,
    ))
    fig.update_layout(
        title="Mapa de Calor - p-valores McNemar entre Modelos",
        height=400,
        margin=dict(t=40),
    )
    return fig
