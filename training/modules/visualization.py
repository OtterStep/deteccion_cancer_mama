import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go


def generate_confusion_matrix_fig(cm, title="Confusion Matrix") -> bytes:
    if isinstance(cm, dict):
        matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    else:
        matrix = cm
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred BENIGN", "Pred MALIGNANT"],
                yticklabels=["Real BENIGN", "Real MALIGNANT"],
                ax=ax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def generate_roc_curve_fig(y_test, probas_dict: dict) -> bytes:
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = {"XGBoost": "#8B5CF6", "Random Forest": "#22c55e",
              "EfficientNetB4": "#3b82f6", "Ensemble": "#f59e0b"}
    for name, y_proba in probas_dict.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=colors.get(name, "#3b82f6"),
                label=f"{name} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves - Model Comparison", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def generate_metrics_bar_fig(results: dict) -> bytes:
    metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(metrics))
    width = 0.3
    colors_list = ["#8B5CF6", "#22c55e", "#3b82f6"]
    for i, (name, res) in enumerate(results.items()):
        vals = [res["accuracy"], res["precision"], res["recall"], res["f1"], res["auc"]]
        ax.bar(x + i * width, vals, width, label=name, color=colors_list[i % 3])
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Model Metrics Comparison", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def generate_feature_importance_fig(importances, feature_names, title="Feature Importance", n_top=15) -> bytes:
    indices = np.argsort(importances)[-n_top:]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(range(len(indices)), importances[indices], color="#8B5CF6")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def generate_cv_bar_fig(fold_results: list) -> bytes:
    folds = [f"Fold {r['fold']}" for r in fold_results]
    accs = [r["accuracy"] for r in fold_results]
    aucs = [r["auc"] for r in fold_results]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(folds))
    ax.bar(x - 0.15, accs, 0.3, label="Accuracy", color="#8B5CF6")
    ax.bar(x + 0.15, aucs, 0.3, label="AUC", color="#22c55e")
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Cross-Validation Results by Fold", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
