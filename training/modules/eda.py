import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


class BreastCancerEDA:
    def __init__(self, base_path: str = None, *,
                 csv_path: str = None,
                 images_path: str = None,
                 tabular_path: str = None,
                 database_path: str = None,
                 results_path: str = None):
        if base_path is not None:
            _base = Path(base_path)
            self.base_path = _base
            self.csv_path = Path(csv_path) if csv_path else _base / "CSVFiles"
            self.images_path = Path(images_path) if images_path else _base / "BreastCancer_Images" / "jpeg"
            self.tabular_path = Path(tabular_path) if tabular_path else _base / "BreastCancer_Tabular"
            self.database_path = Path(database_path) if database_path else _base / "Database"
            self.results_path = Path(results_path) if results_path else _base / "Resultados"
        else:
            from config import DATA_DIR, TABULAR_DIR, RESULTS_DIR
            self.csv_path = Path(csv_path or DATA_DIR)
            self.tabular_path = Path(tabular_path or TABULAR_DIR)
            self.results_path = Path(results_path or RESULTS_DIR)
            try:
                from config import IMAGES_DIR, DATABASE_DIR
            except ImportError:
                IMAGES_DIR = self.csv_path.parent / "images"
                DATABASE_DIR = self.csv_path.parent / "database"
            self.images_path = Path(images_path or IMAGES_DIR)
            self.database_path = Path(database_path or DATABASE_DIR)
            self.base_path = self.csv_path.parent

        self.results_path.mkdir(exist_ok=True)

        self.calc_all: pd.DataFrame = None
        self.mass_all: pd.DataFrame = None
        self.meta: pd.DataFrame = None
        self.dicom_info: pd.DataFrame = None
        self.wisconsin_data: pd.DataFrame = None

    def load_all_data(self):
        self.calc_train = pd.read_csv(self.csv_path / "calc_case_description_train_set.csv")
        self.calc_test = pd.read_csv(self.csv_path / "calc_case_description_test_set.csv")
        self.mass_train = pd.read_csv(self.csv_path / "mass_case_description_train_set.csv")
        self.mass_test = pd.read_csv(self.csv_path / "mass_case_description_test_set.csv")
        self.meta = pd.read_csv(self.csv_path / "meta.csv")

        dicom_path = self.csv_path / "dicom_info.csv"
        self.dicom_info = pd.read_csv(dicom_path) if dicom_path.exists() else None

        self.wisconsin_data = pd.read_csv(self.tabular_path / "data.csv")

        self.calc_all = pd.concat([self.calc_train, self.calc_test], ignore_index=True)
        self.mass_all = pd.concat([self.mass_train, self.mass_test], ignore_index=True)

    @property
    def is_loaded(self) -> bool:
        return self.wisconsin_data is not None

    def get_data_summary(self) -> dict:
        return {
            "calcificaciones": len(self.calc_all) if self.calc_all is not None else 0,
            "masas": len(self.mass_all) if self.mass_all is not None else 0,
            "metadatos": len(self.meta) if self.meta is not None else 0,
            "wisconsin": len(self.wisconsin_data) if self.wisconsin_data is not None else 0,
        }

    def get_pathology_distribution(self) -> dict:
        result = {}
        if self.calc_all is not None and "pathology" in self.calc_all.columns:
            result["calcificaciones"] = self.calc_all["pathology"].value_counts().to_dict()
        if self.mass_all is not None and "pathology" in self.mass_all.columns:
            result["masas"] = self.mass_all["pathology"].value_counts().to_dict()
        if self.wisconsin_data is not None and "diagnosis" in self.wisconsin_data.columns:
            diag = self.wisconsin_data["diagnosis"].map({"M": "MALIGNANT", "B": "BENIGN"}).value_counts()
            result["wisconsin"] = diag.to_dict()
        return result

    def _clean_wisconsin(self) -> pd.DataFrame:
        df = self.wisconsin_data.drop(["id", "Unnamed: 32"], axis=1, errors="ignore")
        return df.dropna()

    def get_wisconsin_descriptive_stats(self) -> pd.DataFrame:
        df = self._clean_wisconsin()
        diag = df["diagnosis"]
        features = df.drop("diagnosis", axis=1)
        stats = features.describe().T
        stats.columns = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        return stats

    def get_wisconsin_stats_by_class(self) -> dict:
        df = self._clean_wisconsin()
        features = df.drop("diagnosis", axis=1)
        ben = features[df["diagnosis"] == "B"]
        mal = features[df["diagnosis"] == "M"]
        return {
            "BENIGN": ben.describe().T,
            "MALIGNANT": mal.describe().T,
        }

    def get_birads_distribution(self) -> dict:
        result = {}
        for name, dd in [("Calcificaciones", self.calc_all), ("Masas", self.mass_all)]:
            if dd is not None and "assessment" in dd.columns:
                ass = dd["assessment"].value_counts().sort_index()
                result[name] = ass.to_dict()
        return result

    def get_subtlety_distribution(self) -> dict:
        result = {}
        for name, dd in [("Calcificaciones", self.calc_all), ("Masas", self.mass_all)]:
            if dd is not None and "subtlety" in dd.columns:
                sub = dd["subtlety"].value_counts().sort_index()
                result[name] = sub.to_dict()
        return result

    def get_correlation_matrix(self) -> pd.DataFrame:
        df = self._clean_wisconsin()
        return df.drop("diagnosis", axis=1).corr()

    def get_image_mapping_stats(self) -> dict:
        if self.meta is None or "SeriesInstanceUID" not in self.meta.columns:
            return {"series_en_meta": 0, "carpetas_imagenes": 0, "coincidencias": 0}
        series_in_meta = set(self.meta["SeriesInstanceUID"].unique())
        image_folders = set()
        if self.images_path.exists():
            image_folders = {d.name for d in self.images_path.iterdir() if d.is_dir()}
        matching = series_in_meta & image_folders
        return {
            "series_en_meta": len(series_in_meta),
            "carpetas_imagenes": len(image_folders),
            "coincidencias": len(matching),
        }

    # ── Plotly figures ──────────────────────────────────────────────────

    def plot_pathology_distribution(self) -> go.Figure:
        dist = self.get_pathology_distribution()
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[f"Calcificaciones", f"Masas", f"Wisconsin"],
            specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]],
        )
        colors = {"BENIGN": "#22c55e", "MALIGNANT": "#ef4444", "BENIGN_WITHOUT_CALLBACK": "#facc15"}
        for i, key in enumerate(["calcificaciones", "masas", "wisconsin"], 1):
            data = dist.get(key, {})
            if data:
                labels = list(data.keys())
                values = list(data.values())
                fig.add_trace(
                    go.Pie(
                        labels=labels, values=values,
                        marker=dict(colors=[colors.get(l, "#94a3b8") for l in labels]),
                        textinfo="label+percent",
                        hole=0.35,
                    ),
                    row=1, col=i,
                )
        fig.update_layout(
            title_text="Distribución de Patologías por Dataset",
            height=400, showlegend=False,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        return fig

    def plot_birads_distribution(self) -> go.Figure:
        birads = self.get_birads_distribution()
        fig = go.Figure()
        for name in birads:
            data = birads[name]
            fig.add_trace(go.Bar(
                name=name, x=list(map(str, data.keys())), y=list(data.values()),
                text=list(data.values()), textposition="outside",
            ))
        fig.update_layout(
            title="Distribución BI-RADS Assessment",
            xaxis_title="Assessment",
            yaxis_title="Cantidad",
            barmode="group",
            height=400,
            margin=dict(t=40),
        )
        return fig

    def plot_subtlety_distribution(self) -> go.Figure:
        subtle = self.get_subtlety_distribution()
        fig = go.Figure()
        for name in subtle:
            data = subtle[name]
            fig.add_trace(go.Bar(
                name=name, x=list(map(str, data.keys())), y=list(data.values()),
                text=list(data.values()), textposition="outside",
            ))
        fig.update_layout(
            title="Distribución de Subtlety (Dificultad de Detección)",
            xaxis_title="Subtlety (1 = Muy sutil, 5 = Muy evidente)",
            yaxis_title="Cantidad",
            barmode="group",
            height=400,
            margin=dict(t=40),
        )
        return fig

    def plot_wisconsin_boxplots(self, n_features: int = 10) -> go.Figure:
        df = self._clean_wisconsin()
        features = df.drop("diagnosis", axis=1).columns[:n_features]
        n_cols = 5
        n_rows = int(np.ceil(len(features) / n_cols))
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(features))
        for i, feat in enumerate(features, 1):
            r = (i - 1) // n_cols + 1
            c = (i - 1) % n_cols + 1
            for cls, color in [("B", "#22c55e"), ("M", "#ef4444")]:
                vals = df[df["diagnosis"] == cls][feat]
                fig.add_trace(
                    go.Box(y=vals, name=cls, marker_color=color, showlegend=i == 1),
                    row=r, col=c,
                )
        fig.update_layout(
            title=f"Comparación BENIGN vs MALIGNANT (primeras {n_features} características)",
            height=250 * n_rows,
            margin=dict(t=40),
        )
        return fig

    def plot_correlation_heatmap(self) -> go.Figure:
        corr = self.get_correlation_matrix()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto",
            height=700,
        )
        fig.update_layout(
            title="Matriz de Correlación - Características Wisconsin",
            margin=dict(t=40),
        )
        return fig

    def plot_feature_distributions(self, n_features: int = 10) -> go.Figure:
        df = self._clean_wisconsin()
        features = df.drop("diagnosis", axis=1).columns[:n_features]
        n_cols = 5
        n_rows = int(np.ceil(len(features) / n_cols))
        fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=list(features))
        for i, feat in enumerate(features, 1):
            r = (i - 1) // n_cols + 1
            c = (i - 1) % n_cols + 1
            for cls, color in [("B", "green"), ("M", "red")]:
                vals = df[df["diagnosis"] == cls][feat]
                fig.add_trace(
                    go.Histogram(x=vals, name=cls, marker_color=color,
                                 opacity=0.6, showlegend=i == 1),
                    row=r, col=c,
                )
        fig.update_layout(
            title=f"Distribución de Características por Clase",
            height=250 * n_rows,
            barmode="overlay",
            margin=dict(t=40),
        )
        return fig
