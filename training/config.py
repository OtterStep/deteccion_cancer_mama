import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(os.getenv("TRAINING_DATA_DIR", str(BASE_DIR / "data" / "CSVFiles")))
TABULAR_DIR = DATA_DIR
MODELS_DIR = Path(os.getenv("TRAINING_MODELS_DIR", str(BASE_DIR / "data" / "models")))
RESULTS_DIR = Path(os.getenv("TRAINING_RESULTS_DIR", str(BASE_DIR / "data" / "resultados")))
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_SUMMARIES_DIR = RESULTS_DIR / "summaries"
FALLBACK_MODELS_DIR = MODELS_DIR
IMAGES_DIR = Path(os.getenv("TRAINING_IMAGES_DIR", str(BASE_DIR / "data" / "images")))
DATABASE_DIR = Path(os.getenv("TRAINING_DATABASE_DIR", str(BASE_DIR / "data" / "database")))

JSON_RESULTS_PATH = RESULTS_DIR / "modelos_cv_tuning.json"
PIPELINE_RESULTS_PATH = RESULTS_DIR / "summaries" / "resumen_pipeline_latest.json"

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
VAL_SIZE = float(os.getenv("VAL_SIZE", "0.2"))

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-4
PATIENCE = 10

DEFAULT_N_FOLDS = 5

APP_TITLE = "Sistema de Detección de Cáncer de Mama - Training Dashboard"
APP_ICON = "🩺"

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)
