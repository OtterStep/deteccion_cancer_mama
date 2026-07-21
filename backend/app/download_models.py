import logging
import os
import shutil
from huggingface_hub import hf_hub_download
from app.config import settings

logger = logging.getLogger("download_models")

REPO_ID = "Jijiploff/cr-mama-fn-models"

FILES = {
    "cnn_efficientnet_20260707_061411.keras": settings.CNN_MODEL_PATH,
    "ensemble_20260707_061411.keras": settings.ENSEMBLE_MODEL_PATH,
    "tabular_20260707_061411.pkl": settings.TABULAR_MODEL_PATH,
}


def download_all_models():
    for filename, dest_path in FILES.items():
        if os.path.isfile(dest_path):
            logger.info("%s ya existe, saltando descarga", dest_path)
            continue
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        logger.info("Descargando %s desde Hugging Face Hub...", filename)
        cached_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
        shutil.copy(cached_path, dest_path)
        logger.info("%s copiado a %s", filename, dest_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    download_all_models()