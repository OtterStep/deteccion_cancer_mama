import logging
import os
from pathlib import Path

logger = logging.getLogger("download_models")

HF_REPO_ID = os.getenv("HF_REPO_ID", "Jijiploff/cr-mama-fn-models")
HF_TOKEN = os.getenv("HF_TOKEN", None)

MODEL_FILES = [
    "tabular_rf_20260720_125151.pkl",
    "cnn_efficientnet_20260720_125151.keras",
    "extractor_hibrid_rf_cnn.keras",
    "classifier_hibrid_rf_cnn.pkl",
]


def download_models(models_dir: Path, repo_id: str = HF_REPO_ID, token: str = HF_TOKEN):
    """
    Descarga los modelos desde HuggingFace Hub si no existen localmente.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface-hub no está instalado. pip install huggingface-hub")
        return False

    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True, parents=True)

    all_ok = True
    for filename in MODEL_FILES:
        dest = models_dir / filename
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            logger.info(f"Ya existe: {filename} ({size_mb:.2f} MB)")
            continue

        try:
            logger.info(f"Descargando {filename} desde {repo_id}...")
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=token,
                local_dir=models_dir,
                local_dir_use_symlinks=False,
            )
            size_mb = Path(downloaded).stat().st_size / (1024 * 1024)
            logger.info(f"  OK {filename} ({size_mb:.2f} MB)")
        except Exception as e:
            logger.warning(f"  Error descargando {filename}: {e}")
            all_ok = False

    return all_ok
