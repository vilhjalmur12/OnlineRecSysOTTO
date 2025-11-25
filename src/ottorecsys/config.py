# src/ottorecsys/config.py
from pathlib import Path
import os

# Root of the project (relative to this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where all artifacts live (model runs, etc.)
ARTIFACT_DIR = PROJECT_ROOT / "models"

# For inference, pick a run directory via env var or default
MODEL_RUN = os.getenv("OTTO_RECSYS_MODEL_RUN", "1.0__df3a9d30")
RUN_DIR = ARTIFACT_DIR / MODEL_RUN

# Any other global serving-time constants
TOP_K_DEFAULT = 20
