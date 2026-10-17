"""
src/utils.py - Utility Functions and Helpers
Provides logging, file validation, JSON serialization for NumPy/custom types,
and formatting utilities.
"""

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
import numpy as np

# Configure standard console logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sentiment_analysis")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy scalar and array types cleanly."""
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, Path)):
            return str(obj)
        return super().default(obj)


def save_json(data: dict, filepath: Path) -> None:
    """Save dictionary data to a JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)
    logger.info(f"Saved JSON data to {filepath}")


def load_json(filepath: Path) -> dict:
    """Load JSON data from a file, returning empty dict if missing."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"JSON file {filepath} not found.")
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def allowed_file(filename: str, allowed_extensions: set = None) -> bool:
    """Validate if the uploaded file has an allowed extension (e.g. .csv)."""
    if allowed_extensions is None:
        allowed_extensions = {"csv"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filenames to prevent path traversal attacks."""
    clean = re.sub(r"[^\w\.-]", "_", Path(filename).name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{clean}"


def format_percentage(val: float, total: int) -> float:
    """Safely calculate percentage rounded to 2 decimal places."""
    if total == 0:
        return 0.0
    return round((val / total) * 100, 2)
