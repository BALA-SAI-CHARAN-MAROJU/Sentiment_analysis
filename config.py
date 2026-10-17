"""
config.py - Central Configuration Module
Provides directory paths, model hyperparameters, dataset configuration,
and environment settings for the Sentiment Analysis application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if present
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# --- Directory Paths ---
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
UPLOADS_DIR = DATA_DIR / "uploads"

MODELS_DIR = BASE_DIR / "models"
DATABASE_DIR = BASE_DIR / "database"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure runtime directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, UPLOADS_DIR, MODELS_DIR, DATABASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# --- File Paths ---
RAW_DATA_FILE = RAW_DATA_DIR / "raw_social_sentiment.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "sentiment_dataset.csv"
SAMPLE_LIVE_POSTS_FILE = DATA_DIR / "sample_live_posts.json"

MODEL_FILE = MODELS_DIR / "sentiment_model.pkl"
VECTORIZER_FILE = MODELS_DIR / "vectorizer.pkl"
METADATA_FILE = MODELS_DIR / "metadata.json"
METRICS_FILE = MODELS_DIR / "model_metrics.json"

DATABASE_FILE = DATABASE_DIR / "sentiment.db"

# --- Training & Preprocessing Settings ---
MAX_TRAINING_ROWS = int(os.getenv("MAX_TRAINING_ROWS", "10000"))
RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_FEATURES = 6000
NGRAM_RANGE = (1, 2)

# Classes and Canonical Label Mappings
CLASSES = ["negative", "neutral", "positive"]
LABEL_MAPPING = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    0: "negative",
    1: "neutral",
    2: "positive",
}

# --- Optional Transformer Settings ---
USE_TRANSFORMER = os.getenv("USE_TRANSFORMER", "false").lower() in ("true", "1", "yes")
TRANSFORMER_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# --- Web App & Security Settings ---
SECRET_KEY = os.getenv("SECRET_KEY", "btech_sentiment_analysis_secure_key_2026")
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size
ALLOWED_EXTENSIONS = {"csv"}

# --- Live Social Media API Credentials & Endpoints ---
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "SocialSentimentAnalyzer/1.0")
MASTODON_INSTANCE = os.getenv("MASTODON_INSTANCE", "https://mastodon.social").rstrip("/")
MASTODON_ACCESS_TOKEN = os.getenv("MASTODON_ACCESS_TOKEN", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
