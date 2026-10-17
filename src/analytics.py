"""
src/analytics.py - Exploratory Data Analysis & Corpus Analytics Module
Computes detailed vocabulary distributions, class percentages, n-gram frequencies,
post length statistics, and generates word clouds for sentiment classes.
"""

from collections import Counter
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config import PROCESSED_DATA_FILE, STATIC_DIR
from src.data_loader import get_dataset
from src.utils import logger

# Output directory for analytics charts / word clouds
IMAGES_DIR = STATIC_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def get_dataset_analytics(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Compute comprehensive dataset statistics for the analytics dashboard:
    - Total posts
    - Sentiment class counts and percentages
    - Post character & word length distributions
    - Top words overall and per sentiment class
    - Top hashtags
    """
    if df is None:
        df = get_dataset()

    total_posts = len(df)
    if total_posts == 0:
        return {"total_posts": 0}

    # Class distribution
    class_counts = df["sentiment"].value_counts().to_dict()
    positive_count = class_counts.get("positive", 0)
    negative_count = class_counts.get("negative", 0)
    neutral_count = class_counts.get("neutral", 0)

    # Post length statistics
    df["char_len"] = df["text"].astype(str).str.len()
    df["word_count"] = df["text"].astype(str).str.split().str.len()

    length_stats = {
        "avg_chars": round(float(df["char_len"].mean()), 1),
        "median_chars": int(df["char_len"].median()),
        "max_chars": int(df["char_len"].max()),
        "min_chars": int(df["char_len"].min()),
        "avg_words": round(float(df["word_count"].mean()), 1),
        "median_words": int(df["word_count"].median()),
    }

    # Top hashtags from original text
    hashtag_counter = Counter()
    for text in df["text"].astype(str):
        tags = re.findall(r"#(\w+)", text)
        for tag in tags:
            hashtag_counter[tag.lower()] += 1

    top_hashtags = [
        {"tag": f"#{tag}", "count": count}
        for tag, count in hashtag_counter.most_common(12)
    ]

    # Stopwords filter for vocabulary analysis
    stopword_set = {
        "the", "and", "to", "of", "in", "it", "is", "for", "on", "you", "my",
        "that", "at", "with", "this", "me", "was", "have", "are", "be", "so",
        "but", "not", "just", "from", "all", "user", "rt", "http", "https",
        "co", "am", "or", "as", "an", "we", "if", "out", "about", "what",
        "up", "by", "like", "will", "your", "they", "one", "get", "has", "can"
    }

    def get_top_words_for_slice(texts: pd.Series, top_n: int = 15) -> List[Dict[str, Any]]:
        word_counter = Counter()
        for t in texts:
            words = str(t).lower().split()
            for w in words:
                cleaned_word = re.sub(r"[^a-zA-Z]", "", w)
                if len(cleaned_word) > 2 and cleaned_word not in stopword_set:
                    word_counter[cleaned_word] += 1
        return [{"word": w, "count": c} for w, c in word_counter.most_common(top_n)]

    top_words_overall = get_top_words_for_slice(df["cleaned_text"], top_n=15)
    top_words_positive = get_top_words_for_slice(
        df[df["sentiment"] == "positive"]["cleaned_text"], top_n=15
    )
    top_words_negative = get_top_words_for_slice(
        df[df["sentiment"] == "negative"]["cleaned_text"], top_n=15
    )
    top_words_neutral = get_top_words_for_slice(
        df[df["sentiment"] == "neutral"]["cleaned_text"], top_n=15
    )

    analytics_data = {
        "total_posts": total_posts,
        "class_distribution": {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": neutral_count,
            "positive_pct": round((positive_count / total_posts) * 100, 1),
            "negative_pct": round((negative_count / total_posts) * 100, 1),
            "neutral_pct": round((neutral_count / total_posts) * 100, 1),
        },
        "length_stats": length_stats,
        "top_hashtags": top_hashtags,
        "top_words_overall": top_words_overall,
        "top_words_positive": top_words_positive,
        "top_words_negative": top_words_negative,
        "top_words_neutral": top_words_neutral,
    }

    return analytics_data


def generate_wordclouds(df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """
    Generate word cloud images for positive, negative, and neutral classes.
    Saves PNG files into static/images/ and returns the relative URLs.
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        logger.warning("WordCloud package not installed. Skipping word cloud generation.")
        return {}

    if df is None:
        df = get_dataset()

    paths = {}
    color_schemes = {
        "positive": ("Greens", "#00ff88"),
        "negative": ("Reds", "#ff4444"),
        "neutral": ("Blues", "#888888"),
    }

    for sentiment in ["positive", "negative", "neutral"]:
        subset = df[df["sentiment"] == sentiment]
        text_corpus = " ".join(subset["cleaned_text"].astype(str).tolist())
        if not text_corpus.strip():
            continue

        output_path = IMAGES_DIR / f"wordcloud_{sentiment}.png"
        colormap, _ = color_schemes.get(sentiment, ("viridis", "#ffffff"))

        wc = WordCloud(
            width=600,
            height=320,
            background_color="#121212",
            colormap=colormap,
            max_words=80,
            contour_width=0,
            random_state=42,
        ).generate(text_corpus)

        wc.to_file(str(output_path))
        paths[sentiment] = f"/static/images/wordcloud_{sentiment}.png"
        logger.info(f"Generated word cloud for {sentiment} at {output_path}")

    return paths


if __name__ == "__main__":
    stats = get_dataset_analytics()
    print("Total:", stats["total_posts"])
    print("Class Dist:", stats["class_distribution"])
    generate_wordclouds()
