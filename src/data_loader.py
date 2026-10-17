"""
src/data_loader.py - Automatic Dataset Acquisition & Preprocessing Pipeline
Checks for local sentiment data; if absent, automatically acquires the
CardiffNLP tweet_eval social sentiment dataset or generates a rich offline
benchmark dataset, standardizes columns to 'text' and 'sentiment',
cleans the text, and persists both raw and processed versions.
"""

from pathlib import Path
import random
from typing import Optional, Tuple
import pandas as pd

from config import (
    CLASSES,
    LABEL_MAPPING,
    MAX_TRAINING_ROWS,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
)
from src.preprocessing import preprocess_dataframe
from src.utils import logger


def generate_offline_fallback_dataset() -> pd.DataFrame:
    """
    Generate a diverse, realistic social media dataset across positive,
    negative, and neutral sentiment categories for offline or restricted environments.
    """
    logger.info("Generating robust offline social media sentiment dataset...")

    positives = [
        "I absolutely love this new smartphone! The camera is stunning 😍✨ #tech #happy",
        "Had such an amazing time at the concert last night! Best experience ever ❤️🎉",
        "Kudos to the customer support team! Solved my issue in five minutes 🙌💯",
        "Just got promoted today! So grateful and thrilled for what's coming next 🚀😊",
        "The food at this new Italian place was delicious, definitely coming back! 🍕🤤",
        "This software update made my laptop run twice as fast. Brilliant job! 👏",
        "Loving the sunny weather today, perfect day for a walk in the park ☀️🌸",
        "Finally finished reading this masterpiece of a book. Highly recommend it to everyone! 📚❤️",
        "The battery life on this gadget is unbelievable. Lasts two full days! 🔋⚡",
        "Genuinely impressed with how smooth this web app is. Great UX design! 👍",
        "Just watched the season finale and it blew my mind! Absolutely fantastic writing 🎬🔥",
        "Received my order two days earlier than expected, excellent service! 📦✨",
        "So proud of my team for delivering this project on time. Great teamwork! 🏆",
        "Waking up to this morning view is pure bliss. Good morning world! 🌄🥰",
        "This new song is on repeat all day! Such good vibes 🎶💃",
        "Super excited for the weekend trip with my best friends! ✈️🏖️",
        "Found a great coffee shop downtown with the coziest atmosphere ☕❤️",
        "Big shoutout to everyone who supported our open source release! Thank you all 🙏",
        "Everything about this vacation has exceeded my expectations! Loving every second 🌴😎",
        "What a thrilling match! Our team played with pure heart and won ⚽🔥",
    ]

    negatives = [
        "This is the worst customer service I have ever experienced. Absolutely unacceptable 😡👎",
        "The new update completely broke my application. It keeps crashing every minute! 🤬",
        "Delayed flight for the third time today. Terrible communication from the airline ✈️💔",
        "Ordered food two hours ago and it still hasn't arrived. Freezing cold when it got here 🤮",
        "The build quality on this laptop is horrible. Screen cracked within a week 💀",
        "Don't waste your money on this product. It stopped working after two days 👎",
        "I am so angry and disappointed with how they handled this refund request 😤",
        "Another pointless meeting that could have easily been a two sentence email 🤦‍♂️",
        "The battery drains from 100% to 10% in just two hours. Total scam! 🔋❌",
        "Stuck in massive traffic for hours in the pouring rain. Horrible day 🌧️🚗",
        "Extremely rude staff at the checkout counter. Never shopping here again 😠",
        "The website is down again during the sale. Utterly pathetic infrastructure 📉",
        "I feel so sick after eating at that diner. Awful hygiene standards 🤢",
        "Why did they ruin the user interface with this awful redesign? It makes zero sense 🙄",
        "Canceled my subscription today. Price went up by 50% while quality plummeted 📉",
        "Worst movie of the year. Boring plot, terrible acting, total waste of time 🥱",
        "Internet has been dropping every 10 minutes all week. Fix your network! 📶❌",
        "The package arrived completely torn and the item inside was damaged 📦💔",
        "My train got canceled with no prior notice and no replacement. Stuck here! 🚆😡",
        "Terrible experience from start to finish. Zero stars if I could give it 👎",
    ]

    neutrals = [
        "The press conference is scheduled to begin at 10 AM tomorrow morning 🕙",
        "Apple announced their quarterly financial earnings report earlier today 📊",
        "The package was delivered to the front porch at 3:15 PM 📦",
        "The train departs from platform four at quarter past noon 🚆",
        "Tomorrow's weather forecast predicts scattered clouds with a high of 22 degrees ⛅",
        "The software update version 4.2.1 is now available for download 💻",
        "The store is open from Monday through Saturday, 9 AM to 8 PM 🏪",
        "The recipe requires two cups of flour, one egg, and half a cup of milk 🥣",
        "Traffic is moving at average speed along the interstate highway 🚗",
        "The meeting minutes have been uploaded to the shared project drive 📁",
        "The library will remain closed on national holidays according to the schedule 📚",
        "The flight duration from Chicago to Denver is approximately two hours and thirty minutes ✈️",
        "Stock markets closed relatively flat ahead of the central bank meeting 📈",
        "The university opened registration for the fall semester today 🎓",
        "A regular maintenance window is planned for Sunday between 2 AM and 4 AM ⚙️",
        "The bus stops at every major junction along the circular route 🚌",
        "Results for the midterm examination will be published on the portal next week 📝",
        "The museum exhibition is open to the public free of charge on the first Sunday 🏛️",
        "Users can reset their password by following the verification link sent via email ✉️",
        "The team announced their final squad lineup for the upcoming tournament 📋",
    ]

    # Combine and multiply with subtle noise to create a sizable demo dataset
    records = []
    modifiers = [
        "", " Honestly,", " Update:", " Quick note:", " FYI:",
        " Just in:", " News:", " Today:", " Random thought:"
    ]

    for _ in range(50):
        for p in positives:
            mod = random.choice(modifiers)
            records.append({"text": f"{mod} {p}".strip(), "sentiment": "positive"})
        for n in negatives:
            mod = random.choice(modifiers)
            records.append({"text": f"{mod} {n}".strip(), "sentiment": "negative"})
        for neu in neutrals:
            mod = random.choice(modifiers)
            records.append({"text": f"{mod} {neu}".strip(), "sentiment": "neutral"})

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} sample social media records.")
    return df


def download_dataset() -> pd.DataFrame:
    """
    Attempt downloading the CardiffNLP tweet_eval sentiment dataset from Hugging Face.
    If the network is unavailable or Hugging Face fails, seamlessly falls back
    to the offline curated social dataset.
    """
    logger.info("[INFO] Checking dataset availability...")

    try:
        from datasets import load_dataset

        logger.info("[INFO] Attempting to download CardiffNLP tweet_eval sentiment dataset...")
        # Load train split; CardiffNLP tweet_eval has 0: negative, 1: neutral, 2: positive
        hf_dataset = load_dataset(
            "cardiffnlp/tweet_eval",
            "sentiment",
            split="train",
            trust_remote_code=False,
        )

        df = hf_dataset.to_pandas()
        logger.info(f"[INFO] Successfully downloaded {len(df)} records from Hugging Face.")

        label_to_name = {0: "negative", 1: "neutral", 2: "positive"}
        df["sentiment"] = df["label"].map(label_to_name)
        df = df[["text", "sentiment"]]

    except Exception as e:
        logger.warning(
            f"[WARNING] Could not fetch online dataset via Hugging Face ({e}). "
            "Falling back to built-in social media sentiment corpus."
        )
        df = generate_offline_fallback_dataset()

    return df


def get_dataset(force_reload: bool = False) -> pd.DataFrame:
    """
    Main entry point for loading the dataset.
    1. Checks if processed data exists and loads it.
    2. Otherwise, loads or downloads raw data, preprocesses it, and caches it.
    """
    if PROCESSED_DATA_FILE.exists() and not force_reload:
        logger.info(f"[INFO] Loading processed dataset from {PROCESSED_DATA_FILE}...")
        df = pd.read_csv(PROCESSED_DATA_FILE)
        logger.info(f"[INFO] Loaded {len(df)} processed rows.")
        return df

    # If raw data already exists locally, use it; otherwise download
    if RAW_DATA_FILE.exists() and not force_reload:
        logger.info(f"[INFO] Found existing raw data at {RAW_DATA_FILE}. Loading...")
        df_raw = pd.read_csv(RAW_DATA_FILE)
    else:
        logger.info("[INFO] Raw dataset not found. Initiating automatic acquisition...")
        df_raw = download_dataset()
        RAW_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(RAW_DATA_FILE, index=False)
        logger.info(f"[INFO] Saved raw dataset to {RAW_DATA_FILE}.")

    # Preprocess
    logger.info("[INFO] Preprocessing dataset...")
    df_processed = preprocess_dataframe(df_raw, text_col="text", sentiment_col="sentiment")

    # Limit training rows if configured to avoid slow runs on low-spec hardware
    if MAX_TRAINING_ROWS > 0 and len(df_processed) > MAX_TRAINING_ROWS:
        logger.info(
            f"[INFO] Subsampling dataset to {MAX_TRAINING_ROWS} rows "
            f"(MAX_TRAINING_ROWS configured) with stratified class representation."
        )
        subsets = []
        per_class = MAX_TRAINING_ROWS // len(CLASSES)
        for cls in CLASSES:
            cls_df = df_processed[df_processed["sentiment"] == cls]
            if not cls_df.empty:
                n_sample = min(len(cls_df), per_class)
                subsets.append(cls_df.sample(n=n_sample, random_state=42))
        if subsets:
            df_processed = pd.concat(subsets, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    PROCESSED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(PROCESSED_DATA_FILE, index=False)
    logger.info(
        f"[INFO] Processed dataset saved to {PROCESSED_DATA_FILE} ({len(df_processed)} rows)."
    )

    return df_processed


if __name__ == "__main__":
    df = get_dataset()
    print("Class Distribution:")
    print(df["sentiment"].value_counts())
