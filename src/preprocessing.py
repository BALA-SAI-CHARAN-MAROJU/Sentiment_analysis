"""
src/preprocessing.py - Text Preprocessing Pipeline
Provides comprehensive cleaning tailored for social media posts, including:
- Emoji preservation and textual conversion (e.g., 😍 -> love, 😡 -> angry)
- Removal of URLs, user mentions, and HTML entities
- Hashtag text extraction and special character normalization
- Negation-aware stop-word removal and optional lemmatization
- Batch DataFrame preprocessing and deduplication
"""

import html
import re
import unicodedata
from typing import Optional, Set
import pandas as pd

# Optional NLTK dependencies with offline fallback
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    _NLTK_STOPWORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
except Exception:
    _NLTK_STOPWORDS = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
        "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
        "the", "and", "if", "or", "because", "as", "until", "while", "of", "at",
        "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to", "from", "up", "down",
        "in", "out", "on", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "only",
        "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just"
    }
    _LEMMATIZER = None

# Crucial negation & sentiment words to NEVER remove during stopword filtering
NEGATION_WORDS: Set[str] = {
    "not", "no", "nor", "neither", "never", "none", "hardly", "scarcely",
    "barely", "cannot", "cant", "couldnt", "didnt", "doesnt", "dont",
    "hadnt", "hasnt", "havent", "isnt", "mustnt", "neednt", "shouldnt",
    "wasnt", "werent", "wont", "wouldnt", "against", "without"
}

# Curated high-impact social media emojis to descriptive sentiment tokens
EMOJI_MAP = {
    "❤️": " love ", "💖": " love ", "💕": " love ", "😍": " love heart eyes ",
    "🥰": " adored ", "😘": " kiss love ", "😊": " happy smiling ", "😁": " grin happy ",
    "😀": " happy ", "😃": " joyful ", "😄": " excited cheerful ", "🎉": " celebration yay ",
    "🔥": " fire awesome ", "✨": " sparkle brilliant ", "💯": " perfect hundred ",
    "👍": " thumbs up good ", "👏": " applause bravo ", "🙌": " celebration cheering ",
    "😎": " cool confident ", "🤩": " starstruck impressed ", "🥳": " party celebrate ",
    "😡": " angry furious ", "🤬": " cursing enraged ", "😠": " mad irritated ",
    "👿": " evil angry ", "😤": " frustrated grumpy ", "💔": " heartbreak sad ",
    "😭": " crying loudly distressed ", "😢": " sad tear ", "😞": " disappointed gloomy ",
    "😔": " pensive depressed ", "😒": " unamused disgusted ", "🤮": " disgust vomit ",
    "🤢": " nauseous sick ", "👎": " thumbs down bad ", "💩": " crap terrible ",
    "💀": " dead dying ", "🤦": " facepalm annoyed ", "🙄": " eye roll skeptical ",
    "😐": " neutral poker face ", "😑": " expressionless indifferent ",
    "🤔": " thinking curious ", "🧐": " inquiring examining ", "🤷": " shrug indifferent ",
    "😶": " speechless silent ", "😴": " sleepy boring ", "🥱": " yawning bored "
}

# Optional emoji package demojize support
try:
    import emoji
    _HAS_EMOJI = True
except ImportError:
    _HAS_EMOJI = False


def convert_emojis(text: str) -> str:
    """
    Convert emojis to descriptive sentiment words.
    First checks the curated high-impact mapping, then converts remaining emojis.
    """
    for emo, meaning in EMOJI_MAP.items():
        text = text.replace(emo, meaning)

    if _HAS_EMOJI:
        # demojize remaining emojis like :sparkles: -> ' sparkles '
        text = emoji.demojize(text, delimiters=(" ", " "))
        text = text.replace("_", " ")

    return text


def clean_text(
    text: Optional[str],
    remove_stopwords: bool = False,
    lemmatize: bool = False
) -> str:
    """
    Execute comprehensive social media text cleaning.

    Steps:
    1. Handle empty / None input
    2. Convert emojis to sentiment words
    3. Decode HTML entities (&amp;, &lt;, etc.)
    4. Remove URLs (http, https, www)
    5. Remove user handles (@username)
    6. Extract hashtag text (#awesome -> awesome)
    7. Remove non-alphabetical special characters and numbers
    8. Normalize repeated characters (e.g. 'loooove' -> 'love')
    9. Lowercase conversion and whitespace strip
    10. Optional stopword removal (protecting negations)
    11. Optional lemmatization
    """
    if text is None:
        return ""

    text = str(text)

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # 2. Decode HTML entities
    text = html.unescape(text)

    # 3. Handle emojis before stripping characters
    text = convert_emojis(text)

    # 4. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 5. Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # 6. Remove user mentions (@username)
    text = re.sub(r"@\w+", " ", text)

    # 7. Extract hashtag text without hash symbol
    text = re.sub(r"#(\w+)", r" \1 ", text)

    # 8. Lowercase
    text = text.lower()

    # 9. Handle contractions / slang with word boundaries
    slang_map = {
        r"\bcan't\b": "cannot", r"\bwon't\b": "will not", r"\bn't\b": " not",
        r"\b're\b": " are", r"\b's\b": " is", r"\b'd\b": " would", r"\b'll\b": " will",
        r"\b've\b": " have", r"\b'm\b": " am", r"\bu\b": "you", r"\bur\b": "your",
        r"\br\b": "are", r"\bthx\b": "thanks", r"\bpls\b": "please", r"\bplz\b": "please",
        r"\bimo\b": "in my opinion", r"\btbh\b": "to be honest",
        # Colloquial social media idioms / slang
        r"\ball this shit\b": "all this stuff",
        r"\ball that shit\b": "all that stuff",
        r"\ba lot of shit\b": "a lot of stuff",
        r"\blot of shit\b": "lot of stuff",
        r"\bthis shit\b": "this stuff",
        r"\bthat shit\b": "that stuff",
        r"\bsome shit\b": "some stuff",
        r"\bgood shit\b": "good stuff",
        r"\bcool shit\b": "cool stuff",
        r"\bthe shit\b": "the best",
        r"\bholy shit\b": "wow",
        r"\bno shit\b": "obviously",
    }
    for pattern, expansion in slang_map.items():
        text = re.sub(pattern, expansion, text)

    # 10. Normalize character elongation (e.g., 'sooooo goooood' -> 'soo good')
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 11. Keep letters and spaces (remove numbers and punctuation symbols)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 12. Tokenize by whitespace
    tokens = text.split()

    # 13. Optional Stop-word removal (protecting negations)
    if remove_stopwords:
        clean_stop = _NLTK_STOPWORDS - NEGATION_WORDS
        tokens = [t for t in tokens if t not in clean_stop and len(t) > 1]
    else:
        # Keep tokens with length > 1
        tokens = [t for t in tokens if len(t) > 1]

    # 14. Optional Lemmatization
    if lemmatize and _LEMMATIZER is not None:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    # 15. Recombine and normalize whitespace
    cleaned = " ".join(tokens).strip()
    return cleaned


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    sentiment_col: str = "sentiment",
    remove_stopwords: bool = False,
    lemmatize: bool = False
) -> pd.DataFrame:
    """
    Clean and filter an entire DataFrame for model ingestion.
    - Standardizes column names
    - Drops null / empty values
    - Cleans social text
    - Standardizes sentiment labels to 'negative', 'neutral', 'positive'
    - Drops duplicates
    """
    df = df.copy()

    # Verify column existence
    if text_col not in df.columns:
        raise ValueError(f"Required text column '{text_col}' not found in dataframe.")

    # Drop nulls in text
    df = df.dropna(subset=[text_col])

    # Clean text column
    df["cleaned_text"] = df[text_col].apply(
        lambda t: clean_text(t, remove_stopwords=remove_stopwords, lemmatize=lemmatize)
    )

    # Filter out empty or whitespace-only cleaned texts
    df = df[df["cleaned_text"].str.strip().str.len() > 2]

    # Standardize sentiment column if present
    if sentiment_col in df.columns:
        df = df.dropna(subset=[sentiment_col])
        df["sentiment"] = df[sentiment_col].astype(str).str.strip().str.lower()

        # Map common label variants (numeric, shorthand, etc.)
        sentiment_normalization_map = {
            "0": "negative", "neg": "negative", "negative": "negative", "0.0": "negative",
            "1": "neutral", "neu": "neutral", "neutral": "neutral", "1.0": "neutral",
            "2": "positive", "pos": "positive", "positive": "positive", "2.0": "positive",
            "4": "positive"  # Sentiment140 convention (0=neg, 4=pos)
        }
        df["sentiment"] = df["sentiment"].map(sentiment_normalization_map)
        df = df.dropna(subset=["sentiment"])
        df = df[df["sentiment"].isin(["negative", "neutral", "positive"])]

    # Drop duplicate cleaned texts
    df = df.drop_duplicates(subset=["cleaned_text"])

    return df.reset_index(drop=True)
