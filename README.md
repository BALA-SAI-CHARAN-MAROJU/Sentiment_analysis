# Sentiment Analysis on Social Media Posts
### Final-Year B.Tech Major Project in Computer Science & Artificial Intelligence / Machine Learning

![Python](https://img.shields.io/badge/Python-3.14%20%7C%203.12%20%7C%203.10-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.x-black.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)
![Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)
![License](https://img.shields.io/badge/License-Academic%20Project-lightgrey.svg)

---

## 1. Project Overview

**Sentiment Analysis on Social Media Posts** is an end-to-end Machine Learning and Natural Language Processing web application designed to automatically collect, clean, classify, and visualize the sentiment polarity of user-generated social media posts into three canonical classes:

- **Positive** (Optimistic, enthusiastic, supportive)
- **Negative** (Critical, frustrated, unhappy)
- **Neutral** (Factual, informative, objective)

The system features automatic dataset acquisition, a social-aware NLP pipeline (preserving emojis, hashtags, and negations), multi-model benchmarking (**Logistic Regression**, **Multinomial Naive Bayes**, and **Linear Support Vector Machine** with probability calibration), local feature explainability, persistent SQLite history, bulk CSV prediction, and a modern dark/grey/white dashboard.

---

## 2. Key Features

- **Automatic Dataset Acquisition**: Automatically checks for datasets upon launch; if absent, automatically pulls the CardiffNLP `tweet_eval` 3-class sentiment corpus (or triggers a built-in offline social corpus if no internet is available).
- **Social NLP Preprocessing**:
  - Translates high-impact emojis (e.g. `😍` -> `love`, `😡` -> `angry`, `🔥` -> `fire awesome`) instead of discarding them.
  - Cleans user mentions (`@user`), URLs (`http://...`), HTML entities (`&amp;`), and normalizes repeated characters (`sooo good` -> `soo good`).
  - Negation-aware stop-word filtering (never removes `not`, `no`, `never`, `cannot`, etc.).
- **Multi-Model Benchmark**:
  - Model 1: **TF-IDF + Logistic Regression**
  - Model 2: **TF-IDF + Multinomial Naive Bayes**
  - Model 3: **TF-IDF + Linear Support Vector Machine (LinearSVC with CalibratedClassifierCV)**
  - Automated selection of the top performer based on **Weighted F1-Score**.
- **Model Explainability**: Highlights the most influential tokens/n-grams that triggered the predicted sentiment.
- **Modern Web Dashboard**:
  - **Dashboard**: Real-time KPI counters, sentiment distribution donut chart, live social feed classifier.
  - **Analyze Post**: Single-post prediction with preset buttons, confidence progress bar, and token pills.
  - **Bulk CSV Analysis**: Upload any CSV file, auto-detect text columns, classify batch in parallel, and export annotated CSVs.
  - **Dataset Analytics**: Post length distributions, high-frequency n-grams, and word clouds for each class.
  - **Model Performance**: Multi-model comparison table, confusion matrix heatmap, per-class precision/recall/F1 metrics.
  - **About Project**: Complete project architecture, Level 0-2 DFDs, mathematical formulas, and viva notes.
- **RESTful API**: Standardized JSON endpoints for single (`/api/predict`) and batch (`/api/bulk-predict`) inference.
- **SQLite Database**: Persistent queryable logging of prediction history with filtering by sentiment, keyword, and confidence.

---

## 3. System Architecture & Data Flow Diagrams

### Level 0: Context Data Flow Diagram
```
┌────────────────────────┐      Raw Post Text       ┌─────────────────────────────────┐
│                        │ ───────────────────────> │                                 │
│  User / Social Source  │                          │ 0.0 Sentiment Analysis System   │
│                        │ <─────────────────────── │                                 │
└────────────────────────┘    Predicted Sentiment   └─────────────────────────────────┘
                              & Confidence Score
```

### Level 1: System Subsystems
```
[Data Ingestion] ──(Raw Post)──> [1.0 Preprocessing] ──(Cleaned Text)──> [2.0 TF-IDF Vectorizer]
                                                                                │
                                                                           (Sparse Matrix)
                                                                                ▼
[Dashboard / API] <──(Results)── [4.0 SQLite DB] <── [3.0 Model Inference & Calibration]
```

### Level 2: Sentiment Classification & Explainability
```
Clean Text ──> [Tokenize & N-Grams] ──> [Compute TF-IDF: tf * log(N / df)]
                     │
                     ▼
          [Classifier Decision Function] ──> [Platt / Softmax Probability Calibration]
                     │
                     ├─> [Extract Top N-gram Weights (Explainability)]
                     └─> [Predicted Class (Pos/Neg/Neu) + Confidence] ──> SQLite Record
```

---

## 4. Project Directory Structure

```
c:/NNDL project/
│
├── app.py                      # Flask application & auto-initialization orchestrator
├── config.py                   # Central paths, hyperparameters, and environment settings
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive academic and practical documentation
├── .gitignore                  # Git ignore rules
├── .env.example                # Example environment variables
├── .env                        # Active environment configuration
│
├── data/
│   ├── raw/                    # Raw downloaded sentiment datasets
│   ├── processed/              # Normalized 3-class dataset (sentiment_dataset.csv)
│   ├── uploads/                # User uploaded CSV files
│   └── sample_social_posts.csv # Sample CSV for demonstration
│
├── models/
│   ├── sentiment_model.pkl     # Best trained classifier (Linear SVM)
│   ├── vectorizer.pkl          # Trained TF-IDF vectorizer
│   ├── metadata.json           # Model configuration, vocabulary size, best metrics
│   └── model_metrics.json      # Comparison metrics for all 3 models
│
├── database/
│   └── sentiment.db            # SQLite database with predictions history
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Automatic dataset checker, downloader, and normalizer
│   ├── preprocessing.py        # Social NLP cleaner (emojis, handles, URLs, negations)
│   ├── train.py                # Multi-model training, benchmark, and model selection
│   ├── evaluate.py             # Accuracy, Precision, Recall, F1, Confusion Matrix
│   ├── predict.py              # Single/batch inference and explainability engine
│   ├── social_media_fetcher.py # Modular free/public API collector (Reddit, Mastodon)
│   ├── database.py             # SQLite schema and query manager
│   ├── analytics.py            # Corpus EDA aggregations and word clouds
│   └── utils.py                # Logging, file sanitization, and JSON helpers
│
├── templates/
│   ├── base.html               # Sleek dark/grey/white layout, navbar, and footer
│   ├── index.html              # Interactive Dashboard with charts and live stream
│   ├── analyze.html            # Real-time text analysis with confidence meter
│   ├── bulk.html               # CSV upload, tabular output, and CSV export
│   ├── analytics.html          # Deep corpus statistics and sentiment word clouds
│   ├── performance.html        # Model benchmark table and confusion matrix
│   └── about.html              # Academic presentation, DFDs, and viva cheat sheet
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom dark/grey/white responsive stylesheet
│   ├── js/
│   │   └── main.js             # AJAX inference, Chart.js graphs, batch processing
│   └── images/                 # Generated word cloud PNGs
│
├── notebooks/
│   └── eda.ipynb               # Jupyter Notebook for lab and viva demonstrations
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py   # Unit tests for text cleaning and emojis
    ├── test_prediction.py      # Unit tests for model inference
    └── test_routes.py          # Integration tests for web and REST endpoints
```

---

## 5. Mathematical Formulations

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + \text{DF}(t)}\right) + 1$$
- $\text{TF}(t, d)$: Number of times term $t$ appears in post $d$.
- $|D|$: Total number of posts in the corpus.
- $\text{DF}(t)$: Number of posts containing term $t$.

### 2. Multinomial Naive Bayes
$$P(c \mid d) \propto P(c) \prod_{i=1}^{n} P(t_i \mid c)$$
With Laplace smoothing ($\alpha = 0.5$):
$$P(t \mid c) = \frac{\text{count}(t, c) + \alpha}{\sum_{w \in V} \text{count}(w, c) + \alpha |V|}$$

### 3. Linear Support Vector Machine
$$\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{N} \max(0, 1 - y_i(w^T x_i + b))$$
Calibrated using **Platt scaling** (cross-validated logistic regression on decision function outputs) to produce true posterior probability estimates.

### 4. Classification Metrics
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total Samples}}$$
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 6. Installation & Quickstart (Windows / VS Code)

### Prerequisites
- Python 3.10+ (Tested and certified on Python 3.14)
- Git (optional)

### Step 1: Open Terminal in VS Code
Open the project folder `c:\NNDL project` in VS Code (`Ctrl + ~` to open PowerShell terminal).

### Step 2: Create and Activate Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Run Application
```powershell
python app.py
```

The system automatically initializes:
```
[INFO] Checking dataset...
[INFO] Processed dataset found at data/processed/sentiment_dataset.csv.
[INFO] Checking sentiment models...
[INFO] Model artifacts found.
[INFO] Loading sentiment model into memory...
[INFO] Model loaded successfully.
[INFO] Initializing SQLite database...
[INFO] SQLite database initialized at database/sentiment.db
[INFO] System auto-initialization sequence complete.
🚀 Sentiment Analysis Web Application Live at: http://127.0.0.1:5000
```

### Step 5: Open Browser
Open your browser and navigate to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 7. Model Training & Evaluation Commands

To manually re-train the models and generate comparison metrics:

```powershell
python -m src.train
```

### Benchmark Results (cardiffnlp/tweet_eval):
| Model Architecture | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Calibrated)** | **61.15%** | **60.87%** | **61.15%** | **60.96%** | **Selected Best Model** |
| Logistic Regression | 60.75% | 60.70% | 60.75% | 60.72% | Evaluated |
| Multinomial Naive Bayes | 60.60% | 60.25% | 60.60% | 60.25% | Evaluated |

*Note: For a noisy 3-class social Twitter dataset, ~61% balanced accuracy without heavy multi-gigabyte neural nets represents strong statistical performance.*

---

## 8. REST API Usage

### 1. Single Post Prediction
- **Endpoint**: `POST /api/predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "text": "I absolutely love this new smartphone! The camera is stunning 😍✨ #tech #happy",
    "source": "API Test"
  }
  ```
- **Response**:
  ```json
  {
    "text": "I absolutely love this new smartphone! The camera is stunning 😍✨ #tech #happy",
    "cleaned_text": "absolutely love this new smartphone the camera is stunning love heart eyes sparkle brilliant happy",
    "sentiment": "Positive",
    "confidence": 0.8739,
    "probabilities": {
      "negative": 0.0826,
      "neutral": 0.0435,
      "positive": 0.8739
    },
    "explainability": {
      "top_words": [
        { "word": "love", "score": 0.4145 },
        { "word": "camera", "score": 0.3946 },
        { "word": "brilliant", "score": 0.3835 },
        { "word": "absolutely", "score": 0.357 },
        { "word": "eyes", "score": 0.357 }
      ]
    },
    "model_used": "Linear SVM"
  }
  ```

### 2. Bulk Prediction
- **Endpoint**: `POST /api/bulk-predict`
- **Request Body**:
  ```json
  {
    "texts": [
      "Loved this so much! 😍",
      "Horrible delay and broken product 😡"
    ]
  }
  ```

### 3. Real-Time Social Media Fetcher Endpoint
- **Endpoint**: `GET /api/fetch-live`
- **Query Parameters**:
  - `source`: `reddit`, `mastodon`, `youtube`, or `all` (default: `all`)
  - `limit`: Number of posts to retrieve (default: `10`)
  - `subreddit`: Target subreddit for Reddit queries (e.g. `technology`, `programming`, `gaming`)
  - `video_id`: Target video ID for YouTube queries (e.g. `dQw4w9WgXcQ`)
- **Response**:
  ```json
  {
    "source": "all",
    "count": 10,
    "summary": {
      "positive": 5,
      "negative": 2,
      "neutral": 3
    },
    "posts": [
      {
        "text": "The new battery optimization update for laptops has doubled my battery life! 😍🔋",
        "source": "Reddit (r/technology)",
        "sentiment": "Positive",
        "confidence": 0.8739,
        "timestamp": "2026-10-05 18:20:00",
        "url": "https://reddit.com/r/technology/comments/..."
      }
    ]
  }
  ```

### 4. Statistics Endpoint
- **Endpoint**: `GET /api/stats`
- Returns total counts and positive/negative/neutral percentages.

---

## 9. Running Tests

Run the full automated test suite (24 unit and integration tests):

```powershell
python -m pytest -v tests/
```

Test coverage includes:
- Text normalization & emoji keyword translation
- Negation word retention (`not`, `no`, `cannot`)
- Model inference & probability bounds
- Flask web routes (`/`, `/analyze`, `/bulk`, `/live-feed`, `/analytics`, `/performance`, `/about`)
- REST APIs (`/api/predict`, `/api/fetch-live`, `/api/stats`, `/api/history`)
- CSV bulk upload & parsing

---

## 10. Optional Social Media APIs (.env)

The application works completely out-of-the-box using free public endpoints. If you wish to connect your own credentials, configure the `.env` file:

```env
# Optional Reddit API Credentials (PRAW)
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=SentimentAnalyzer/1.0

# Optional YouTube Data API v3
YOUTUBE_API_KEY=your_google_api_key_here

# Mastodon Instance Configuration (Default: https://mastodon.social)
MASTODON_INSTANCE=https://mastodon.social
MASTODON_ACCESS_TOKEN=your_token_here
```

If left blank, the app uses public unauthenticated endpoints and seamless fallback social streams.

---

## 11. Frequently Asked Questions / Common Errors

### Q1: Emojis cause a UnicodeEncodeError on Windows terminal.
- **Fix**: The codebase includes `sys.stdout.reconfigure(encoding='utf-8')` across entrypoints. If running custom scripts, set `$env:PYTHONIOENCODING="utf-8"`.

### Q2: How do I train on the entire 45,000+ tweet dataset instead of 10,000 rows?
- Change `MAX_TRAINING_ROWS=0` in `.env` and re-run `python -m src.train`.

### Q3: How do I test CSV upload during the project demonstration?
- Go to `http://127.0.0.1:5000/bulk` and upload the prepared demo file: `data/sample_social_posts.csv`.

---

## 12. Viva Voce Presentation Guide

1. **Why is Sentiment Analysis on Social Media harder than on formal reviews (e.g. IMDb)?**
   - Social posts are short, unstructured, full of sarcasm, emojis, misspellings, and hashtags. Context is scarce, making feature extraction through n-grams and emoji mapping crucial.
2. **Why Linear SVM with Platt Calibration?**
   - In high-dimensional sparse text vectors (thousands of TF-IDF features), classes are often linearly separable. Linear SVM finds the maximum-margin hyperplane, minimizing structural risk. Platt scaling (calibrated classifier) transforms raw decision distances into reliable probabilities.
3. **What is data leakage and how did you prevent it?**
   - Data leakage happens when information from outside the training dataset is used to create the model. We split the data before fitting the TF-IDF vectorizer so the test set's vocabulary and frequencies remain completely unseen during training.
#   S e n t i m e n t _ a n a l y s i s  
 