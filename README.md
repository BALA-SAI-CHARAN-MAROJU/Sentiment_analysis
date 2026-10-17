# Sentify — Social Media Sentiment Analysis & Real-Time Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Tests-30%20Passed-2ea44f?logo=pytest&logoColor=white)](#10-automated-testing--quality-assurance)
[![Architecture](https://img.shields.io/badge/Architecture-Modular%20MVC%20%2B%20PyTorch-blueviolet)](#3-system-architecture--data-flow-diagrams)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning, Deep Learning, Natural Language Processing, and Web Intelligence system engineered to ingest, clean, classify, explain, and visualize the sentiment polarity of user-generated social media posts across three canonical categories:

$$\text{Class} \in \{\textbf{Positive}, \textbf{Neutral}, \textbf{Negative}\}$$

Built as an academic and production-ready Neural Networks & Deep Learning (NNDL) capstone project featuring a social-aware NLP preprocessing pipeline, multi-model benchmark tournament (**Linear SVM**, **Logistic Regression**, **Naive Bayes**, **Multi-Layer Perceptron (NN)**, and **PyTorch Deep Neural Network**), Platt-calibrated probability scoring, local token explainability, live social stream ingestion (Reddit, Mastodon, YouTube), bulk CSV batch processing, persistent SQLite analytics, and an interactive dark-themed web dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key System Features](#2-key-system-features)
3. [System Architecture & Data Flow Diagrams](#3-system-architecture--data-flow-diagrams)
4. [Project Directory Structure](#4-project-directory-structure)
5. [Mathematical & Theoretical Formulations](#5-mathematical--theoretical-formulations)
6. [Model Benchmark & Performance Comparison](#6-model-benchmark--performance-comparison)
7. [Installation & Quickstart Guide](#7-installation--quickstart-guide)
8. [Interactive Web Dashboard Walkthrough](#8-interactive-web-dashboard-walkthrough)
9. [RESTful API Documentation](#9-restful-api-documentation)
10. [Automated Testing & Quality Assurance](#10-automated-testing--quality-assurance)
11. [Configuration & Optional API Integrations](#11-configuration--optional-api-integrations)
12. [Academic Viva Voce & Defense Q&A](#12-academic-viva-voce--defense-qa)
13. [Troubleshooting & FAQ](#13-troubleshooting--faq)
14. [Contributing & Authors](#14-contributing--authors)

---

## 1. Project Overview

Social media content is notoriously noisy, dense with slang, acronyms, emojis, hashtags, and unpunctuated syntax. Traditional off-the-shelf sentiment analyzers fail on social micro-text because they discard punctuation, misinterpret negations, or strip emotional cues like emojis and repeated letters.

**Sentify** resolves these challenges through a targeted social NLP pipeline combined with a multi-model benchmarking suite:
- **Zero-Friction Startup**: Automatically downloads and prepares the `cardiffnlp/tweet_eval` social sentiment benchmark corpus or falls back to an offline social dataset if an internet connection is unavailable.
- **Social-Aware Token Engineering**: Emojis are translated to emotive lexemes (`😍` $\rightarrow$ `love heart eyes`, `😡` $\rightarrow$ `angry`), negations are preserved, and handles/URLs are normalized.
- **Deep Learning & Statistical Tournament**: Evaluates **Linear Support Vector Machines (LinearSVC)** with Platt calibration, **Logistic Regression**, **Multinomial Naive Bayes**, **Multi-Layer Perceptrons (MLP)**, and **PyTorch Deep Neural Networks** (`SentimentNeuralNetwork` with Batch Normalization and Dropout).
- **Explainable AI (XAI)**: Surfaces the highest-weighted n-grams driving each prediction to give users transparent, interpretable decisions.

---

## 2. Key System Features

| Feature | Description |
| :--- | :--- |
| **Social NLP Preprocessing** | Cleans URLs, HTML entities, and user mentions (`@user`), expands emotional emojis, compresses elongated characters (`sooo good` $\rightarrow$ `soo good`), and preserves grammatical negations (`not`, `never`, `cannot`). |
| **Deep Learning & ML Tournament** | Trains and cross-evaluates **Linear SVM**, **Logistic Regression**, **Multinomial Naive Bayes**, **Multi-Layer Perceptron (NN)**, and **PyTorch Deep Neural Network** on identical TF-IDF feature matrices, saving the top-performing model ranked by **Weighted F1-Score**. |
| **PyTorch Neural Network Engine** | Deep feedforward neural network (`src/neural_network.py`) featuring fully connected linear layers, Batch Normalization (`BatchNorm1d`), ReLU activations, Dropout regularization (0.3), AdamW optimization, and Scikit-Learn wrapper compatibility. |
| **Platt Probability Calibration** | Wraps linear maximum-margin classifiers with `CalibratedClassifierCV` (sigmoid scaling) to provide true confidence scores rather than uncalibrated margin distances. |
| **Local Token Explainability** | Ranks and displays the exact tokens and n-grams that steered the prediction toward Positive, Negative, or Neutral polarity. |
| **Live Social Media Stream** | Ingests real-time public posts from Reddit, Mastodon, and YouTube with sentiment tags, confidence meters, and original post URLs. |
| **Bulk CSV Batch Processing** | Drag-and-drop CSV dataset uploads with automated column detection, batch inference, dynamic table pagination, and one-click annotated CSV exports. |
| **Exploratory Data Analytics (EDA)** | Dynamic sentiment distribution charts, character and word-count histograms, high-frequency n-grams, and pre-rendered word clouds for each sentiment class. |
| **Persistent SQLite Audit Log** | Records every single and batch prediction in a local SQLite database with search, filter (by sentiment, confidence, keyword), and history management. |
| **Comprehensive REST API** | Fully documented JSON endpoints for single inference, bulk classification, live social feeds, statistics, and prediction history. |

---

## 3. System Architecture & Data Flow Diagrams

### Level 0: Context Data Flow Diagram (DFD)

```
                     ┌────────────────────────────────┐
                     │   User / Client / REST API     │
                     └──────┬──────────────────▲──────┘
                            │                  │
                Raw Post /  │                  │ Predicted Sentiment,
                CSV Batch   │                  │ Confidence & Explainability
                            ▼                  │
              ┌────────────────────────────────┴───────────────┐
              │                                               │
              │   0.0 Sentify Sentiment Intelligence Engine   │
              │                                               │
              └───────────────────────┬───────────────────────┘
                                      │
                         Read / Write │ Prediction Records
                                      ▼
                           [( SQLite Database )]
```

### Level 1: Subsystem Data Flow Diagram

```
[Raw Social Post / CSV]
         │
         ▼
 ┌───────────────────────┐
 │ 1.0 Preprocessing     │  --> Strips URLs, translates emojis, retains negations
 └───────────┬───────────┘
             │ Cleaned Text
             ▼
 ┌───────────────────────┐
 │ 2.0 TF-IDF Vectorizer │  --> Sublinear TF, 1-2 N-Grams, Max 6,000 features
 └───────────┬───────────┘
             │ Sparse Feature Matrix
             ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 3.0 Model Tournament (ML & PyTorch Neural Networks)         │
 │   - Linear SVM (Calibrated)   - Multi-Layer Perceptron (NN) │
 │   - Logistic Regression       - PyTorch Deep Neural Network │
 │   - Multinomial Naive Bayes                                 │
 └───────────┬─────────────────────────────────────────────────┘
             │ Posterior Probabilities & Softmax Output
             ▼
 ┌───────────────────────┐
 │ 4.0 XAI Explainability│  --> Extracts top positive/negative/neutral token weights
 └───────────┬───────────┘
             │
     ┌───────┴────────────────────────┐
     ▼                                ▼
┌───────────────────────┐ ┌───────────────────────┐
│ 5.0 SQLite Store      │ │ 6.0 Web Dashboard &   │
│     (sentiment.db)    │ │     REST API JSON     │
└───────────────────────┘ └───────────────────────┘
```

### Level 2: Sentiment Inference & Explainability Deep-Dive

```
Clean Tokens ──► [N-Gram Matrix: (1, 2)] ──► [TF-IDF Feature Alignment]
                                                    │
                                                    ▼
                             [Classifier / Neural Network Decision Function]
                                                    │
                   ┌────────────────────────────────┴──────────────────────────────┐
                   ▼                                                               ▼
        [Sigmoid / Softmax Probabilities]                            [Feature Weight Multiplication]
                   │                                                               │
                   ▼                                                               ▼
       [P(Neg), P(Neu), P(Pos)]                                    [Top Ranked Explanatory Tokens]
                   │                                                               │
                   └────────────────────────┬──────────────────────────────────────┘
                                            ▼
                           [Final Prediction Response Payload]
```

---

## 4. Project Directory Structure

```
c:/NNDL project/
│
├── app.py                      # Flask orchestrator, auto-initialization & API routes
├── config.py                   # Central hyperparameters, file paths, and environment settings
├── requirements.txt            # Pinned project dependencies
├── README.md                   # Comprehensive project documentation
├── .gitignore                  # Git ignore rules for virtual environments, caches, and DBs
├── .env.example                # Template for optional external API credentials
├── .env                        # Local environment configuration
│
├── data/
│   ├── raw/
│   │   └── raw_social_sentiment.csv  # Raw benchmark social corpus (CardiffNLP TweetEval)
│   ├── processed/
│   │   └── sentiment_dataset.csv     # Cleaned, standardized 3-class training dataset
│   ├── uploads/                      # Temporary storage for user-uploaded CSV batches
│   │   └── .gitkeep
│   ├── sample_social_posts.csv       # Sample demo CSV for bulk analysis testing
│   └── sample_live_posts.json        # Curated mock live stream for offline demonstrations
│
├── models/
│   ├── sentiment_model.pkl     # Best trained classifier (Linear SVM Calibrated)
│   ├── sentiment_nn_pytorch.pt # PyTorch Deep Neural Network serialized checkpoint
│   ├── vectorizer.pkl          # Trained TF-IDF feature extractor
│   ├── metadata.json           # Model configuration, vocabulary size, and metrics
│   ├── model_metrics.json      # Comparative benchmark results across all 5 models
│   └── .gitkeep
│
├── database/
│   ├── sentiment.db            # Persistent SQLite database storing prediction logs
│   └── .gitkeep
│
├── src/
│   ├── __init__.py             # Module initializer
│   ├── data_loader.py          # Automatic dataset fetcher, fallback builder, and validator
│   ├── preprocessing.py        # Social NLP cleaner (emojis, mentions, URLs, negations)
│   ├── neural_network.py       # PyTorch SentimentNeuralNetwork, dataset, and estimator wrapper
│   ├── train.py                # Multi-model benchmarking tournament & model selection
│   ├── evaluate.py             # Accuracy, precision, recall, F1, and confusion matrix engine
│   ├── predict.py              # Single and batch inference with token explainability
│   ├── social_media_fetcher.py # Real-time API connectors (Reddit, Mastodon, YouTube)
│   ├── database.py             # SQLite schema manager and transactional queries
│   ├── analytics.py            # Corpus EDA statistics, n-gram tables, and word clouds
│   └── utils.py                # Logging, file sanitization, and JSON helpers
│
├── templates/
│   ├── base.html               # Master layout, navigation, dark/light theme, and footer
│   ├── index.html              # Main dashboard with KPI cards, charts, and live stream
│   ├── analyze.html            # Real-time text analyzer with confidence meter and pills
│   ├── bulk.html               # Drag-and-drop CSV batch processor and exporter
│   ├── analytics.html          # Dataset analytics, distributions, and word clouds
│   ├── performance.html        # Model benchmark table, metrics, and confusion matrix
│   ├── live_feed.html          # Live multi-platform social media stream view
│   └── about.html              # Academic presentation, DFDs, formulas, and viva notes
│
├── static/
│   ├── css/
│   │   └── style.css           # Modern dark-mode responsive stylesheet
│   ├── js/
│   │   └── main.js             # AJAX requests, Chart.js visualizations, and batch UI
│   └── images/
│       ├── wordcloud_positive.png  # High-frequency positive n-gram word cloud
│       ├── wordcloud_negative.png  # High-frequency negative n-gram word cloud
│       └── wordcloud_neutral.png   # High-frequency neutral n-gram word cloud
│
├── notebooks/
│   └── eda.ipynb               # Interactive Jupyter Notebook for exploratory data analysis
│
└── tests/
    ├── __init__.py
    ├── test_neural_network.py  # Unit tests for PyTorch SentimentNN & Scikit-Learn MLP
    ├── test_preprocessing.py   # Unit tests for text cleaning and emoji handling
    ├── test_prediction.py      # Unit tests for model inference and probabilities
    └── test_routes.py          # Integration tests for Flask web routes and REST APIs
```

---

## 5. Mathematical & Theoretical Formulations

### 1. Term Frequency - Inverse Document Frequency (TF-IDF)
TF-IDF reflects how important a token $t$ is to a document $d$ within a corpus $D$:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Using sublinear term frequency scaling and smooth inverse document frequency:

$$\text{TF}(t, d) = 1 + \ln(\text{count}(t, d)) \quad \text{for } \text{count}(t, d) > 0$$

$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + \text{DF}(t, D)}\right) + 1$$

### 2. Multinomial Naive Bayes with Laplace Smoothing
Under the conditional independence assumption:

$$P(c \mid d) \propto P(c) \prod_{i=1}^{n} P(t_i \mid c)$$

Laplace additive smoothing ($\alpha = 0.5$) prevents zero-probability penalties for unseen test vocabulary:

$$P(t_i \mid c) = \frac{\text{count}(t_i, c) + \alpha}{\sum_{t \in V} \text{count}(t, c) + \alpha |V|}$$

### 3. Linear Support Vector Machine with Platt Calibration
Linear SVM seeks the optimal maximum-margin hyperplane:

$$\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{N} \max\left(0, 1 - y_i(w^T x_i + b)\right)$$

Because raw SVM margin distances $f(x) = w^T x + b$ do not represent probabilities, **Platt Scaling** fits a logistic sigmoid over cross-validated decision function scores:

$$P(y = 1 \mid f(x)) = \frac{1}{1 + \exp(A \cdot f(x) + B)}$$

### 4. Deep Feedforward Neural Network (PyTorch / Multi-Layer Perceptron)
The PyTorch `SentimentNeuralNetwork` and `MLPClassifier` process the high-dimensional sparse TF-IDF vectors through successive non-linear hidden representations:

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

$$\hat{\mathbf{z}}^{[l]} = \frac{\mathbf{z}^{[l]} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \cdot \gamma + \beta \quad (\text{Batch Normalization})$$

$$\mathbf{a}^{[l]} = \max\left(0, \hat{\mathbf{z}}^{[l]}\right) \quad (\text{Rectified Linear Unit Activation})$$

$$\mathbf{r} \sim \text{Bernoulli}(1 - p), \quad \tilde{\mathbf{a}}^{[l]} = \mathbf{r} \odot \mathbf{a}^{[l]} \quad (\text{Dropout Regularization})$$

Output Softmax posterior probabilities over classes $c \in \{0, 1, 2\}$:

$$P(y = c \mid \mathbf{x}) = \frac{\exp\left(z_c^{[\text{out}]}\right)}{\sum_{k=1}^{3} \exp\left(z_k^{[\text{out}]}\right)}$$

Trained using **Categorical Cross-Entropy Loss** with class weighting and AdamW weight decay:

$$\mathcal{L} = -\sum_{c=1}^{3} w_c \, y_c \log\left(\hat{y}_c\right) + \frac{\lambda}{2} \|\mathbf{W}\|_2^2$$

### 5. Evaluation Metrics
For multi-class sentiment classification, metrics are weighted by class support:

$$\text{Accuracy} = \frac{\sum_{c} \text{TP}_c}{N}$$

$$\text{Precision}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}, \quad \text{Recall}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$

$$\text{F1-Score}_c = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

$$\text{Weighted F1} = \sum_{c} \left( \frac{N_c}{N} \times \text{F1-Score}_c \right)$$

---

## 6. Model Benchmark & Performance Comparison

The benchmark tournament was evaluated using an 80/20 train-test split on the standardized social dataset with 6,000 maximum n-gram features across all 5 candidate architectures:

| Model Architecture | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM (Calibrated)** | **61.15%** | **60.87%** | **61.15%** | **60.96%** | **Selected Production Model** |
| Logistic Regression | 60.75% | 60.70% | 60.75% | 60.72% | Evaluated |
| Multinomial Naive Bayes | 60.60% | 60.25% | 60.60% | 60.25% | Evaluated |
| Multi-Layer Perceptron (NN) | 59.35% | 59.24% | 59.35% | 59.27% | Evaluated |
| PyTorch Deep Neural Network | 57.90% | 58.33% | 57.90% | 58.06% | Evaluated & Saved (`.pt`) |

> **Academic Insight**: On a noisy, three-class social media corpus (Twitter / X data with balanced neutral classes), ~61% balanced accuracy without multi-gigabyte deep neural networks represents strong statistical performance. Linear SVM with Platt calibration minimizes structural risk while producing calibrated posterior probabilities. The neural network models capture non-linear feature interactions, providing an ideal comparative study for Neural Networks & Deep Learning courses.

---

## 7. Installation & Quickstart Guide

### Prerequisites
- **Python**: Version 3.10, 3.12, or 3.14
- **Operating System**: Windows, macOS, or Linux
- **Git**: Installed and configured on your system

### Step 1: Clone the Repository
```bash
git clone https://github.com/BALA-SAI-CHARAN-MAROJU/Sentiment_analysis.git
cd Sentiment_analysis
```

### Step 2: Create and Activate a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
python app.py
```

### Automated Initialization Sequence
Upon launch, the application automatically handles all setup steps without manual intervention:
```
============================================================
[INFO] Initializing Sentiment Analysis System...
============================================================
[INFO] Checking dataset...
[INFO] Processed dataset found at data/processed/sentiment_dataset.csv.
[INFO] Checking sentiment models...
[INFO] Model artifacts found.
[INFO] Loading sentiment model into memory...
[INFO] Model loaded successfully.
[INFO] Initializing SQLite database...
[INFO] SQLite database initialized at database/sentiment.db
[INFO] Generating initial word clouds for analytics dashboard...
[INFO] System auto-initialization sequence complete.
[INFO] Web server ready to accept requests.
============================================================
 * Running on http://127.0.0.1:5000 (Press CTRL+C to quit)
```

### Step 5: Open the Web Dashboard
Navigate to your local browser at:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 8. Interactive Web Dashboard Walkthrough

The web application is structured into seven purpose-built views:

1. **Executive Dashboard (`/`)**:
   - High-level metric counters showing total predictions, sentiment breakdown, and model confidence.
   - Interactive Chart.js donut chart reflecting real-time sentiment distribution.
   - Live stream ticker previewing incoming social media posts.
2. **Single Post Analyzer (`/analyze`)**:
   - Interactive text input with one-click test sample presets.
   - Real-time classification badge (`Positive`, `Negative`, `Neutral`).
   - Confidence probability progress bars for each class.
   - **XAI Token Pills**: Highlights the exact words and n-grams that influenced the prediction.
3. **Bulk CSV Batch Processor (`/bulk`)**:
   - Drag-and-drop CSV file upload.
   - Auto-detects text columns (e.g., `text`, `tweet`, `post`, `content`).
   - Real-time batch progress bar and interactive paginated results table.
   - Downloadable enriched CSV containing original content plus predicted sentiment and confidence.
4. **Dataset Analytics (`/analytics`)**:
   - Class balance breakdown across the training corpus.
   - Character length and token count distributions.
   - High-frequency n-gram tables.
   - High-resolution word clouds for Positive, Negative, and Neutral classes.
5. **Model Performance (`/performance`)**:
   - Multi-model comparison matrix across all 5 architectures (Accuracy, Precision, Recall, F1).
   - Interactive Confusion Matrix heatmap.
   - Per-class performance metrics.
6. **Live Social Media Stream (`/live-feed`)**:
   - Aggregated social posts from Reddit, Mastodon, YouTube, or curated streams.
   - Filter by platform source and sentiment classification.
7. **Academic Presentation & Viva Notes (`/about`)**:
   - Interactive Level 0-2 Data Flow Diagrams.
   - Mathematical equations for linear, probabilistic, and deep neural network models.
   - Comprehensive viva voce preparation sheet.

---

## 9. RESTful API Documentation

Sentify exposes standardized JSON REST endpoints for external integrations.

### 1. Single Post Prediction
- **Endpoint**: `POST /api/predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "text": "The battery life on this laptop is incredible! Super fast delivery 😍🔋 #tech #happy",
  "source": "API Client"
}
```

- **Example cURL Command**:
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The battery life on this laptop is incredible! Super fast delivery 😍🔋"}'
```

- **Response Payload**:
```json
{
  "text": "The battery life on this laptop is incredible! Super fast delivery 😍🔋",
  "cleaned_text": "the battery life on this laptop is incredible super fast delivery love heart eyes battery",
  "sentiment": "Positive",
  "confidence": 0.8842,
  "probabilities": {
    "negative": 0.0412,
    "neutral": 0.0746,
    "positive": 0.8842
  },
  "explainability": {
    "top_words": [
      { "word": "incredible", "score": 0.4412 },
      { "word": "battery", "score": 0.3821 },
      { "word": "love", "score": 0.3540 },
      { "word": "super", "score": 0.3129 }
    ]
  },
  "model_used": "Linear SVM"
}
```

---

### 2. Bulk Batch Prediction
- **Endpoint**: `POST /api/bulk-predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "texts": [
    "I absolutely love this new update! 😍",
    "Terrible customer service and defective screen 😡",
    "The package arrived on Tuesday as scheduled."
  ]
}
```

- **Response Payload**:
```json
{
  "count": 3,
  "summary": {
    "positive": 1,
    "negative": 1,
    "neutral": 1
  },
  "results": [
    {
      "text": "I absolutely love this new update! 😍",
      "sentiment": "Positive",
      "confidence": 0.8912
    },
    {
      "text": "Terrible customer service and defective screen 😡",
      "sentiment": "Negative",
      "confidence": 0.9324
    },
    {
      "text": "The package arrived on Tuesday as scheduled.",
      "sentiment": "Neutral",
      "confidence": 0.8145
    }
  ]
}
```

---

### 3. Live Social Media Stream
- **Endpoint**: `GET /api/fetch-live`
- **Query Parameters**:
  - `source`: `reddit`, `mastodon`, `youtube`, or `all` (default: `all`)
  - `limit`: Number of posts to return (default: `10`, max: `50`)
  - `subreddit`: Target subreddit for Reddit queries (e.g. `technology`)
  - `video_id`: YouTube video ID for comments (e.g. `dQw4w9WgXcQ`)

---

### 4. Database Statistics & Audit Logs
- **Get Global Stats**: `GET /api/stats`
- **Query Prediction History**: `GET /api/history?limit=50&sentiment=Positive&keyword=battery`
- **Clear History**: `DELETE /api/history`

---

## 10. Automated Testing & Quality Assurance

The repository includes a comprehensive automated test suite with **30 unit and integration tests** built using `pytest`:

```bash
python -m pytest tests/ -v
```

### Test Suite Structure:
- **`tests/test_neural_network.py` (5 tests)**:
  - PyTorch `SentimentNeuralNetwork` forward pass, batchnorm, and output shapes.
  - `SparseTextDataset` sparse-to-dense batch tensor streaming.
  - `PyTorchSentimentClassifier` Scikit-Learn wrapper `fit`, `predict`, and `predict_proba`.
  - PyTorch model checkpoint serialization and restoration (`.pt` file).
  - Scikit-Learn `MLPClassifier` multi-class convergence and probability output.
- **`tests/test_preprocessing.py` (5 tests)**:
  - Emoji expansion into sentiment-carrying keywords.
  - User mention (`@username`) and URL stripping.
  - HTML entity decoding (`&amp;` $\rightarrow$ `&`).
  - Negation word retention (`not`, `never`, `cannot`).
  - Normalization of elongated words (`sooo` $\rightarrow$ `soo`).
- **`tests/test_prediction.py` (6 tests)**:
  - Model and vectorizer artifact loading verification.
  - Sentiment classification output validity.
  - Probability bounds verification ($0.0 \le P \le 1.0$ and $\sum P = 1.0$).
  - Explainability token extraction and score sorting.
  - Batch inference output length alignment.
- **`tests/test_routes.py` (14 tests)**:
  - HTTP 200 OK verification on all web views (`/`, `/analyze`, `/bulk`, `/analytics`, `/performance`, `/live-feed`, `/about`).
  - REST endpoint contracts (`/api/predict`, `/api/bulk-predict`, `/api/stats`, `/api/fetch-live`, `/api/history`).
  - CSV file upload, automated column detection, and error handling for empty files.

---

## 11. Configuration & Optional API Integrations

The system is configured via environment variables. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### Configuration Options:
```env
# Flask Application Settings
FLASK_ENV=development
FLASK_DEBUG=1
SECRET_KEY=sentiment_secret_key_btech_project_2026

# Model & Training Settings
MAX_TRAINING_ROWS=10000         # Set to 0 to train on all 45,000+ rows
MODEL_TYPE=linear_svm           # Options: linear_svm, logistic_regression, naive_bayes
USE_TRANSFORMER=false           # Set to true for optional HuggingFace RoBERTa model

# Optional Reddit API (PRAW)
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=SentimentAnalyzer/1.0

# Optional YouTube Data API v3
YOUTUBE_API_KEY=

# Mastodon Instance
MASTODON_INSTANCE=https://mastodon.social
MASTODON_ACCESS_TOKEN=
```

> **Note**: External API credentials are completely optional. If left unconfigured, Sentify automatically uses public endpoints and curated social feeds for seamless demonstrations.

---

## 12. Academic Viva Voce & Defense Q&A

### Q1: Why is sentiment analysis on social media harder than on long-form reviews (e.g., IMDb)?
**Answer**: Social media posts (tweets, Reddit comments) are short, informal, and context-sparse. They rely heavily on abbreviations, emojis, sarcasm, hashtags, and spelling variations. Review datasets typically consist of structured, multi-paragraph text with rich syntactic clues. Social NLP requires specialized preprocessing to preserve emotive tokens and handle negations without discarding context.

### Q2: Why compare Linear SVM with Deep Neural Networks?
**Answer**: In high-dimensional sparse TF-IDF spaces (thousands of features), text vectors are largely linearly separable. Linear SVM finds the maximum-margin hyperplane, minimizing structural risk with minimal computational overhead. Deep Neural Networks (MLP and PyTorch) model non-linear hierarchical representations and cross-feature interactions using ReLU activations, Batch Normalization, and Dropout. Comparing both paradigms illustrates the trade-off between margin-based convex optimization and gradient-based deep representation learning.

### Q3: Why is probability calibration (Platt Scaling) necessary for SVMs?
**Answer**: Standard SVMs output uncalibrated geometric margins $f(x) \in (-\infty, +\infty)$ representing the distance to the separating hyperplane, not posterior probabilities. Using Platt scaling (`CalibratedClassifierCV`), a logistic sigmoid is fitted to decision function values using out-of-fold cross-validation. This converts raw margins into true calibrated probabilities $P(y \mid x) \in [0, 1]$ required for confidence meters and threshold-based decision making.

### Q4: How is data leakage prevented in this pipeline?
**Answer**: Data leakage is prevented by strictly splitting the raw dataset into training and testing partitions **before** fitting the TF-IDF vectorizer. The vectorizer learns vocabulary frequencies solely from the training fold; unseen test tokens are mapped to zero without leaking corpus statistics into the model.

### Q5: Why preserve negation words instead of using a standard stop-word list?
**Answer**: Standard stop-word lists (such as NLTK's English stop words) discard words like `not`, `no`, `never`, `cannot`, and `neither`. In sentiment analysis, discarding negations reverses the polarity: `"The product is not good"` would be cleaned to `"product good"`, causing the model to incorrectly classify a negative review as positive. Our social preprocessing pipeline explicitly retains all negation operators.

---

## 13. Troubleshooting & FAQ

### Issue: `UnicodeEncodeError` when printing emojis on Windows
- **Solution**: The codebase explicitly enforces UTF-8 encoding across entrypoints via `sys.stdout.reconfigure(encoding='utf-8')`. If using custom scripts, set the environment variable in PowerShell:
  ```powershell
  $env:PYTHONIOENCODING = "utf-8"
  ```

### Issue: Re-training models from scratch
- **Solution**: To re-run the multi-model tournament (including neural networks) and regenerate benchmark metrics:
  ```powershell
  python -m src.train
  ```

### Issue: How to test CSV batch upload during a viva demonstration
- **Solution**: Navigate to `http://127.0.0.1:5000/bulk` and upload the prepared demo file: `data/sample_social_posts.csv`. The application will automatically detect the `text` column and display predictions.

---

## 14. Contributing & Authors

### Project Author
- **Bala Sai Charan Maroju**
- **Repository**: [https://github.com/BALA-SAI-CHARAN-MAROJU/Sentiment_analysis.git](https://github.com/BALA-SAI-CHARAN-MAROJU/Sentiment_analysis.git)

### Academic Discipline
- **Department**: Computer Science & Artificial Intelligence / Machine Learning
- **Subject**: Neural Networks & Deep Learning / NLP Capstone Project

### License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.