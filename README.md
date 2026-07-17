# CineAssist

**CineAssist** is an NLP-based movie recommendation chatbot. Users write a free-text request or answer optional starter questions. The system converts the input into a structured preference object, uses TF-IDF + cosine similarity to retrieve ranked recommendations, and explains each result in plain language.

---

## Project Information

| Field              | Details                                                                           |
| ------------------ | --------------------------------------------------------------------------------- |
| Course             | AML-2403 AI and ML Lab                                                            |
| Semester           | Spring 2026                                                                       |
| Section            | OTT01                                                                             |
| Group              | Group 1                                                                           |
| Project Title      | CineAssist: An AI Movie Recommendation Chatbot with Optional Multilingual Support |
| Faculty Supervisor | William Pourmajidi                                                                |

---

## Team Members

| Name                            | Student ID | Main Contribution Area                                                |
| ------------------------------- | ---------: | --------------------------------------------------------------------- |
| Alejandro Parparcen Grillet     |   C0960408 | Architecture, backend integration, GitHub coordination, final project |
| Carlos Antonio Graniel Manrique |   C0966684 | Movie database, recommendation logic, feature schema                  |
| Lili Marcela Perez Clavijo      |   C0964898 | Chatbot conversation flow, NLP preference extraction                  |
| Brayan Yesid Roncancio Suarez   |   C0966032 | Multilingual normalization, language detection                        |
| David Aponte Monroy             |   C0967956 | Testing, evaluation, metrics, model comparison                        |
| Motunrayo Aduloju               |   C0968107 | Dataset preparation, feature engineering, prototype support           |

---

## Architecture Overview

The system is divided into four blocks:

```
User Input (any language)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Frontend  (app/streamlit_app.py)                               │
│  • Chatbox / textarea                                           │
│  • Optional starter questions (sidebar)                         │
│  • Movie recommendation cards                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │ raw_text + form_data
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Translation Layer  (src/translation/ — Brayan)                 │
│                                                                 │
│  1. detect_language()      → identify user's language           │
│  2. translate_to_english() → convert input to English           │
│     (skipped if user is already writing in English)             │
└────────────────────────┬────────────────────────────────────────┘
                         │ English text
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Pipeline  (src/chatbot/chatbot_flow.py → run_pipeline)         │
│  The single pipeline shared by the chat app, NLP Inspector,     │
│  and the /recommend API.                                        │
│                                                                 │
│  1. language_service   → fast domain-term normalization         │
│  2. nlp_preferences    → extract structured preferences         │
│  3. keyword_extractor  → build focused TF-IDF query             │
│  4. recommender_engine → TF-IDF + cosine similarity             │
│  5. explanation_generator → per-movie explanation               │
└────────────────────────┬────────────────────────────────────────┘
                         │ English response
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Translation Layer  (src/translation/ — Brayan)                 │
│                                                                 │
│  5. translate_from_english() → convert response back to user's  │
│     language (skipped if user wrote in English)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │ Response in user's original language
                         ▼
                      Frontend
```

---

## Multilingual Support — Two-Layer Architecture

> **Author:** Brayan Yesid Roncancio Suarez
> **Files:** `src/translation/`, `backend/services/translation_service.py`

CineAssist handles non-English input through **two complementary layers** that serve different purposes.

---

### Layer 1 — Domain Normalization (Fast, No ML)

**File:** `backend/services/language_service.py`

This layer runs on **every single request** in the API pipeline. It is designed to be instant — no model loading, no network calls.

**What it does:**
1. Detects language via a stopword heuristic (counts known Spanish/French/Portuguese words in the text).
2. Maps movie-domain vocabulary in any supported language to canonical English labels:
   - `chistosa`, `graciosa`, `comédie` → `comedy`
   - `de terror`, `aterradora`, `horreur` → `horror`
   - `años 90`, `los 90` → year range `[1990, 1999]`
3. Returns a normalized English string that the NLP extractor can process.

**Why this exists separately:**
Most users writing in Spanish or French to ask about movies use a small, predictable vocabulary (genre names, moods, decades). For this majority case, full neural translation is unnecessary overhead. The domain mapping handles it in microseconds with zero dependencies.

---

### Layer 2 — Neural Machine Translation (Accurate, ML-Based)

**Files:** `src/translation/lang_detector.py`, `src/translation/translator.py`, `src/translation/fine_tune.py`

This layer handles **full sentence translation** for complex free-text queries where domain mapping is not enough. It is used in:
- `src/chatbot/chatbot_flow.py` — the Streamlit chat path, one translation per user turn
- `GET /translate` API endpoint — exposed as a standalone service

#### Technology: Helsinki-NLP MarianMT via HuggingFace Transformers

MarianMT is a family of neural sequence-to-sequence translation models originally developed at the University of Helsinki and later published on HuggingFace. They are trained on the **OPUS corpus**, a collection of 100+ million parallel sentence pairs extracted from sources like OpenSubtitles, Wikipedia, EU parliament proceedings, and news corpora.

Each language pair uses a separate model (~300 MB each):

| Language pair | HuggingFace model ID |
|---|---|
| Spanish ↔ English | `Helsinki-NLP/opus-mt-es-en` / `opus-mt-en-es` |
| French ↔ English | `Helsinki-NLP/opus-mt-fr-en` / `opus-mt-en-fr` |
| Portuguese ↔ English | `Helsinki-NLP/opus-mt-pt-en` / `opus-mt-en-ROMANCE` |
| German ↔ English | `Helsinki-NLP/opus-mt-de-en` / `opus-mt-en-de` |
| Italian ↔ English | `Helsinki-NLP/opus-mt-it-en` / `opus-mt-en-it` |

Models are downloaded **once** and cached in `~/.cache/huggingface/`. Subsequent calls use an in-memory model cache (`_model_cache` dict in `translator.py`) and run in milliseconds.

#### Translation Pipeline (one user turn in the Streamlit chatbot)

```
User types: "Quiero ver una película de terror de los 90"
                │
                ▼
        detect_language()           ← langdetect (Google statistical library)
        → detected: 'es'
        stored in state_dict['ui_language'] for this conversation
                │
                ▼
        translate_to_english("Quiero ver...", 'es')
        → "I want to watch a horror movie from the 90s"
                │
                ▼
        extract_preferences("I want to watch a horror movie from the 90s")
        → { genres: ['horror'], year_range: [1990, 1999] }
                │
                ▼
        recommend_on_the_fly(...)   ← TF-IDF cosine similarity
        → top-5 movies
                │
                ▼
        generate_explanation(movie, prefs)
        → "Recommended because it matches your interest in horror and fits the 1990s era."
                │
                ▼
        translate_from_english("Recommended because...", 'es')
        → "Te la recomiendo porque coincide con tu interés en terror y es de los años 90."
                │
                ▼
  User reads: "Te la recomiendo porque coincide con tu interés en terror..."
```

#### Language Detection

**File:** `src/translation/lang_detector.py`

Uses the `langdetect` library — a Python port of Google's language identification library. It analyzes **character n-grams** (sequences of 1-4 characters) and compares them against statistical profiles of 55 languages. `DetectorFactory.seed = 42` is set to make detection deterministic (same input always returns same result).

Detection runs **only on the first message** of a conversation. The result is stored in `state_dict['ui_language']` and reused for every subsequent turn so the entire conversation stays in one language.

Fallback to `'en'` occurs when:
- Input is empty or whitespace only
- Detected language is not in our supported set (e.g. Japanese, Arabic)
- Detection raises an exception (e.g. single-character input)

---

### Why Pre-trained MarianMT Instead of Training a Model from Scratch

This is a deliberate design decision based on three engineering principles:

#### 1. Data requirements make scratch training impossible

Training a neural machine translation model from scratch requires **tens of millions of parallel sentence pairs** (source sentence + its human translation). Standard NMT training datasets:

| Dataset | Size |
|---|---|
| WMT 2014 EN-DE | 4.5 million sentence pairs |
| OPUS (used by Helsinki-NLP) | 100+ million pairs |
| Google Translate training | Billions of pairs |

The CineAssist movie corpus has **35 domain-specific sentence pairs** in `fine_tune.py`. Training a model from scratch on 35 examples would produce a model that memorizes those 35 sentences and fails completely on any new input. This is not a limitation of the approach — it is a fundamental statistical reality of sequence-to-sequence learning.

#### 2. Compute requirements are out of scope for an academic project

Training a competitive NMT model from scratch using standard architectures (Transformer, as used in MarianMT) requires:

| Resource | Requirement |
|---|---|
| Hardware | 8+ NVIDIA V100 or A100 GPUs |
| Time | 1–4 weeks of continuous training |
| Energy | Tens of thousands of GPU-hours |
| Cost | $10,000–$100,000 on cloud compute |

A laptop running Python, even with a discrete GPU, cannot produce a useful translation model from scratch in any reasonable timeframe.

#### 3. Transfer learning is the industry standard — even at Google

The correct approach for adapting a general model to a specific domain is **fine-tuning**, not re-training. This is what Google, Microsoft, and DeepL all do:

- Take a model already pre-trained on 100M+ sentence pairs (general linguistic knowledge)
- Fine-tune it on a small domain-specific dataset (movie vocabulary, chatbot phrases)
- The model retains its general translation ability and adapts to the new domain

This is the same principle behind BERT, GPT, and every other production NLP system used today. `fine_tune.py` implements this for CineAssist: 3 epochs of Seq2SeqTrainer on 35 movie sentence pairs, saving the result to `models/translation/en-es/`.

The fine-tuned model is then automatically loaded by `translator.py` instead of the base HuggingFace model, so movie-specific phrases like `"Here are your recommendations"` or `"What genre would you like?"` are translated more naturally.

#### 4. Fully offline — no API keys, no cost, no rate limits

Unlike Google Translate API or DeepL (both require paid accounts and send data to external servers), MarianMT runs **entirely on the local machine**:

- No API key needed
- No internet required after initial model download
- No per-character cost
- No rate limits
- User data stays private

This makes CineAssist deployable in any environment, including offline demos and academic presentations.

---

### Fine-Tuning (Optional — Run Once)

**File:** `src/translation/fine_tune.py`

```bash
# Fine-tune EN↔ES using the built-in 35 movie sentence pairs
python src/translation/fine_tune.py

# Fine-tune with a larger custom CSV (columns: 'en', 'es')
python src/translation/fine_tune.py --csv path/to/data.csv

# Fine-tune a different pair
python src/translation/fine_tune.py --src en --tgt fr
```

Output is saved to `models/translation/en-es/` (and `es-en/` for the reverse direction). The translator loads this automatically on the next run.

---

### Recommendation Pipeline

```
query_text  →  TfidfVectorizer.transform()  →  cosine_similarity()
            →  filter by language / year / min_rating
            →  sort by similarity_score
            →  top-N results + explanations
```

An optional **Random Forest reranker** can be added as a second stage using features: `cosine_score`, `genre_overlap`, `mood_keyword_overlap`, `year_match`, `language_match`, `rating_norm`, `popularity_norm`. The base TF-IDF recommender remains unchanged.

---

## Repository Structure

```
Capstone-Project-AI-ML/
├── app/
│   ├── streamlit_app.py          # Streamlit main page — Chat UI (run this)
│   └── pages/
│       ├── 1_Metrics.py          # Evaluation dashboard (Precision/Recall/F1/MRR)
│       └── 2_NLP_Inspector.py    # NLP pipeline trace for any input text
├── backend/
│   ├── main.py                   # /recommend adapter over run_pipeline
│   ├── api/
│   │   └── routes.py             # FastAPI routes (optional REST API)
│   └── services/
│       ├── language_service.py   # Language detection + domain normalization
│       └── translation_service.py# Translation wrapper (wired to src/translation/)
├── src/
│   ├── nlp/
│   │   └── nlp_preferences.py    # Core preference extraction (English)
│   ├── recommender/
│   │   └── recommender_engine.py # TF-IDF + cosine similarity engine
│   ├── utils/
│   │   └── explanation_generator.py # Natural-language explanation builder
│   ├── chatbot/
│   │   └── chatbot_flow.py       # Conversational wrapper (used by Streamlit)
│   ├── translation/              # Brayan — neural machine translation module
│   │   ├── __init__.py           # Package interface, re-exports all functions
│   │   ├── lang_detector.py      # Language identification (Google langdetect)
│   │   ├── translator.py         # Helsinki-NLP MarianMT EN↔ES/FR/PT/DE/IT
│   │   └── fine_tune.py          # Domain fine-tuning script (run once)
│   ├── metrics/
│   │   ├── evaluator.py          # Evaluator class
│   │   ├── metrics.py            # Precision/Recall/F1/MRR/Accuracy functions
│   │   ├── test_data.py          # 10 predefined test scenarios with real movieIds
│   │   └── setup_and_run.py      # CLI runner for local evaluation
│   ├── data/
│   │   └── preprocess.py         # Downloads raw data + builds processed CSV + models
│   └── evaluation/               # Additional evaluation scripts
├── data/
│   ├── raw/                      # Downloaded source files (TMDB + MovieLens)
│   └── processed/
│       └── movies_final.csv      # Merged, cleaned, ready for the recommender
├── models/
│   ├── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
│   ├── tfidf_matrix.pkl          # Pre-computed movie TF-IDF matrix
│   └── random_forest_reranker.pkl# (optional) RF reranker — not yet trained
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_NLP_cleaning.ipynb
│   ├── 03_Vectorization.ipynb
│   └── 04_Preferences_Extraction.ipynb
├── tests/
│   ├── test_nlp_preferences.py
│   └── test_language_service.py
├── app.py                        # Legacy redirect → app/streamlit_app.py
├── run.sh                        # Linux/macOS runner
├── run.ps1                       # Windows PowerShell runner
├── run.cmd                       # Windows CMD launcher (handles execution policy)
├── requirements.txt
└── README.md
```

---

## Quick Start

### Option A — scripts (recommended)

**Linux / macOS**
```bash
git clone https://github.com/aleistrying/Capstone-Project-AI-ML.git
cd Capstone-Project-AI-ML
chmod +x run.sh
./run.sh            # installs deps, runs tests, launches Streamlit
```

**Windows (CMD — no setup needed)**
```cmd
git clone https://github.com/aleistrying/Capstone-Project-AI-ML.git
cd Capstone-Project-AI-ML
run                 # run.cmd handles execution policy automatically
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/aleistrying/Capstone-Project-AI-ML.git
cd Capstone-Project-AI-ML
.\run.ps1
```

Available commands for all three runners:

| Command | What it does |
|---|---|
| `./run.sh` / `run` / `.\run.ps1` | deps → tests → Streamlit |
| `... setup` | install/update dependencies only |
| `... test` | run test suite |
| `... app` | launch Streamlit at http://localhost:8501 |
| `... api` | launch FastAPI at http://localhost:8000 |
| `... all` | both services together |

---

### Option B — manual setup

```bash
git clone https://github.com/aleistrying/Capstone-Project-AI-ML.git
cd Capstone-Project-AI-ML
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Prepare data and models

Run the preprocessing script — it downloads the datasets automatically and builds the TF-IDF models:

```bash
python src/data/preprocess.py
```

This downloads TMDB 5000 + MovieLens from public sources, merges them, stems the text, fits the TF-IDF vectorizer, and writes:
- `data/processed/movies_final.csv`
- `models/tfidf_vectorizer.pkl`
- `models/tfidf_matrix.pkl`

The notebooks (`01`–`04`) document each step in detail and are useful for exploration, but are not required for setup.

### Run manually

```bash
# Streamlit UI
streamlit run app/streamlit_app.py

# FastAPI (optional)
uvicorn backend.api.routes:app --reload
# → docs at http://localhost:8000/docs

# Tests
python -m pytest tests/ -v
```

---

## API Contract

The FastAPI server exposes three endpoints. Interactive docs are available at `http://localhost:8000/docs` when the API is running.

---

**POST `/recommend`**

```json
{
  "raw_text": "quiero una pelicula chistosa para familia de los 2000",
  "form_data": {
    "genre": null,
    "mood": null,
    "year_range": null,
    "language": null,
    "min_rating": null,
    "similar_to": null
  }
}
```

**Response**

```json
{
  "detected_language": "es",
  "normalized_query": "comedy family funny feel-good from the 2000s quiero una pelicula chistosa para familia de los 2000",
  "preferences": {
    "genres": ["comedy", "family"],
    "mood": ["funny", "feel-good"],
    "year_range": [2000, 2009],
    "language": null,
    "min_rating": null,
    "similar_to": null
  },
  "recommendations": [
    {
      "title": "Movie Title",
      "year": 2004,
      "genres": ["Family", "Comedy"],
      "rating": 7.8,
      "score": 0.86,
      "poster_url": null,
      "explanation": "Recommended because it matches your interest in comedy, family and fits the 2000s era you asked for."
    }
  ],
  "metadata": { "model": "tfidf_cosine", "reranker": "none" }
}
```

---

**POST `/translate`** — *Brayan's endpoint*

Translates any supported language to English using Helsinki-NLP MarianMT. Useful for testing the translation module independently.

```json
{ "text": "Quiero ver una película de terror", "source_language": "es" }
```

Response:
```json
{
  "original_text": "Quiero ver una película de terror",
  "translated_text": "I want to watch a horror movie",
  "detected_language": "es",
  "was_translated": true
}
```

`source_language` is optional — omit it and the language is auto-detected.

---

**GET `/languages`** — *Brayan's endpoint*

Returns all languages the translation module supports.

```json
{
  "en": "English",
  "es": "Spanish",
  "fr": "French",
  "pt": "Portuguese",
  "de": "German",
  "it": "Italian"
}
```

---

## Evaluation

The `src/metrics/` module provides `Evaluator`, `calculate_precision`, `calculate_recall`, `calculate_f1_score`, and `calculate_mean_reciprocal_rank`. Predefined test scenarios live in `src/metrics/test_data.py`.

**CLI runner:**
```bash
python src/metrics/setup_and_run.py           # run all 10 scenarios, print table
python src/metrics/setup_and_run.py --explore  # print dataset summary + top movies by genre
```

**Streamlit Metrics page:** Launch the app and navigate to **📊 Metrics** in the sidebar — runs all scenarios interactively, shows bar charts, per-scenario detail, and a CSV download.

Metrics tracked: **Precision@5**, **Recall@5**, **F1**, **MRR**, **Accuracy**.

---

## MVP vs Extensions

| Feature                                         | Status                         |
| ----------------------------------------------- | ------------------------------ |
| Streamlit UI with chatbox + sidebar             | Done                           |
| Free-text input + optional starter questions    | Done                           |
| Domain-specific multilingual normalization      | Done (language_service.py)     |
| Full neural translation EN↔ES/FR/PT/DE/IT       | Done (src/translation/)        |
| TF-IDF + cosine similarity recommender          | Done                           |
| Feature-based natural-language explanations     | Done                           |
| FastAPI REST backend with /recommend endpoint   | Done                           |
| /translate and /languages API endpoints         | Done (Brayan)                  |
| Precision / Recall / F1 / MRR evaluation        | Done (src/metrics/)            |
| Streamlit Metrics and NLP Inspector pages       | Done                           |
| Domain fine-tuning for translation models       | Done (fine_tune.py, run once)  |
| Random Forest reranker                          | Extension (not yet trained)    |
| Persistent chat history across sessions         | Extension                      |
| NDCG, MAP evaluation metrics                   | Extension                      |

---

## Technology Stack

| Category        | Tools                                                             |
| --------------- | ----------------------------------------------------------------- |
| Language        | Python 3.10+                                                      |
| ML / Data       | pandas, NumPy, scikit-learn, scipy                                |
| NLP             | TF-IDF, cosine similarity, spaCy, NLTK, rapidfuzz                |
| Translation     | HuggingFace Transformers (MarianMT), PyTorch, sentencepiece, langdetect |
| API             | FastAPI, uvicorn, pydantic                                        |
| UI              | Streamlit                                                         |
| Evaluation      | Custom precision/recall/MRR/F1 harness, sacrebleu (BLEU score)   |
| Version Control | GitHub + GitHub Projects                                          |

---

## License

Academic project — AML-2403 AI and ML Lab, Spring 2026, Lambton College.
