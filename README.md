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
User Input
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Frontend  (app/streamlit_app.py or frontend/)                  │
│  • Chatbox / textarea                                           │
│  • Optional starter questions (sidebar)                        │
│  • Movie recommendation cards                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │ raw_text + form_data
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Backend Controller  (backend/main.py)                          │
│                                                                 │
│  1. language_service  → detect language, normalize domain terms │
│  2. nlp_service       → extract structured preferences          │
│  3. recommender_service → TF-IDF + cosine similarity           │
│  4. explanation_service → generate per-movie explanation        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  Source Modules  (src/)                                         │
│  • nlp/nlp_preferences.py    — preference extraction           │
│  • recommender/recommender_engine.py — cosine similarity       │
│  • utils/explanation_generator.py   — explanation text         │
│  • metrics/                         — evaluation harness       │
└─────────────────────────────────────────────────────────────────┘
```

### Multilingual Normalization Scope

The in-house translation layer targets the **movie domain only** — not general-purpose translation. It:

1. Detects language via stopword heuristics (English, Spanish, French, Portuguese).
2. Maps multilingual synonyms to canonical English genre/mood labels (`chistosa` → `comedy`, `de terror` → `horror`).
3. Parses decade expressions in multiple languages (`años 90` → `[1990, 1999]`).
4. Returns a normalized English string and a structured preference object for the recommender.

Full sentence translation is a planned extension. The integration point is `src/translation/translator.py` — the functions are stubbed with clear `TODO` comments and wiring instructions for Google Translate, DeepL, or HuggingFace offline models. The `backend/services/translation_service.py` wrapper calls these stubs and is ready to activate once a backend is chosen.

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
│   ├── main.py                   # Central controller
│   ├── api/
│   │   └── routes.py             # FastAPI routes (optional REST API)
│   └── services/
│       ├── language_service.py   # Language detection + domain normalization
│       ├── nlp_service.py        # Preference extraction wrapper
│       ├── recommender_service.py# TF-IDF recommendation wrapper
│       ├── explanation_service.py# Explanation wrapper
│       └── translation_service.py# Translation wrapper (stub — see src/translation/)
├── src/
│   ├── nlp/
│   │   └── nlp_preferences.py    # Core preference extraction (English)
│   ├── recommender/
│   │   └── recommender_engine.py # TF-IDF + cosine similarity engine
│   ├── utils/
│   │   └── explanation_generator.py # Natural-language explanation builder
│   ├── chatbot/
│   │   └── chatbot_flow.py       # Conversational wrapper (used by Streamlit)
│   ├── translation/
│   │   ├── __init__.py
│   │   └── translator.py         # Translation stubs (TODO: wire real backend)
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

| MVP                                        | Extension if time allows  |
| ------------------------------------------ | ------------------------- |
| Streamlit UI with chatbox + sidebar        | React/Next.js + FastAPI   |
| Free-text input + starter questions        | Persistent chat history   |
| Domain-specific multilingual normalization | Broader language coverage |
| TF-IDF + cosine similarity                 | Random Forest reranker    |
| Feature-based explanations                 | User feedback loop        |
| Precision/Recall/MRR evaluation            | NDCG, MAP                 |

---

## Technology Stack

| Category        | Tools                                   |
| --------------- | --------------------------------------- |
| Language        | Python 3.10+                            |
| ML / Data       | pandas, NumPy, scikit-learn, scipy      |
| NLP             | TF-IDF, cosine similarity, spaCy, NLTK  |
| API             | FastAPI, uvicorn, pydantic              |
| UI              | Streamlit (MVP); React/Next.js optional |
| Evaluation      | Custom precision/recall/MRR harness     |
| Version Control | GitHub + GitHub Projects                |

---

## License

Academic project — AML-2403 AI and ML Lab, Spring 2026, Lambton College.
