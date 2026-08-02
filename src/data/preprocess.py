"""
CineAssist data preprocessing pipeline.

Reads raw TMDB + MovieLens files from data/raw/ and produces:
  data/processed/movies_final.csv   — merged, cleaned, combined_features
  models/tfidf_vectorizer.pkl
  models/tfidf_matrix.pkl

Run:
    python -m src.data.preprocess
"""

import ast
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.text_cleaning import clean_text

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
PROC.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_json_names(raw, key="name", n=None):
    """Parse a JSON string like '[{"id":1,"name":"Action"},...]' → 'Action Comedy'."""
    if not isinstance(raw, str) or not raw.strip().startswith("["):
        return ""
    try:
        items = ast.literal_eval(raw)
        names = [item[key] for item in items if isinstance(item, dict) and key in item]
        if n:
            names = names[:n]
        return " ".join(names)
    except (TypeError, ValueError, SyntaxError, KeyError):
        return ""


def _parse_json_names_list(raw, key="name", n=None):
    """Like _parse_json_names but returns a Python list."""
    if not isinstance(raw, str) or not raw.strip().startswith("["):
        return []
    try:
        items = ast.literal_eval(raw)
        names = [item[key] for item in items if isinstance(item, dict) and key in item]
        if n:
            names = names[:n]
        return names
    except (TypeError, ValueError, SyntaxError, KeyError):
        return []


def _extract_director(crew_raw):
    """Find the first Director in the crew JSON string."""
    if not isinstance(crew_raw, str):
        return ""
    try:
        crew = ast.literal_eval(crew_raw)
        for person in crew:
            if isinstance(person, dict) and person.get("job") == "Director":
                return person.get("name", "")
    except (TypeError, ValueError, SyntaxError):
        return ""
    return ""


# ── Step 1: Load raw files ────────────────────────────────────────────────────
print("Loading raw files...")
movies_raw   = pd.read_csv(RAW / "tmdb_5000_movies.csv")
credits_raw  = pd.read_csv(RAW / "tmdb_5000_credits.csv")
links_raw    = pd.read_csv(RAW / "movielens_links.csv")

print(f"  movies:  {len(movies_raw):,} rows  |  cols: {list(movies_raw.columns)}")
print(f"  credits: {len(credits_raw):,} rows  |  cols: {list(credits_raw.columns)}")
print(f"  links:   {len(links_raw):,} rows")

# ── Step 2: Merge movies + credits ───────────────────────────────────────────
print("\nMerging movies + credits...")

# credits may have 'movie_id' or 'id' as join key
join_col = "movie_id" if "movie_id" in credits_raw.columns else "id"
credits_raw = credits_raw.rename(columns={join_col: "id"})

merged = movies_raw.merge(credits_raw[["id", "cast", "crew"]], on="id", how="left")
print(f"  merged: {len(merged):,} rows")

# ── Step 3: Parse JSON columns ───────────────────────────────────────────────
print("Parsing JSON fields...")
merged["genres_list"]  = merged["genres"].apply(_parse_json_names_list)
merged["Cast_list"]    = merged["cast"].apply(lambda x: _parse_json_names(x, n=5))
merged["Director"]     = merged["crew"].apply(_extract_director)
merged["keywords_str"] = merged["keywords"].apply(_parse_json_names)

# ── Step 4: Join with MovieLens links to get movieId ────────────────────────
print("Joining with MovieLens links for movieId...")

# links.csv tmdbId is a float — coerce
links_raw["tmdbId"] = pd.to_numeric(links_raw["tmdbId"], errors="coerce")
merged["id"] = pd.to_numeric(merged["id"], errors="coerce")

merged = merged.merge(
    links_raw[["movieId", "tmdbId", "imdbId"]],
    left_on="id", right_on="tmdbId", how="left"
)

# Build imdb_id in tt format (used by the existing notebooks)
merged["imdb_id"] = merged["imdbId"].apply(
    lambda x: f"tt{int(x):07d}" if pd.notna(x) else None
)
merged["imdb_id_clean"] = merged["imdbId"].fillna(0).astype(int)

print(f"  movies with movieId: {merged['movieId'].notna().sum():,} / {len(merged):,}")

# ── Step 5: Select and clean final columns ───────────────────────────────────
print("Building final dataset...")

movies_final = merged[[
    "movieId", "imdb_id", "title", "genres_list", "overview",
    "keywords_str", "Cast_list", "Director", "vote_average",
    "release_date", "original_language", "popularity"
]].copy()

movies_final = movies_final.rename(columns={"keywords_str": "keywords"})

# Fill missing text fields
for col in ["overview", "keywords", "Cast_list", "Director"]:
    movies_final[col] = movies_final[col].fillna("")

# Extract release_year for the recommender engine
movies_final["release_year"] = pd.to_datetime(
    movies_final["release_date"], errors="coerce"
).dt.year

# Drop rows without a title or vote_average
movies_final = movies_final.dropna(subset=["title", "vote_average"])
movies_final = movies_final[movies_final["title"].str.strip() != ""]

print(f"  final rows: {len(movies_final):,}")
print(f"  sample genres_list: {movies_final['genres_list'].iloc[0]}")

# ── Step 6: Build combined_features (mirrors notebook 02) ───────────────────
print("Building combined_features...")

movies_final["combined_features"] = (
    movies_final["movieId"].fillna(0).astype(int).astype(str) + " " +
    movies_final["title"].fillna("") + " " +
    movies_final["genres_list"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x)) + " " +
    movies_final["overview"].fillna("") + " " +
    movies_final["keywords"].fillna("") + " " +
    movies_final["Director"].fillna("") + " " +
    movies_final["Cast_list"].fillna("") + " " +
    movies_final["vote_average"].astype(str) + " " +
    movies_final["release_date"].fillna("").astype(str)
)

# ── Step 7: NLP cleaning (shared with the query side — see src/utils/text_cleaning) ─
print("Cleaning combined_features (this takes ~30s)...")

movies_final["combined_features_stemmed"] = (
    movies_final["combined_features"].apply(clean_text)
)

# ── Step 8: TF-IDF vectorization ─────────────────────────────────────────────
print("Fitting TF-IDF vectorizer...")

corpus = movies_final["combined_features_stemmed"].fillna("")

# min_df=2 (absolute count) keeps any term appearing in ≥2 movies, so titles,
# cast names and niche keywords survive into the vocabulary. The old min_df=0.01
# required a term in ≥1% of movies, which pruned the vocab to ~1k common words.
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8)
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

# ── Step 9: Save everything ──────────────────────────────────────────────────
print("Saving outputs...")

out_csv = PROC / "movies_final.csv"
movies_final.to_csv(out_csv, index=False)
print(f"  → {out_csv}  ({out_csv.stat().st_size / 1e6:.1f} MB)")

joblib.dump(tfidf_vectorizer, MODELS / "tfidf_vectorizer.pkl")
print(f"  → {MODELS / 'tfidf_vectorizer.pkl'}")

joblib.dump(tfidf_matrix, MODELS / "tfidf_matrix.pkl")
print(f"  → {MODELS / 'tfidf_matrix.pkl'}")

print("\nDone. Run: streamlit run app/streamlit_app.py")
