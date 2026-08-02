"""
Retrain the TF-IDF vectorizer + matrix from an existing processed
data/processed/movies_final.csv (e.g. the consolidated ~89K dataset).

Unlike preprocess.py — which rebuilds movies_final.csv from the raw TMDB-5000
files — this script assumes movies_final.csv already exists (with a
`combined_features` column) and only:
  1. ensures a `release_year` column (derived from release_date),
  2. fits the SAME TF-IDF config used in preprocess.py on clean_text(combined_features),
  3. writes models/tfidf_vectorizer.pkl + models/tfidf_matrix.pkl.

Run:
    python -m src.data.retrain_from_final
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.text_cleaning import clean_text

ROOT = Path(__file__).resolve().parent.parent.parent
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

csv_path = PROC / "movies_final.csv"
print(f"Loading {csv_path} ...")
df = pd.read_csv(csv_path)
print(f"  rows: {len(df):,}  |  cols: {list(df.columns)}")

if "combined_features" not in df.columns:
    raise SystemExit("ERROR: movies_final.csv has no 'combined_features' column.")

# Ensure release_year exists (the recommender/UI use it for the soft-decade ranking)
if "release_year" not in df.columns and "release_date" in df.columns:
    print("Deriving release_year from release_date ...")
    df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

# Clean the corpus exactly like queries are cleaned (shared clean_text)
print("Cleaning combined_features (this can take a minute on ~89K rows) ...")
corpus = df["combined_features"].fillna("").apply(clean_text)

# Same config as preprocess.py so behaviour is identical, just more data.
print("Fitting TF-IDF (ngram 1-2, min_df=2, max_df=0.8) ...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.8)
matrix = vectorizer.fit_transform(corpus)
print(f"  TF-IDF matrix: {matrix.shape}  |  vocab: {len(vectorizer.vocabulary_):,}")

# Persist the (possibly release_year-augmented) dataframe so rows align with the matrix
df.to_csv(csv_path, index=False)
print(f"  re-saved {csv_path}")

joblib.dump(vectorizer, MODELS / "tfidf_vectorizer.pkl")
joblib.dump(matrix, MODELS / "tfidf_matrix.pkl")
print(f"  → {MODELS/'tfidf_vectorizer.pkl'}")
print(f"  → {MODELS/'tfidf_matrix.pkl'}")
print("Done.")
