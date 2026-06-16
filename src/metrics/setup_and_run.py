"""
Local evaluation runner for CineAssist.

Usage:
    python src/metrics/setup_and_run.py               # run all scenarios
    python src/metrics/setup_and_run.py --explore     # list movies by genre
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.metrics.evaluator import Evaluator
from src.metrics.test_data import get_test_scenarios
from src.recommender.recommender_engine import recommend_on_the_fly


def load_data():
    path = ROOT / "data" / "processed" / "movies_final.csv"
    if not path.exists():
        print(f"[ERROR] {path} not found — run: python src/data/preprocess.py")
        return None
    df = pd.read_csv(path)
    print(f"[OK] Loaded {len(df):,} movies from {path.name}")
    return df


def load_models():
    vec_path = ROOT / "models" / "tfidf_vectorizer.pkl"
    mat_path = ROOT / "models" / "tfidf_matrix.pkl"
    if not vec_path.exists() or not mat_path.exists():
        print("[ERROR] Model files missing — run: python src/data/preprocess.py")
        return None, None
    vectorizer   = joblib.load(vec_path)
    tfidf_matrix = joblib.load(mat_path)
    print(f"[OK] Loaded TF-IDF vectorizer + matrix {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix


def explore_movies(df):
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Total movies : {len(df):,}")
    print(f"Languages    : {df['original_language'].value_counts().head(5).to_dict()}")
    print(f"Avg rating   : {df['vote_average'].mean():.2f}")
    print(f"Year range   : {int(df['release_year'].min())} – {int(df['release_year'].max())}")

    print("\n--- TOP MOVIES BY GENRE (highest rated) ---")
    genres = ["Action", "Comedy", "Drama", "Science Fiction", "Horror", "Romance", "Animation"]
    for g in genres:
        subset = df[df["genres_list"].fillna("").str.contains(g, case=False)]
        top = subset.nlargest(3, "vote_average")[["movieId", "title", "vote_average"]]
        ids = top["movieId"].dropna().astype(int).tolist()
        print(f"  {g:<18} top movieIds: {ids}")


def run_evaluation(df, vectorizer, tfidf_matrix):
    evaluator = Evaluator(
        recommender_func=recommend_on_the_fly,
        movies_df=df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
    )
    results = evaluator.evaluate_all_scenarios(top_n=5)
    evaluator.print_results(results)

    out = ROOT / "evaluation_results.csv"
    evaluator.export_results_csv(str(out), results)
    print(f"\nResults saved to {out}")
    return results


def main():
    parser = argparse.ArgumentParser(description="CineAssist evaluation runner")
    parser.add_argument("--explore", action="store_true", help="Explore dataset instead of evaluating")
    args = parser.parse_args()

    df = load_data()
    if df is None:
        sys.exit(1)

    if args.explore:
        explore_movies(df)
        return

    vectorizer, tfidf_matrix = load_models()
    if vectorizer is None:
        sys.exit(1)

    run_evaluation(df, vectorizer, tfidf_matrix)


if __name__ == "__main__":
    main()
