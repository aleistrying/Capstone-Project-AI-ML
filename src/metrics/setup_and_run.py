"""
Setup and Run Metrics Evaluation

This script helps you:
1. Load movie data from your dataset
2. Find movieIds for different genres
3. Populate test scenarios
4. Run the evaluation

USAGE:
    python setup_and_run.py --action "explore"     # See available movies by genre
    python setup_and_run.py --action "run"          # Run evaluation with updated test data
"""

import pandas as pd
import sys
from pathlib import Path

# Add src to path FIRST (before importing modules)
sys.path.insert(0, str(Path(__file__).parent.parent))

from recommender.recommender_engine import recommend_on_the_fly
from metrics.test_data import get_test_scenarios
from metrics.evaluator import Evaluator


def load_data_local():
    """Try to load data from local CSV files."""
    try:
        movies_df = pd.read_csv("data/movies_final.csv")
        print("[OK] Loaded movies_final.csv")
        return movies_df
    except FileNotFoundError:
        print("[ERROR] Could not find data/movies_final.csv")
        return None


def load_data_databricks():
    """Try to load data from Databricks (requires Spark session)."""
    try:
        # This requires Databricks notebook environment
        movies_df = spark.read.table(
            "workspace.datasets.movies_final").toPandas()
        print("[OK] Loaded from Databricks workspace.datasets.movies_final")
        return movies_df
    except:
        print("[ERROR] Databricks not available (not in notebook environment)")
        return None


def load_movies_data():
    """Load movie data from available sources."""
    print("\n" + "=" * 80)
    print("LOADING MOVIE DATA")
    print("=" * 80)

    movies_df = load_data_local()
    if movies_df is None:
        movies_df = load_data_databricks()

    if movies_df is None:
        print("\n[WARNING] ERROR: Could not load movie data from any source.")
        print("\nYou have two options:")
        print("1. Run this script from a Databricks notebook")
        print("2. Export your movie data to CSV: data/movies_final.csv")
        return None

    print(f"\nDataset loaded: {len(movies_df)} movies")
    print(f"Columns: {list(movies_df.columns)}")
    return movies_df


def explore_movies(movies_df):
    """Explore available movies and help populate test data."""
    print("\n" + "=" * 80)
    print("EXPLORE MOVIES BY GENRE/CRITERIA")
    print("=" * 80)

    # Show sample
    print("\nSample movies:")
    cols_to_show = ['movieId', 'title', 'genres_list', 'vote_average']
    if 'release_year' in movies_df.columns:
        cols_to_show.append('release_year')
    elif 'release_date' in movies_df.columns:
        cols_to_show.append('release_date')
    print(movies_df[cols_to_show].head(10))

    # Analyze genres
    if 'genres_list' in movies_df.columns:
        print("\n--- GENRE DISTRIBUTION ---")
        all_genres = set()
        for genres in movies_df['genres_list']:
            if isinstance(genres, list):
                all_genres.update(genres)
            elif isinstance(genres, str):
                # Handle string representation of list
                try:
                    import ast
                    genre_list = ast.literal_eval(genres)
                    all_genres.update(genre_list)
                except:
                    pass

        for genre in sorted(all_genres):
            count = sum(1 for g in movies_df['genres_list'] if isinstance(
                g, list) and genre in g)
            print(f"  {genre}: {count} movies")

    # Show movies by genre
    print("\n--- FIND MOVIEIDS BY GENRE ---")
    genres_to_find = ['action', 'comedy',
                      'drama', 'horror', 'sci-fi', 'romance']

    for target_genre in genres_to_find:
        matching = []
        for idx, row in movies_df.iterrows():
            genres = row.get('genres_list', [])
            if isinstance(genres, list):
                if any(target_genre.lower() in str(g).lower() for g in genres):
                    matching.append(row['movieId'])
            elif isinstance(genres, str):
                if target_genre.lower() in genres.lower():
                    matching.append(row['movieId'])

        if matching:
            print(f"\n{target_genre.upper()} movies (first 5 movieIds):")
            print(f"  {matching[:5]}")

    # High-rated movies
    print("\n--- HIGH RATED MOVIES (> 8.0) ---")
    high_rated = movies_df[movies_df['vote_average'] > 8.0]
    if len(high_rated) > 0:
        print(f"Found {len(high_rated)} highly-rated movies")
        print(f"Sample movieIds: {high_rated['movieId'].head(5).tolist()}")

    # Recent movies
    if 'release_year' in movies_df.columns:
        print("\n--- MOVIES FROM 2000s (2000-2009) ---")
        year_2000s = movies_df[(movies_df['release_year'] >= 2000) & (
            movies_df['release_year'] <= 2009)]
        if len(year_2000s) > 0:
            print(f"Found {len(year_2000s)} movies from 2000s")
            print(f"Sample movieIds: {year_2000s['movieId'].head(5).tolist()}")
    elif 'release_date' in movies_df.columns:
        print("\n--- MOVIES FROM 2000s (2000-2009) ---")
        movies_df['year'] = pd.to_datetime(movies_df['release_date'], errors='coerce').dt.year
        year_2000s = movies_df[(movies_df['year'] >= 2000) & (movies_df['year'] <= 2009)]
        if len(year_2000s) > 0:
            print(f"Found {len(year_2000s)} movies from 2000s")
            print(f"Sample movieIds: {year_2000s['movieId'].head(5).tolist()}")

    # Language
    if 'original_language' in movies_df.columns:
        print("\n--- MOVIES BY LANGUAGE ---")
        lang_counts = movies_df['original_language'].value_counts()
        for lang, count in lang_counts.head(10).items():
            print(f"  {lang}: {count} movies")

        spanish = movies_df[movies_df['original_language'] == 'es']
        if len(spanish) > 0:
            print(
                f"\nSpanish movies - Sample movieIds: {spanish['movieId'].head(5).tolist()}")


def load_models():
    """Load TF-IDF vectorizer and matrix."""
    import joblib

    print("\n" + "=" * 80)
    print("LOADING ML MODELS")
    print("=" * 80)

    try:
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        print("[OK] Loaded tfidf_vectorizer.pkl")
    except:
        print("[ERROR] Could not load tfidf_vectorizer.pkl")
        return None, None

    try:
        tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")
        print("[OK] Loaded tfidf_matrix.pkl")
    except:
        print("[ERROR] Could not load tfidf_matrix.pkl")
        tfidf_matrix = None

    return vectorizer, tfidf_matrix


def run_evaluation(movies_df, vectorizer, tfidf_matrix):
    """Run the evaluation with current test data."""
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)

    evaluator = Evaluator(
        recommender_func=recommend_on_the_fly,
        movies_df=movies_df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix
    )

    # Evaluate all test scenarios
    results = evaluator.evaluate_all_scenarios(top_n=5)

    # Print results
    evaluator.print_results(results)

    # Export to CSV
    output_file = "evaluation_results.csv"
    evaluator.export_results_csv(output_file, results)

    return results


def main():
    """Main execution flow."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Setup and run metrics evaluation for CineAssist"
    )
    parser.add_argument(
        "--action",
        choices=["explore", "run", "setup"],
        default="setup",
        help="Action to perform"
    )

    args = parser.parse_args()

    # Load movies
    movies_df = load_movies_data()
    if movies_df is None:
        sys.exit(1)

    if args.action == "explore":
        explore_movies(movies_df)

    elif args.action == "run":
        print("\n" + "=" * 80)
        print("CHECKING TEST DATA")
        print("=" * 80)

        scenarios = get_test_scenarios()
        empty_scenarios = [s for s in scenarios if not s['relevant_movie_ids']]

        if empty_scenarios:
            print(
                f"\n[WARNING] {len(empty_scenarios)} test scenarios have empty relevant_movie_ids")
            print(
                "\nRun with --action explore first to find movieIds, then update test_data.py")
            print("\nEmpty scenarios:")
            for s in empty_scenarios:
                print(f"  - {s['id']}: {s['description']}")
            sys.exit(1)

        # Load models
        vectorizer, tfidf_matrix = load_models()
        if vectorizer is None:
            print("\n✗ Cannot run evaluation without models")
            sys.exit(1)

        # Run evaluation
        results = run_evaluation(movies_df, vectorizer, tfidf_matrix)

    elif args.action == "setup":
        print("\n" + "=" * 80)
        print("SETUP GUIDE")
        print("=" * 80)

        print("""
STEP 1: Explore available movies
    python src/metrics/setup_and_run.py --action explore

    This will show you:
    - Sample movies from your dataset
    - Movies grouped by genre
    - High-rated movies (> 8.0)
    - Movies from specific years
    - Movies in different languages

STEP 2: Populate test data
    Edit src/metrics/test_data.py

    Replace the relevant_movie_ids with actual movieIds from your dataset.
    Example:
        {
            "id": "test_01_action_movie",
            "user_input": "I want an action movie with lots of fights and explosions",
            "relevant_movie_ids": [1, 2, 3, 5, 50],  # ← Use movieIds from explore output
            "description": "Action movie with fighting and explosions"
        }

STEP 3: Run evaluation
    python src/metrics/setup_and_run.py --action run

    This will:
    - Load your movie data
    - Load the TF-IDF models
    - Test each scenario
    - Print metrics (Precision, Recall, Accuracy, F1, MRR)
    - Save results to evaluation_results.csv

NEXT STEPS:
    1. Run: python src/metrics/setup_and_run.py --action explore
    2. Edit: src/metrics/test_data.py (copy movieIds from explore output)
    3. Run: python src/metrics/setup_and_run.py --action run
        """)


if __name__ == "__main__":
    main()
