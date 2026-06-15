"""
Setup and Run Metrics Evaluation

This script helps you:
1. Load movie data from your dataset
2. Find movieIds for different genres
3. Populate test scenarios
4. Run the evaluation

USAGE:
    # %run ./setup_and_run --action "explore"     # See available movies by genre
    # %run ./setup_and_run --action "run"         # Run evaluation with updated test data
    # Databricks: Use %run or %python as appropriate, or run cells directly.
"""

# import pandas as pd
# import sys
# from pathlib import Path

# Add src to path FIRST (before importing modules)
# sys.path.insert(0, str(Path(__file__).parent.parent))

# from recommender.recommender_engine import recommend_on_the_fly
# from metrics.test_data import get_test_scenarios
# from metrics.evaluator import Evaluator

def load_data_local():
    """Try to load data from local CSV files."""
    try:
        # movies_df = pd.read_csv("data/movies_final.csv")
        # print("[OK] Loaded movies_final.csv")
        # return movies_df
        pass  # Not used in Databricks
    except FileNotFoundError:
        # print("[ERROR] Could not find data/movies_final.csv")
        return None

def load_data_databricks():
    """Try to load data from Databricks (requires Spark session)."""
    try:
        # movies_df = spark.read.table("workspace.default.movies").toPandas()
        # print("[OK] Loaded from Databricks workspace.default.movies")
        # return movies_df
        movies_df = spark.read.table("workspace.default.movies")
        print("[OK] Loaded from Databricks workspace.default.movies")
        return movies_df
    except Exception as e:
        print(f"[ERROR] Databricks not available (not in notebook environment): {e}")
        return None

def load_movies_data():
    """Load movie data from available sources."""
    print("\n" + "=" * 80)
    print("LOADING MOVIE DATA")
    print("=" * 80)

    # movies_df = load_data_local()
    # if movies_df is None:
    #     movies_df = load_data_databricks()
    movies_df = load_data_databricks()

    if movies_df is None:
        print("\n[WARNING] ERROR: Could not load movie data from any source.")
        print("\nYou have two options:")
        print("1. Run this script from a Databricks notebook")
        print("2. Export your movie data to CSV: data/movies_final.csv")
        return None

    print(f"\nDataset loaded: {movies_df.count()} movies")
    print(f"Columns: {movies_df.columns}")
    return movies_df

def explore_movies(movies_df):
    """Explore available movies and help populate test data."""
    print("\n" + "=" * 80)
    print("EXPLORE MOVIES BY GENRE/CRITERIA")
    print("=" * 80)

    # Show sample
    cols_to_show = ['movieId', 'title', 'genres_list', 'vote_average']
    if 'release_year' in movies_df.columns:
        cols_to_show.append('release_year')
    elif 'release_date' in movies_df.columns:
        cols_to_show.append('release_date')
    # print(movies_df[cols_to_show].head(10))
    display(movies_df.select(cols_to_show).limit(10))

    # Analyze genres
    if 'genres_list' in movies_df.columns:
        print("\n--- GENRE DISTRIBUTION ---")
        # all_genres = set()
        # for genres in movies_df['genres_list']:
        #     if isinstance(genres, list):
        #         all_genres.update(genres)
        #     elif isinstance(genres, str):
        #         try:
        #             import ast
        #             genre_list = ast.literal_eval(genres)
        #             all_genres.update(genre_list)
        #         except:
        #             pass
        from pyspark.sql.functions import explode, from_json, col
        from pyspark.sql.types import ArrayType, StringType
        genres_exploded = movies_df.select(explode(from_json(col("genres_list"), ArrayType(StringType()))).alias("genre"))
        all_genres = [row.genre for row in genres_exploded.distinct().collect() if row.genre]
        for genre in sorted(all_genres):
            count = movies_df.filter(movies_df.genres_list.contains(genre)).count()
            print(f"  {genre}: {count} movies")

    # Show movies by genre
    print("\n--- FIND MOVIEIDS BY GENRE ---")
    genres_to_find = ['action', 'comedy', 'drama', 'horror', 'sci-fi', 'romance']

    for target_genre in genres_to_find:
        # matching = []
        # for idx, row in movies_df.iterrows():
        #     genres = row.get('genres_list', [])
        #     if isinstance(genres, list):
        #         if any(target_genre.lower() in str(g).lower() for g in genres):
        #             matching.append(row['movieId'])
        #     elif isinstance(genres, str):
        #         if target_genre.lower() in genres.lower():
        #             matching.append(row['movieId'])
        matching_df = movies_df.filter(movies_df.genres_list.contains(target_genre))
        matching_ids = [row.movieId for row in matching_df.select("movieId").limit(5).collect()]
        if matching_ids:
            print(f"\n{target_genre.upper()} movies (first 5 movieIds):")
            print(f"  {matching_ids}")

    # High-rated movies
    print("\n--- HIGH RATED MOVIES (> 8.0) ---")
    high_rated = movies_df.filter(movies_df.vote_average > 8.0)
    count_high = high_rated.count()
    if count_high > 0:
        print(f"Found {count_high} highly-rated movies")
        sample_ids = [row.movieId for row in high_rated.select("movieId").limit(5).collect()]
        print(f"Sample movieIds: {sample_ids}")

    # Recent movies
    if 'release_year' in movies_df.columns:
        print("\n--- MOVIES FROM 2000s (2000-2009) ---")
        year_2000s = movies_df.filter((movies_df.release_year >= 2000) & (movies_df.release_year <= 2009))
        count_2000s = year_2000s.count()
        if count_2000s > 0:
            print(f"Found {count_2000s} movies from 2000s")
            sample_ids = [row.movieId for row in year_2000s.select("movieId").limit(5).collect()]
            print(f"Sample movieIds: {sample_ids}")
    elif 'release_date' in movies_df.columns:
        print("\n--- MOVIES FROM 2000s (2000-2009) ---")
        from pyspark.sql.functions import year
        movies_df = movies_df.withColumn("year", year("release_date"))
        year_2000s = movies_df.filter((movies_df.year >= 2000) & (movies_df.year <= 2009))
        count_2000s = year_2000s.count()
        if count_2000s > 0:
            print(f"Found {count_2000s} movies from 2000s")
            sample_ids = [row.movieId for row in year_2000s.select("movieId").limit(5).collect()]
            print(f"Sample movieIds: {sample_ids}")

    # Language
    if 'original_language' in movies_df.columns:
        print("\n--- MOVIES BY LANGUAGE ---")
        lang_counts = movies_df.groupBy("original_language").count().orderBy("count", ascending=False)
        for row in lang_counts.limit(10).collect():
            print(f"  {row.original_language}: {row['count']} movies")
        spanish = movies_df.filter(movies_df.original_language == 'es')
        count_spanish = spanish.count()
        if count_spanish > 0:
            sample_ids = [row.movieId for row in spanish.select("movieId").limit(5).collect()]
            print(f"\nSpanish movies - Sample movieIds: {sample_ids}")

def load_models():
    """Load TF-IDF vectorizer and matrix."""
    # import joblib
    print("\n" + "=" * 80)
    print("LOADING ML MODELS")
    print("=" * 80)
    # Databricks: Model loading should use MLflow or DBFS, not local files
    try:
        # vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        # print("[OK] Loaded tfidf_vectorizer.pkl")
        # Example: Load from DBFS or MLflow if available
        vectorizer = None
        print("[ERROR] Could not load tfidf_vectorizer.pkl in Databricks")
    except:
        print("[ERROR] Could not load tfidf_vectorizer.pkl")
        return None, None

    try:
        # tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")
        # print("[OK] Loaded tfidf_matrix.pkl")
        tfidf_matrix = None
        print("[ERROR] Could not load tfidf_matrix.pkl in Databricks")
    except:
        print("[ERROR] Could not load tfidf_matrix.pkl")
        tfidf_matrix = None

    return vectorizer, tfidf_matrix

def run_evaluation(movies_df, vectorizer, tfidf_matrix):
    """Run the evaluation with current test data."""
    print("\n" + "=" * 80)
    print("RUNNING EVALUATION")
    print("=" * 80)

    # evaluator = Evaluator(
    #     recommender_func=recommend_on_the_fly,
    #     movies_df=movies_df,
    #     vectorizer=vectorizer,
    #     tfidf_matrix=tfidf_matrix
    # )
    # results = evaluator.evaluate_all_scenarios(top_n=5)
    # evaluator.print_results(results)
    # output_file = "evaluation_results.csv"
    # evaluator.export_results_csv(output_file, results)
    # return results
    print("[ERROR] Evaluation not supported in Databricks without model loading adjustment.")
    return None

def main():
    """Main execution flow."""
    # import argparse
    # parser = argparse.ArgumentParser(
    #     description="Setup and run metrics evaluation for CineAssist"
    # )
    # parser.add_argument(
    #     "--action",
    #     choices=["explore", "run", "setup"],
    #     default="setup",
    #     help="Action to perform"
    # )
    # args, unknown = parser.parse_known_args()
    # Databricks: Use notebook parameters or cell execution, not argparse
    action = "run"  # Default for Databricks, override as needed

    # Load movies
    movies_df = load_movies_data()
    if movies_df is None:
        return

    if action == "explore":
        explore_movies(movies_df)

    elif action == "run":
        print("\n" + "=" * 80)
        print("CHECKING TEST DATA")
        print("=" * 80)
        # scenarios = get_test_scenarios()
        # empty_scenarios = [s for s in scenarios if not s['relevant_movie_ids']]
        # if empty_scenarios:
        #     print(
        #         f"\n[WARNING] {len(empty_scenarios)} test scenarios have empty relevant_movie_ids")
        #     print(
        #         "\nRun with --action explore first to find movieIds, then update test_data.py")
        #     print("\nEmpty scenarios:")
        #     for s in empty_scenarios:
        #         print(f"  - {s['id']}: {s['description']}")
        #     sys.exit(1)
        # vectorizer, tfidf_matrix = load_models()
        # if vectorizer is None:
        #     print("\n✗ Cannot run evaluation without models")
        #     sys.exit(1)
        # results = run_evaluation(movies_df, vectorizer, tfidf_matrix)
        print("[ERROR] Run evaluation not supported in Databricks without model loading adjustment.")

    elif action == "setup":
        print("\n" + "=" * 80)
        print("SETUP GUIDE")
        print("=" * 80)
        print("""
STEP 1: Explore available movies
    # %run ./setup_and_run --action explore

    This will show you:
    - Sample movies from your dataset
    - Movies grouped by genre
    - High-rated movies (> 8.0)
    - Movies from specific years
    - Movies in different languages

STEP 2: Populate test data
    # Edit src/metrics/test_data.py

    Replace the relevant_movie_ids with actual movieIds from your dataset.
    Example:
        {
            "id": "test_01_action_movie",
            "user_input": "I want an action movie with lots of fights and explosions",
            "relevant_movie_ids": [1, 2, 3, 5, 50],  # ← Use movieIds from explore output
            "description": "Action movie with fighting and explosions"
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
