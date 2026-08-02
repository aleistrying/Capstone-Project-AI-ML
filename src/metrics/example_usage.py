"""
Example usage of the metrics evaluation module.

This script demonstrates how to evaluate the CineAssist recommender
using the Evaluator class and predefined test scenarios.

NOTE: This is a template. Adapt paths and ground-truth movie IDs based on your dataset.
"""

import pandas as pd
import joblib

from src.metrics.evaluator import Evaluator
from src.metrics.test_data import print_test_scenarios
from src.recommender.recommender_engine import recommend_on_the_fly


def example_1_single_query_evaluation():
    """Evaluate a single user query."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Single Query Evaluation")
    print("=" * 80)

    # Load data and models
    try:
        movies_df = pd.read_csv("../../data/movies_final.csv")
        vectorizer = joblib.load("../../models/tfidf_vectorizer.pkl")
        tfidf_matrix = joblib.load("../../models/tfidf_matrix.pkl")
    except (OSError, ValueError, EOFError) as exc:
        print(f"Error loading files: {exc}")
        print("Make sure data files are in the correct location.")
        return

    # Create evaluator
    evaluator = Evaluator(
        recommender_func=recommend_on_the_fly,
        movies_df=movies_df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
    )

    # Evaluate a single query
    user_input = "I want an action movie with lots of fights and explosions"
    relevant_ids = [1, 2, 3, 5, 50]  # These should be actual movieIds from your dataset

    result = evaluator.evaluate_single_query(user_input, relevant_ids, top_n=5)

    print(f"\nUser Input: {user_input}")
    print(f"Recommended: {result['predicted_ids']}")
    print(f"Relevant:    {result['relevant_ids']}")
    print("\nMetrics:")
    print(f"  Precision: {result['precision']:.3f}")
    print(f"  Recall:    {result['recall']:.3f}")
    print(f"  Accuracy:  {result['accuracy']:.3f}")
    print(f"  F1 Score:  {result['f1_score']:.3f}")
    print(f"  MRR:       {result['mrr']:.3f}")


def example_2_batch_evaluation():
    """Evaluate all predefined test scenarios."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Batch Evaluation (All Test Scenarios)")
    print("=" * 80)

    # Load data and models
    try:
        movies_df = pd.read_csv("../../data/movies_final.csv")
        vectorizer = joblib.load("../../models/tfidf_vectorizer.pkl")
        tfidf_matrix = joblib.load("../../models/tfidf_matrix.pkl")
    except (OSError, ValueError, EOFError) as exc:
        print(f"Error loading files: {exc}")
        return

    # Create evaluator
    evaluator = Evaluator(
        recommender_func=recommend_on_the_fly,
        movies_df=movies_df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
    )

    # Evaluate all scenarios
    results = evaluator.evaluate_all_scenarios(top_n=5)

    # Print results
    evaluator.print_results(results)

    # Export to CSV
    evaluator.export_results_csv("evaluation_results.csv", results)


def example_3_view_test_scenarios():
    """View all available test scenarios."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: View Test Scenarios")
    print("=" * 80)

    print_test_scenarios()


def example_4_custom_evaluation():
    """Example of custom evaluation with your own test cases."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Custom Evaluation")
    print("=" * 80)

    # Load data and models
    try:
        movies_df = pd.read_csv("../../data/movies_final.csv")
        vectorizer = joblib.load("../../models/tfidf_vectorizer.pkl")
        tfidf_matrix = joblib.load("../../models/tfidf_matrix.pkl")
    except (OSError, ValueError, EOFError) as exc:
        print(f"Error loading files: {exc}")
        return

    # Create evaluator
    evaluator = Evaluator(
        recommender_func=recommend_on_the_fly,
        movies_df=movies_df,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
    )

    # Define custom test cases
    custom_tests = [
        {
            "user_input": "I want a romantic comedy that's feel-good",
            "relevant_ids": [10, 15, 20, 25],
        },
        {
            "user_input": "Find me a thriller with lots of suspense",
            "relevant_ids": [35, 40, 45, 50],
        },
        {
            "user_input": "I'm looking for a drama from the 1990s",
            "relevant_ids": [60, 65, 70],
        },
    ]

    # Evaluate each custom test
    results = []
    for i, test in enumerate(custom_tests, 1):
        result = evaluator.evaluate_single_query(
            test["user_input"], test["relevant_ids"], top_n=5
        )
        result["test_num"] = i
        results.append(result)

    # Print results
    evaluator.print_results(results)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CINEASSIST METRICS EVALUATION - USAGE EXAMPLES")
    print("=" * 80)

    # Run examples (comment out as needed)
    # example_3_view_test_scenarios()  # Start here to see available scenarios
    # example_1_single_query_evaluation()
    # example_2_batch_evaluation()
    # example_4_custom_evaluation()

    print("\n" + "=" * 80)
    print("INSTRUCTIONS:")
    print("=" * 80)
    print("""
1. First, run example_3_view_test_scenarios() to see all test cases

2. Update src/metrics/test_data.py with actual movieIds from your dataset:
   - Use the movie titles/genres to find matching movieIds
   - Update the relevant_movie_ids lists in TEST_SCENARIOS

3. Then run example_1_single_query_evaluation() to test a single query

4. Finally, run example_2_batch_evaluation() to evaluate all scenarios

5. Check evaluation_results.csv for detailed results
    """)
