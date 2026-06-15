"""
Main evaluator class for comprehensive chatbot recommendation evaluation.
"""

from .metrics import (
    calculate_precision,
    calculate_recall,
    calculate_accuracy,
    calculate_f1_score,
    calculate_mean_reciprocal_rank
)
from .test_data import get_test_scenarios


class Evaluator:
    """
    Evaluates chatbot recommendation quality using precision, recall, and accuracy.
    """

    def __init__(self, recommender_func, movies_df, vectorizer, tfidf_matrix=None):
        """
        Initialize the evaluator.

        Args:
            recommender_func: Function that takes query and returns recommended movie IDs
            movies_df: DataFrame containing movie data
            vectorizer: TF-IDF vectorizer (or any vectorizer used)
            tfidf_matrix: Pre-computed TF-IDF matrix (optional)
        """
        self.recommender_func = recommender_func
        self.movies_df = movies_df
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.total_movies = len(movies_df) if movies_df is not None else 0
        self.results = []

    def evaluate_single_query(self, user_input, relevant_ids, top_n=5):
        """
        Evaluate a single user query.

        Args:
            user_input: Natural language user input
            relevant_ids: List of movie IDs that are relevant
            top_n: Number of recommendations to return

        Returns:
            dict: Evaluation metrics for this query
        """
        # Get recommendations from the recommender
        query_text = user_input
        recommendations = self.recommender_func(
            query_text,
            self.movies_df,
            self.vectorizer,
            self.tfidf_matrix,
            top_n=top_n
        )

        # Extract movie IDs from recommendations (assuming movieId column)
        if recommendations.empty:
            predicted_ids = []
        else:
            # Handle different possible column names
            if 'movieId' in recommendations.columns:
                predicted_ids = recommendations['movieId'].tolist()
            elif 'id' in recommendations.columns:
                predicted_ids = recommendations['id'].tolist()
            else:
                predicted_ids = []

        # Calculate metrics
        precision = calculate_precision(predicted_ids, relevant_ids)
        recall = calculate_recall(predicted_ids, relevant_ids)
        f1 = calculate_f1_score(predicted_ids, relevant_ids)
        mrr = calculate_mean_reciprocal_rank(predicted_ids, relevant_ids)

        # Accuracy requires total items in dataset
        accuracy = calculate_accuracy(predicted_ids, relevant_ids, self.total_movies)

        result = {
            "user_input": user_input,
            "predicted_ids": predicted_ids,
            "relevant_ids": relevant_ids,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1_score": f1,
            "mrr": mrr,
            "top_n": top_n
        }

        self.results.append(result)
        return result

    def evaluate_all_scenarios(self, top_n=5):
        """
        Evaluate all predefined test scenarios.

        Args:
            top_n: Number of recommendations to return per query

        Returns:
            list: Results for all scenarios
        """
        scenarios = get_test_scenarios()
        results = []

        for scenario in scenarios:
            result = self.evaluate_single_query(
                scenario["user_input"],
                scenario["relevant_movie_ids"],
                top_n=top_n
            )
            result["scenario_id"] = scenario["id"]
            result["scenario_description"] = scenario["description"]
            results.append(result)

        return results

    def get_average_metrics(self, results=None):
        """
        Calculate average metrics across all evaluations.

        Args:
            results: List of result dicts. If None, uses self.results

        Returns:
            dict: Average metrics
        """
        if results is None:
            results = self.results

        if not results:
            return {}

        n = len(results)
        avg_precision = sum(r["precision"] for r in results) / n
        avg_recall = sum(r["recall"] for r in results) / n
        avg_accuracy = sum(r["accuracy"] for r in results) / n
        avg_f1 = sum(r["f1_score"] for r in results) / n
        avg_mrr = sum(r["mrr"] for r in results) / n

        return {
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_accuracy": avg_accuracy,
            "avg_f1_score": avg_f1,
            "avg_mrr": avg_mrr,
            "total_tests": n
        }

    def print_results(self, results=None):
        """
        Print evaluation results in a readable format.

        Args:
            results: List of result dicts. If None, uses self.results
        """
        if results is None:
            results = self.results

        if not results:
            print("No evaluation results available.")
            return

        print("\n" + "=" * 80)
        print("CINEASSIST RECOMMENDATION EVALUATION RESULTS")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            scenario_id = result.get("scenario_id", f"Query {i}")
            description = result.get("scenario_description", "")

            print(f"\n[{scenario_id}] {description}")
            print(f"User Input: {result['user_input']}")
            print(f"\nRecommended Movie IDs: {result['predicted_ids']}")
            print(f"Relevant Movie IDs:    {result['relevant_ids']}")
            print(f"\nMetrics:")
            print(f"  Precision:  {result['precision']:.3f}")
            print(f"  Recall:     {result['recall']:.3f}")
            print(f"  Accuracy:   {result['accuracy']:.3f}")
            print(f"  F1 Score:   {result['f1_score']:.3f}")
            print(f"  MRR:        {result['mrr']:.3f}")

        # Print averages
        avg_metrics = self.get_average_metrics(results)
        print("\n" + "=" * 80)
        print("AVERAGE METRICS")
        print("=" * 80)
        print(f"Average Precision:  {avg_metrics['avg_precision']:.3f}")
        print(f"Average Recall:     {avg_metrics['avg_recall']:.3f}")
        print(f"Average Accuracy:   {avg_metrics['avg_accuracy']:.3f}")
        print(f"Average F1 Score:   {avg_metrics['avg_f1_score']:.3f}")
        print(f"Average MRR:        {avg_metrics['avg_mrr']:.3f}")
        print(f"Total Tests:        {avg_metrics['total_tests']}")
        print("=" * 80 + "\n")

    def export_results_csv(self, filename, results=None):
        """
        Export evaluation results to CSV file.

        Args:
            filename: Output CSV file path
            results: List of result dicts. If None, uses self.results
        """
        import csv

        if results is None:
            results = self.results

        if not results:
            print("No results to export.")
            return

        keys = [
            "scenario_id", "user_input", "predicted_ids", "relevant_ids",
            "precision", "recall", "accuracy", "f1_score", "mrr"
        ]

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()

            for result in results:
                row = {key: result.get(key, "") for key in keys}
                writer.writerow(row)

        print(f"Results exported to {filename}")
