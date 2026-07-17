"""
Main evaluator class for comprehensive chatbot recommendation evaluation.

RETRIEVAL ALIGNMENT (why this evaluator mirrors the deployed pipeline)
----------------------------------------------------------------------
An earlier version fed the RAW user sentence straight into the recommender, so
filler words ("I would like to see a movie that ...") diluted the query vector
and retrieval was far worse than the live app -- which drove every metric to
0.00. The deployed system never does that: `run_pipeline` (src.chatbot.
chatbot_flow) first builds a FOCUSED query (keyword extraction + genre-vocabulary
expansion) and a structured filter `state_dict` (language / rating / year), then
calls `recommend_on_the_fly`.

This evaluator now reproduces that same Stage-1/2/3 query construction before
retrieval, using the exact building blocks the pipeline uses:
    - `extract_preferences` (src.nlp.nlp_preferences) -> genres/mood/language/
      year_range/min_rating,
    - `build_query`         (src.nlp.keyword_extractor) -> focused query string,
constructed identically to `run_pipeline`, then passes the focused query and
`state_dict` to the recommender. We call `recommend_on_the_fly` directly (rather
than `run_pipeline`) because its returned DataFrame INCLUDES the `movieId`
column, which the ID-based metrics need -- `run_pipeline`'s trace.recommendations
dicts do not carry movieId. Chatbot_flow is intentionally NOT imported so the
metrics module stays free of the optional translation dependencies.
"""

from .metrics import (
    calculate_precision,
    calculate_recall,
    calculate_accuracy,
    calculate_f1_score,
    calculate_mean_reciprocal_rank
)
from .test_data import get_test_scenarios

from src.nlp.nlp_preferences import extract_preferences
from src.nlp.keyword_extractor import build_genre_vocabulary, build_query


class Evaluator:
    """
    Evaluates chatbot recommendation quality using precision, recall, and accuracy.
    """

    def __init__(self, recommender_func, movies_df, vectorizer, tfidf_matrix=None):
        """
        Initialize the evaluator.

        Args:
            recommender_func: Recommender callable. Expected signature matches
                `recommend_on_the_fly(query, movies_df, vectorizer, tfidf_matrix,
                state_dict=..., top_n=...)` and it must return a DataFrame with a
                `movieId` column.
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

        # Query-building assets, mirroring run_pipeline: the dataset genre
        # vocabulary (for genre detection) and the model vocabulary (so thematic
        # expansion never introduces out-of-vocabulary noise).
        self._genre_vocab = (
            build_genre_vocabulary(movies_df) if movies_df is not None else set()
        )
        self._model_vocab = set(getattr(vectorizer, "vocabulary_", {})) or None

    def _build_focused_query(self, user_input):
        """
        Turn a raw user sentence into the (focused_query, state_dict) pair the
        deployed pipeline would use. Mirrors run_pipeline Stages 1-3.

        Returns:
            (query_text, state_dict) where state_dict carries the language /
            rating / year filters the recommender applies.
        """
        prefs = extract_preferences(user_input)
        extracted = build_query(user_input, self._genre_vocab, vocab=self._model_vocab)

        query_parts = (
            (prefs.get("genres") or [])
            + (prefs.get("mood") or [])
            + extracted["entities"]["genres"]
            + extracted["query"].split()
        )
        query_text = " ".join(dict.fromkeys(p for p in query_parts if p)).strip()

        state_dict = {
            "language": prefs.get("language"),
            "rating": prefs.get("min_rating") or prefs.get("rating"),
            "year": prefs.get("year_range")[0] if prefs.get("year_range") else None,
        }
        return query_text, state_dict

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
        # Build the focused query + filters exactly like the deployed pipeline,
        # then retrieve. This is what makes predicted IDs come from the SAME path
        # the live app uses (see module docstring).
        query_text, state_dict = self._build_focused_query(user_input)

        if not query_text:
            recommendations = self.movies_df.iloc[0:0]
        else:
            try:
                recommendations = self.recommender_func(
                    query_text,
                    self.movies_df,
                    self.vectorizer,
                    self.tfidf_matrix,
                    state_dict=state_dict,
                    top_n=top_n,
                )
            except TypeError:
                # Fallback for a recommender that doesn't accept state_dict.
                recommendations = self.recommender_func(
                    query_text,
                    self.movies_df,
                    self.vectorizer,
                    self.tfidf_matrix,
                    top_n=top_n,
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
            "focused_query": query_text,   # what the pipeline actually retrieves on
            "filters": state_dict,         # language / rating / year applied
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
