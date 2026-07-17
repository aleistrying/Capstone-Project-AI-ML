"""
Centralized metrics & evaluation module for CineAssist.

Two complementary evaluation modes live here:

1. ID-based ground-truth evaluation (Carlos)
   - Pure metric functions: precision / recall / accuracy / f1 / MRR over
     movie-ID lists (``metrics.py``).
   - ``Evaluator`` runs a recommender over labelled scenarios and compares
     predicted movieIds against ground-truth relevant_movie_ids
     (``evaluator.py`` + ``test_data.py``).

2. Live-pipeline genre-overlap benchmark (David)
   - ``run_benchmark`` / ``BENCHMARK`` run a fixed prompt set through the live
     ``get_chat_recommendations`` pipeline and score genre overlap
     (``benchmark.py``). Also runnable as ``python -m src.metrics.benchmark``.
"""

from .metrics import (
    calculate_precision,
    calculate_recall,
    calculate_accuracy,
    calculate_f1_score,
    calculate_mean_reciprocal_rank,
)
from .evaluator import Evaluator
from .test_data import get_test_scenarios, get_test_scenario, TEST_SCENARIOS
from .benchmark import run_benchmark, evaluate_query, BENCHMARK

__all__ = [
    # ID-based ground-truth evaluation (Carlos)
    "calculate_precision",
    "calculate_recall",
    "calculate_accuracy",
    "calculate_f1_score",
    "calculate_mean_reciprocal_rank",
    "Evaluator",
    "get_test_scenarios",
    "get_test_scenario",
    "TEST_SCENARIOS",
    # Live-pipeline genre-overlap benchmark (David)
    "run_benchmark",
    "evaluate_query",
    "BENCHMARK",
]
