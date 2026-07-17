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

3. Translation quality (BLEU + human evaluation)
   - ``evaluate_translation_quality`` / ``compute_bleu`` / ``compare_pre_post``
     score MarianMT translation with sacreBLEU (pre- vs post-fine-tune), plus a
     human-eval template/aggregator (``translation_quality.py``).
     CLI: ``python -m src.metrics.translation_quality``.

4. Runtime performance & scalability
   - ``run_performance_suite`` measures inference latency, end-to-end response
     time (EN vs ES), memory footprint, and throughput/scalability
     (``performance.py``). CLI: ``python -m src.metrics.performance``.

The heavy dependencies used by (3) and (4) — torch/transformers/sacrebleu — are
imported lazily inside those modules, so ``import src.metrics`` stays light.
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
from .translation_quality import (
    evaluate_translation_quality,
    compute_bleu,
    compare_pre_post,
    export_human_eval_template,
    aggregate_human_scores,
)
from .performance import run_performance_suite, print_report

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
    # Translation quality — BLEU + human eval
    "evaluate_translation_quality",
    "compute_bleu",
    "compare_pre_post",
    "export_human_eval_template",
    "aggregate_human_scores",
    # Runtime performance & scalability
    "run_performance_suite",
    "print_report",
]
