"""
CineAssist recommendation benchmark — consistent, repeatable evaluation.

Runs a FIXED set of labelled prompts through the live recommendation pipeline
(get_chat_recommendations) and reports, per query and in aggregate:
  - precision@k   : fraction of the top-k results whose genres overlap the
                    expected genres for that query (automatic relevance proxy)
  - match%        : mean cosine-similarity (decade-aware) of the top-k results
  - pass/fail     : precision@k >= PASS_THRESHOLD

This gives David (recommender) and Carlos (metrics) ONE consistent target so
methodology changes can be compared run-to-run. Extend BENCHMARK to ~100 prompts
as agreed in Meeting 7.

Run:
    python src/evaluation/benchmark.py
"""

import glob
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.chatbot.chatbot_flow import get_chat_recommendations, initialize_conversation_state

TOP_K = 5
PASS_THRESHOLD = 0.40  # a query "passes" if >= 40% of top-k results are on-genre

# ---------------------------------------------------------------------------
# Labelled benchmark set — extend toward ~100 prompts.
# expected_genres are matched case-insensitively against each movie's genres_list.
# ---------------------------------------------------------------------------
BENCHMARK = [
    {"query": "a funny family movie from the 90s",                 "expected": {"comedy", "family"}},
    {"query": "scary horror movie about a haunted house",          "expected": {"horror"}},
    {"query": "a romantic love story",                             "expected": {"romance"}},
    {"query": "psychological thriller with a twist",               "expected": {"thriller", "mystery"}},
    {"query": "action packed superhero movie",                     "expected": {"action", "adventure"}},
    {"query": "an animated movie for kids",                        "expected": {"animation", "family"}},
    {"query": "science fiction space adventure with aliens",       "expected": {"science fiction", "adventure"}},
    {"query": "a dark crime drama about the mafia",                "expected": {"crime", "drama"}},
    {"query": "a war movie about soldiers",                        "expected": {"war"}},
    {"query": "fantasy adventure with magic and dragons",          "expected": {"fantasy", "adventure"}},
    {"query": "something like Inception, dark and mind-bending",   "expected": {"science fiction", "thriller", "mystery"}},
    {"query": "a western with cowboys",                            "expected": {"western"}},
    {"query": "a feel-good comedy to relax",                       "expected": {"comedy"}},
    {"query": "a documentary about nature",                        "expected": {"documentary"}},
]


def load_assets():
    csv = glob.glob(str(ROOT / "data" / "processed" / "*.csv"))[0]
    df = pd.read_csv(csv)
    vec = joblib.load(ROOT / "models" / "tfidf_vectorizer.pkl")
    npz = ROOT / "models" / "tfidf_matrix.npz"
    if npz.exists():
        from scipy.sparse import load_npz
        mat = load_npz(str(npz))
    else:
        mat = joblib.load(ROOT / "models" / "tfidf_matrix.pkl")
    return df, vec, mat, Path(csv).name


def _genres_lower(rec) -> set[str]:
    return {str(g).lower() for g in (rec.get("genres") or [])}


def evaluate_query(case, df, vec, mat):
    _, recs, _, _ = get_chat_recommendations(
        case["query"], initialize_conversation_state(), df, vec, mat
    )
    top = recs[:TOP_K]
    expected = {g.lower() for g in case["expected"]}

    hits = [bool(_genres_lower(r) & expected) for r in top]
    relevant = sum(hits)
    precision = relevant / max(1, len(top))
    # recall proxy: did we surface at least one on-genre title in the top-k?
    recall = 1.0 if relevant > 0 else 0.0
    match = (sum(r["similarity"] for r in top) / len(top)) if top else 0.0

    return {
        "query": case["query"],
        "precision": precision,
        "recall": recall,
        "match": match,
        "passed": precision >= PASS_THRESHOLD,
        "top_title": top[0]["title"] if top else "—",
        "top_match": top[0]["similarity"] if top else 0.0,
    }


def main():
    df, vec, mat, csv_name = load_assets()
    print(f"Dataset: {csv_name}  ({len(df):,} movies)   |   matrix: {mat.shape}")
    print(f"Top-k = {TOP_K}   pass threshold = precision@{TOP_K} >= {PASS_THRESHOLD:.0%}\n")

    rows = [evaluate_query(c, df, vec, mat) for c in BENCHMARK]

    header = f"{'Query':<48} {'P@5':>5} {'Rec':>4} {'Match%':>7}  {'Pass':>4}  Top result"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['query'][:46]:<48} {r['precision']*100:>4.0f}% {r['recall']:>4.0f} "
            f"{r['match']*100:>6.1f}%  {'PASS' if r['passed'] else 'FAIL':>4}  "
            f"{r['top_title'][:30]} ({r['top_match']*100:.0f}%)"
        )

    n = len(rows)
    print("-" * len(header))
    print(
        f"{'AGGREGATE':<48} {sum(r['precision'] for r in rows)/n*100:>4.0f}% "
        f"{sum(r['recall'] for r in rows)/n:>4.2f} "
        f"{sum(r['match'] for r in rows)/n*100:>6.1f}%  "
        f"{sum(r['passed'] for r in rows)}/{n} pass"
    )


if __name__ == "__main__":
    main()
