"""
Performance & scalability metrics for CineAssist.

This is the THIRD evaluation mode in the metrics module. Where ``evaluator.py``
(Carlos) and ``benchmark.py`` (David) measure *recommendation quality*, this
module measures *system performance*: how fast, how memory-hungry, and how
scalable the recommender and the full conversational pipeline are.

It measures four things, all with the REAL models loaded once up front:

  1. Inference latency  — time ``recommend_on_the_fly`` (the TF-IDF cosine core)
     over a set of representative queries. Reports mean / median / p90 / p95 /
     p99 / min / max in ms. Model-load time is excluded (load once, warm up,
     then time).

  2. End-to-end response time — time the FULL ``run_pipeline`` for (a) English
     queries and (b) Spanish queries (which add MarianMT translation). Reports
     percentiles for each separately plus the translation overhead (ES - EN).

  3. Memory usage — peak process RSS during the suite (``psutil`` if importable,
     else ``resource.getrusage``), the loaded-model footprint (vectorizer vocab
     + TF-IDF matrix bytes), and the ``tracemalloc`` peak allocation of a single
     pipeline call.

  4. Scalability / throughput — queries-per-second for sequential execution, and
     how latency / QPS vary with ``top_n``. Kept deliberately small/bounded so it
     stays quick and does not hammer the machine.

The suite loads models/data ONCE, warms up, then times. Nothing here touches or
instruments the pipeline internals — everything is measured by wrapping and
timing public calls.

Run:
    python -m src.metrics.performance
"""

import gc
import glob
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.recommender.recommender_engine import recommend_on_the_fly
from src.chatbot.chatbot_flow import run_pipeline, initialize_conversation_state

# ---------------------------------------------------------------------------
# Representative prompts (reused from the live benchmark style). English set is
# used for inference-latency + EN end-to-end + scalability. Spanish set exercises
# the MarianMT translation path for the ES end-to-end comparison.
# ---------------------------------------------------------------------------
EN_QUERIES = [
    "a funny family movie from the 90s",
    "scary horror movie about a haunted house",
    "a romantic love story",
    "psychological thriller with a twist",
    "action packed superhero movie",
    "an animated movie for kids",
    "science fiction space adventure with aliens",
    "a dark crime drama about the mafia",
    "a war movie about soldiers",
    "fantasy adventure with magic and dragons",
]

ES_QUERIES = [
    "una comedia familiar divertida de los 90",
    "una pelicula de terror sobre una casa embrujada",
    "una historia de amor romantica",
    "un thriller psicologico con un giro inesperado",
    "una pelicula de superheroes llena de accion",
    "una pelicula animada para ninos",
    "una aventura de ciencia ficcion en el espacio con extraterrestres",
    "un drama criminal oscuro sobre la mafia",
    "una pelicula de guerra sobre soldados",
    "una aventura de fantasia con magia y dragones",
]


# ---------------------------------------------------------------------------
# Asset loading (model-load time is deliberately outside the timed sections).
# ---------------------------------------------------------------------------
def load_assets():
    """Load dataset + TF-IDF vectorizer/matrix from the standard project paths."""
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


# ---------------------------------------------------------------------------
# Small stats helpers.
# ---------------------------------------------------------------------------
def _percentile(sorted_vals, pct):
    """Linear-interpolated percentile (pct in [0, 100]) over a sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


def _summarize_ms(samples):
    """Turn a list of millisecond timings into a percentile summary dict."""
    if not samples:
        return {k: 0.0 for k in
                ("count", "mean", "median", "p90", "p95", "p99", "min", "max", "stdev")}
    s = sorted(samples)
    return {
        "count": len(s),
        "mean": statistics.fmean(s),
        "median": statistics.median(s),
        "p90": _percentile(s, 90),
        "p95": _percentile(s, 95),
        "p99": _percentile(s, 99),
        "min": s[0],
        "max": s[-1],
        "stdev": statistics.stdev(s) if len(s) > 1 else 0.0,
    }


def _time_call(fn, *args, **kwargs):
    """Return elapsed milliseconds for a single call to ``fn``."""
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Memory helpers — psutil if available, otherwise resource.getrusage.
# ---------------------------------------------------------------------------
def _peak_rss_mb():
    """Peak resident set size of THIS process, in MB. psutil optional."""
    try:
        import psutil  # optional; do NOT add as a dependency
        proc = psutil.Process()
        mem = proc.memory_info()
        # peak/high-water-mark if the platform exposes it, else current RSS.
        peak = getattr(mem, "peak_wset", None) or getattr(mem, "rss", 0)
        return peak / (1024 * 1024)
    except Exception:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports ru_maxrss in KB, macOS in bytes.
        if sys.platform == "darwin":
            return ru / (1024 * 1024)
        return ru / 1024


def _model_footprint(vectorizer, tfidf_matrix):
    """In-memory footprint of the loaded TF-IDF assets."""
    data = getattr(tfidf_matrix, "data", None)
    indices = getattr(tfidf_matrix, "indices", None)
    indptr = getattr(tfidf_matrix, "indptr", None)
    matrix_bytes = 0
    if data is not None and indices is not None and indptr is not None:
        matrix_bytes = data.nbytes + indices.nbytes + indptr.nbytes
    vocab = getattr(vectorizer, "vocabulary_", {}) or {}
    return {
        "matrix_bytes": int(matrix_bytes),
        "matrix_mb": matrix_bytes / (1024 * 1024),
        "matrix_shape": list(getattr(tfidf_matrix, "shape", (0, 0))),
        "matrix_nnz": int(getattr(tfidf_matrix, "nnz", 0)),
        "vocab_size": len(vocab),
    }


# ---------------------------------------------------------------------------
# The suite.
# ---------------------------------------------------------------------------
def run_performance_suite(
    df=None,
    vec=None,
    mat=None,
    latency_reps=40,
    e2e_reps=15,
    scale_reps=40,
    top_n_configs=(5, 10, 25),
    include_spanish=True,
    warmup=2,
):
    """
    Run the full performance & scalability suite and return a structured dict.

    Assets (df/vec/mat) are loaded from the standard project paths if not passed.
    Everything is measured with the models already loaded and warmed up, so
    model-load time never leaks into the reported numbers.

    Returns a nested dict::

        {
          "meta": {...dataset/matrix info...},
          "latency_ms": {mean, median, p90, p95, p99, min, max, ...},
          "end_to_end_ms": {
              "en": {...percentiles...},
              "es": {...percentiles...} | None,
              "translation_overhead_ms": float | None,
          },
          "memory": {peak_rss_mb, model_footprint_mb, vocab_size,
                     tracemalloc_peak_mb, ...},
          "scalability": {qps, total_time_s, n_queries, by_top_n: {...}},
        }

    Repetition counts are intentionally small/bounded (a few dozen calls per
    config) so the suite stays quick and light on the machine.
    """
    if df is None or vec is None or mat is None:
        df, vec, mat, csv_name = load_assets()
    else:
        csv_name = "<provided>"

    results = {"meta": {
        "dataset": csv_name,
        "n_movies": int(len(df)),
        "matrix_shape": list(getattr(mat, "shape", (0, 0))),
        "latency_reps": latency_reps,
        "e2e_reps": e2e_reps,
        "scale_reps": scale_reps,
        "top_n_configs": list(top_n_configs),
    }}

    # ── Warm up: first calls compile/allocate lazily; don't time those. ────────
    for q in EN_QUERIES[:max(1, warmup)]:
        recommend_on_the_fly(q, df, vec, mat, top_n=5)
        run_pipeline(q, initialize_conversation_state(), df, vec, mat, top_n=5)

    # ── 1. Inference latency — the TF-IDF cosine core in isolation. ────────────
    lat_samples = []
    for i in range(latency_reps):
        q = EN_QUERIES[i % len(EN_QUERIES)]
        lat_samples.append(
            _time_call(recommend_on_the_fly, q, df, vec, mat, top_n=5)
        )
    results["latency_ms"] = _summarize_ms(lat_samples)

    # ── 2. End-to-end — full run_pipeline, EN and (optionally) ES. ─────────────
    en_samples = []
    for i in range(e2e_reps):
        q = EN_QUERIES[i % len(EN_QUERIES)]
        en_samples.append(
            _time_call(run_pipeline, q, initialize_conversation_state(),
                       df, vec, mat, top_n=5)
        )
    en_summary = _summarize_ms(en_samples)

    es_summary = None
    overhead = None
    if include_spanish:
        # Warm the translation models first. The first call for a given source
        # language downloads/loads its MarianMT model, and langdetect can tag a
        # short Spanish prompt as e.g. Italian — pulling in a *second* model
        # mid-run. Warming EVERY ES prompt once ensures all such one-time model
        # loads happen OUTSIDE the timed samples (otherwise they blow up the tail).
        for q in ES_QUERIES:
            run_pipeline(q, initialize_conversation_state(),
                         df, vec, mat, top_n=5)
        es_samples = []
        for i in range(e2e_reps):
            q = ES_QUERIES[i % len(ES_QUERIES)]
            es_samples.append(
                _time_call(run_pipeline, q, initialize_conversation_state(),
                           df, vec, mat, top_n=5)
            )
        es_summary = _summarize_ms(es_samples)
        overhead = es_summary["median"] - en_summary["median"]

    results["end_to_end_ms"] = {
        "en": en_summary,
        "es": es_summary,
        "translation_overhead_ms": overhead,
    }

    # ── 3. Memory — model footprint, tracemalloc for one call, peak RSS. ───────
    footprint = _model_footprint(vec, mat)

    gc.collect()
    tracemalloc.start()
    run_pipeline(EN_QUERIES[0], initialize_conversation_state(),
                 df, vec, mat, top_n=5)
    _cur, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results["memory"] = {
        "peak_rss_mb": _peak_rss_mb(),
        "model_footprint_mb": footprint["matrix_mb"],
        "model_footprint": footprint,
        "vocab_size": footprint["vocab_size"],
        "tracemalloc_peak_mb": tm_peak / (1024 * 1024),
    }

    # ── 4. Scalability / throughput — QPS sequential, and latency vs top_n. ────
    by_top_n = {}
    for tn in top_n_configs:
        samples = []
        for i in range(scale_reps):
            q = EN_QUERIES[i % len(EN_QUERIES)]
            samples.append(
                _time_call(recommend_on_the_fly, q, df, vec, mat, top_n=tn)
            )
        total_s = sum(samples) / 1000.0
        by_top_n[tn] = {
            "latency_ms": _summarize_ms(samples),
            "qps": (len(samples) / total_s) if total_s > 0 else 0.0,
            "total_time_s": total_s,
            "n_queries": len(samples),
        }

    # Headline QPS uses the default top_n=5 config.
    headline = by_top_n.get(top_n_configs[0], next(iter(by_top_n.values())))
    results["scalability"] = {
        "qps": headline["qps"],
        "total_time_s": headline["total_time_s"],
        "n_queries": headline["n_queries"],
        "by_top_n": by_top_n,
    }

    return results


# ---------------------------------------------------------------------------
# Pretty printer for the CLI.
# ---------------------------------------------------------------------------
def _fmt_pctile(d):
    return (f"mean {d['mean']:.1f}  median {d['median']:.1f}  "
            f"p90 {d['p90']:.1f}  p95 {d['p95']:.1f}  p99 {d['p99']:.1f}  "
            f"min {d['min']:.1f}  max {d['max']:.1f}  (n={d['count']})")


def print_report(results):
    """Pretty-print a run_performance_suite() result dict."""
    meta = results["meta"]
    line = "=" * 72
    print(line)
    print("CineAssist — Performance & Scalability Report")
    print(line)
    print(f"Dataset        : {meta['dataset']}  ({meta['n_movies']:,} movies)")
    print(f"TF-IDF matrix  : {meta['matrix_shape'][0]:,} x {meta['matrix_shape'][1]:,}")
    print()

    print("1. INFERENCE LATENCY  (recommend_on_the_fly, TF-IDF cosine core) [ms]")
    print(f"   {_fmt_pctile(results['latency_ms'])}")
    print()

    e2e = results["end_to_end_ms"]
    print("2. END-TO-END RESPONSE TIME  (full run_pipeline) [ms]")
    print(f"   EN : {_fmt_pctile(e2e['en'])}")
    if e2e["es"] is not None:
        print(f"   ES : {_fmt_pctile(e2e['es'])}")
        print(f"   Translation overhead (ES median - EN median): "
              f"{e2e['translation_overhead_ms']:.1f} ms")
    else:
        print("   ES : (skipped)")
    print()

    mem = results["memory"]
    fp = mem["model_footprint"]
    print("3. MEMORY USAGE")
    print(f"   Peak process RSS      : {mem['peak_rss_mb']:.1f} MB")
    print(f"   TF-IDF matrix in-mem  : {fp['matrix_mb']:.1f} MB "
          f"(nnz={fp['matrix_nnz']:,}, shape={fp['matrix_shape'][0]:,}"
          f"x{fp['matrix_shape'][1]:,})")
    print(f"   Vocabulary size       : {mem['vocab_size']:,} terms")
    print(f"   tracemalloc peak (1 pipeline call): {mem['tracemalloc_peak_mb']:.1f} MB")
    print()

    scale = results["scalability"]
    print("4. SCALABILITY / THROUGHPUT")
    print(f"   Sequential QPS (top_n={meta['top_n_configs'][0]}): "
          f"{scale['qps']:.1f} queries/s  "
          f"({scale['n_queries']} queries in {scale['total_time_s']:.2f}s)")
    print("   Latency & QPS by top_n:")
    print(f"     {'top_n':>6} {'mean ms':>9} {'p95 ms':>8} {'qps':>8}")
    for tn, d in scale["by_top_n"].items():
        lat = d["latency_ms"]
        print(f"     {tn:>6} {lat['mean']:>9.1f} {lat['p95']:>8.1f} {d['qps']:>8.1f}")
    print(line)


def main():
    print("Loading models & dataset (not timed)...", flush=True)
    df, vec, mat, csv_name = load_assets()
    print(f"Loaded {csv_name}: {len(df):,} movies, matrix {mat.shape}.\n", flush=True)
    results = run_performance_suite(df, vec, mat)
    print_report(results)


if __name__ == "__main__":
    main()
