"""
CineAssist — Performance & Scalability

Runs the runtime performance suite (src/metrics/performance.py) and shows the
metrics the professor asked for: inference latency, end-to-end response time
(English vs Spanish), memory footprint, and throughput/scalability.

Everything here comes from the ONE centralized pipeline (run_pipeline) and the
same TF-IDF models the live app uses — so these numbers reflect the real system.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.performance import run_performance_suite

st.set_page_config(page_title="CineAssist — Performance", page_icon="⚡", layout="wide")
st.title("⚡ Performance & Scalability")
st.caption(
    "Inference latency · end-to-end response time (EN vs ES) · memory footprint · "
    "throughput — measured on the real TF-IDF models via the centralized pipeline."
)

with st.sidebar:
    st.subheader("Suite settings")
    latency_reps = st.slider("Latency reps", 10, 100, 30, step=10)
    e2e_reps = st.slider("End-to-end reps (per language)", 5, 30, 10, step=5)
    include_spanish = st.checkbox("Include Spanish (loads MarianMT)", value=True)
    st.caption("Larger rep counts give tighter percentiles but take longer.")

if not st.button("▶ Run performance suite", type="primary"):
    st.info(
        "Click **Run performance suite** to benchmark the recommender. "
        "The first Spanish run warms up the translation model (one-time)."
    )
    st.stop()

with st.spinner("Benchmarking (loading models, warming up, timing)…"):
    results = run_performance_suite(
        latency_reps=latency_reps,
        e2e_reps=e2e_reps,
        include_spanish=include_spanish,
    )

lat = results.get("latency_ms", {})
e2e = results.get("end_to_end_ms", {})
mem = results.get("memory", {})
scal = results.get("scalability", {})

st.subheader("Inference latency — recommend_on_the_fly (ms)")
c = st.columns(5)
c[0].metric("Mean", f"{lat.get('mean', 0):.1f}")
c[1].metric("Median", f"{lat.get('median', 0):.1f}")
c[2].metric("p90", f"{lat.get('p90', 0):.1f}")
c[3].metric("p95", f"{lat.get('p95', 0):.1f}")
c[4].metric("p99", f"{lat.get('p99', 0):.1f}")

st.subheader("End-to-end response time — run_pipeline (ms)")
en = e2e.get("en", {})
es = e2e.get("es", {})
c = st.columns(4)
c[0].metric("EN median", f"{en.get('median', 0):.1f}")
c[1].metric("EN p95", f"{en.get('p95', 0):.1f}")
if es:
    c[2].metric("ES median", f"{es.get('median', 0):.1f}")
    c[3].metric("Translation overhead", f"{e2e.get('translation_overhead_ms', 0):.1f} ms")

st.subheader("Memory")
c = st.columns(4)
c[0].metric("Peak RSS (MB)", f"{mem.get('peak_rss_mb', 0):.0f}")
c[1].metric("Model footprint (MB)", f"{mem.get('model_footprint_mb', 0):.1f}")
c[2].metric("Vocabulary size", f"{mem.get('vocab_size', 0):,}")
c[3].metric("1-call peak alloc (MB)", f"{mem.get('tracemalloc_peak_mb', 0):.1f}")

st.subheader("Scalability")
c = st.columns(2)
c[0].metric("Throughput (QPS)", f"{scal.get('qps', 0):.1f}")
c[1].metric("Queries timed", f"{scal.get('n_queries', 0)}")
by_top_n = scal.get("by_top_n")
if by_top_n:
    st.write("**Latency / throughput vs top_n**")
    st.json(by_top_n)

with st.expander("Raw results (full dict)"):
    st.json(results)
