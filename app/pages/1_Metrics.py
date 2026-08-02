"""
CineAssist — Metrics Page

Shows Precision@5, Recall@5, F1, MRR, and Accuracy for all predefined
test scenarios. Runs the evaluation on demand.
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.assets import load_assets_or_stop
from src.metrics.evaluator import Evaluator
from src.metrics.test_data import get_test_scenarios
from src.recommender.recommender_engine import recommend_on_the_fly

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CineAssist — Metrics", page_icon="📊", layout="wide")
st.title("📊 Evaluation Metrics")
st.caption(
    "Precision@5 · Recall@5 · F1 · MRR · Accuracy across 10 predefined test scenarios"
)

# ---------------------------------------------------------------------------
# Asset loading — shared cache with the main app and every other page
# ---------------------------------------------------------------------------

movies_df, vectorizer, tfidf_matrix = load_assets_or_stop()

# ---------------------------------------------------------------------------
# Sidebar — test scenario info
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Test Scenarios")
    scenarios = get_test_scenarios()
    for s in scenarios:
        st.markdown(f"**{s['id']}**  \n{s['description']}")
    st.divider()
    top_n = st.slider(
        "Recommendations per query (top-N)", min_value=3, max_value=10, value=5
    )

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

st.subheader("Run Evaluation")

if st.button("▶ Run all scenarios", type="primary"):
    with st.spinner("Evaluating all scenarios…"):
        evaluator = Evaluator(
            recommender_func=recommend_on_the_fly,
            movies_df=movies_df,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
        )
        t0 = time.perf_counter()
        results = evaluator.evaluate_all_scenarios(top_n=top_n)
        elapsed = time.perf_counter() - t0
        avg = evaluator.get_average_metrics(results)

    st.success(f"Completed {len(results)} scenarios in {elapsed:.1f}s")

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    st.subheader("Average Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision@5", f"{avg['avg_precision']:.3f}")
    c2.metric("Recall@5", f"{avg['avg_recall']:.3f}")
    c3.metric("F1 Score", f"{avg['avg_f1_score']:.3f}")
    c4.metric("MRR", f"{avg['avg_mrr']:.3f}")
    c5.metric("Accuracy", f"{avg['avg_accuracy']:.3f}")

    # ── Per-scenario table ────────────────────────────────────────────────────
    st.subheader("Per-Scenario Results")
    rows = [
        {
            "Scenario": result.get("scenario_id", ""),
            "Description": result.get("scenario_description", ""),
            "Precision": round(result["precision"], 3),
            "Recall": round(result["recall"], 3),
            "F1": round(result["f1_score"], 3),
            "MRR": round(result["mrr"], 3),
            "Accuracy": round(result["accuracy"], 3),
        }
        for result in results
    ]
    df_results = pd.DataFrame(rows)
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    # ── Bar chart: Precision & Recall per scenario ────────────────────────────
    st.subheader("Precision vs Recall")
    chart_df = df_results.set_index("Scenario")[["Precision", "Recall", "F1"]]
    st.bar_chart(chart_df)

    # ── MRR chart ─────────────────────────────────────────────────────────────
    st.subheader("MRR per Scenario")
    mrr_df = df_results.set_index("Scenario")[["MRR"]]
    st.bar_chart(mrr_df)

    # ── Detail expander per scenario ─────────────────────────────────────────
    st.subheader("Details")
    for r in results:
        with st.expander(
            f"{r.get('scenario_id')} — {r.get('scenario_description', '')}"
        ):
            st.write(f"**Query:** {r['user_input']}")
            col1, col2 = st.columns(2)
            col1.write("**Recommended IDs**")
            col1.write(r["predicted_ids"])
            col2.write("**Relevant IDs (ground truth)**")
            col2.write(r["relevant_ids"])
            hits = set(r["predicted_ids"]) & set(r["relevant_ids"])
            st.write(f"**Hits:** {sorted(hits) if hits else 'none'}")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Precision", f"{r['precision']:.3f}")
            m2.metric("Recall", f"{r['recall']:.3f}")
            m3.metric("F1", f"{r['f1_score']:.3f}")
            m4.metric("MRR", f"{r['mrr']:.3f}")
            m5.metric("Accuracy", f"{r['accuracy']:.3f}")

    # ── Export CSV ────────────────────────────────────────────────────────────
    csv_data = df_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download results as CSV",
        data=csv_data,
        file_name="cineassist_evaluation.csv",
        mime="text/csv",
    )

else:
    st.info(
        "Click **Run all scenarios** to evaluate the recommender against the predefined test set."
    )

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

with st.expander("Metric definitions"):
    st.markdown("""
| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Precision@K** | `|relevant ∩ predicted| / K` | Of K recommendations, how many were relevant? |
| **Recall@K** | `|relevant ∩ predicted| / |relevant|` | Of all relevant movies, how many were returned? |
| **F1** | `2 × P × R / (P + R)` | Harmonic mean — balanced view of P and R |
| **MRR** | `1 / rank_of_first_hit` | How early does the first relevant result appear? |
| **Accuracy** | `(TP + TN) / total` | Overall correct classifications across dataset |
    """)
