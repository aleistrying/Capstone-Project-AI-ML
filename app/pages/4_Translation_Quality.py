"""
CineAssist — Translation Quality (BLEU + human evaluation)

Surfaces the professor-requested translation metrics from
src/metrics/translation_quality.py:
  - sacreBLEU on a curated in-domain parallel set (the PRE-fine-tune baseline)
  - a documented PRE vs POST fine-tune comparison
  - a human-evaluation template + aggregation scaffold
"""

import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.translation_quality import (
    evaluate_translation_quality,
    export_human_eval_template,
    _check_runtime,
)

st.set_page_config(
    page_title="CineAssist — Translation BLEU", page_icon="🌐", layout="wide"
)
st.title("🌐 Translation Quality — BLEU")
st.caption(
    "sacreBLEU on a curated movie-request parallel set (MarianMT, Helsinki-NLP). "
    "This is the PRE-fine-tune baseline; re-run after fine-tuning to compare."
)

runtime_msg = _check_runtime()
if runtime_msg:
    st.warning(runtime_msg)

if not st.button("▶ Run BLEU evaluation (base model)", type="primary"):
    st.info(
        "Click to translate the eval set with the current MarianMT models and score sacreBLEU."
    )
    st.stop()

with st.spinner("Translating eval set and scoring BLEU…"):
    result = evaluate_translation_quality()

per_pair = result.get("per_pair", {})
agg = result.get("aggregate", {})

st.subheader("Base-model sacreBLEU")
cols = st.columns(len(per_pair) + 1 if per_pair else 1)
for i, (direction, r) in enumerate(per_pair.items()):
    bleu = r.get("bleu", 0.0) if isinstance(r, dict) else r
    cols[i].metric(direction, f"{bleu:.2f}")
if agg:
    agg_bleu = agg.get("bleu", agg) if isinstance(agg, dict) else agg
    cols[-1].metric("Aggregate", f"{agg_bleu:.2f}")

st.divider()
st.subheader("Pre vs post fine-tune")
st.markdown(
    "1. Fine-tune a pair: `python src/translation/fine_tune.py --src es --tgt en` "
    "(saved to `models/translation/es-en/`, auto-preferred by the translator).\n"
    "2. Re-run this page (or `evaluate_translation_quality()`) to get the POST result.\n"
    "3. `compare_pre_post(pre, post)` reports per-pair BLEU deltas + verdicts."
)

st.divider()
st.subheader("Human evaluation")
st.markdown(
    "Human raters score **adequacy** and **fluency** (1–5). Export a template "
    "pre-filled with the base model's output, collect scores, then aggregate."
)
if st.button("Export human-eval template CSV"):
    out = Path(tempfile.gettempdir()) / "cineassist_human_eval_template.csv"
    path = export_human_eval_template(str(out))
    st.success(f"Template written to `{path}`")
    try:
        st.code(Path(path).read_text()[:2000])
    except Exception:
        pass

with st.expander("Raw BLEU result (full dict)"):
    st.json(result)
