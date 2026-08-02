"""
Translation-quality metrics for CineAssist — BLEU + human evaluation.

WHAT THIS DOES
--------------
CineAssist translates user messages to English (for NLP) and its responses back
to the user's language, using Helsinki-NLP MarianMT (see
``src/translation/translator.py``). This module measures HOW GOOD those
translations are, in two complementary ways:

1. BLEU (automatic).  ``sacreBLEU`` corpus BLEU comparing machine translations
   against curated human references (``translation_eval_data.py``). BLEU is a
   0–100 score; higher is better. It is the standard automatic MT metric and is
   the same metric ``fine_tune.py`` optimises during training.

2. Human evaluation (manual).  A lightweight CSV harness where human raters
   score each translation for adequacy (does it preserve the meaning?) and
   fluency (does it read naturally?), each on a 1–5 scale. BLEU rewards n-gram
   overlap; human scores capture meaning/naturalness that BLEU misses. Reporting
   both is the accepted practice for MT evaluation.

PRE- vs POST-FINE-TUNING
------------------------
The professor's request is to compare translation quality BEFORE and AFTER
fine-tuning. This module is structured for exactly that:

  * ``evaluate_translation_quality()`` scores the CURRENT (base / pre-fine-tune)
    MarianMT model using the real ``translate_*`` functions. Run it now to get
    your baseline.

  * After fine-tuning (``src/translation/fine_tune.py``), build a POST
    ``translate_fn`` and pass it back in. Because ``translator.py._load_model``
    automatically prefers a fine-tuned model saved under
    ``models/translation/<src>-<tgt>/``, simply re-running
    ``evaluate_translation_quality()`` after fine-tuning already scores the
    fine-tuned model — no code change needed. To score a fine-tuned model at a
    non-standard path without touching the global cache, pass a custom
    ``translate_fn`` (see ``make_translate_fn`` docstring below).

  * ``compare_pre_post(pre, post)`` diffs the two result dicts and returns a
    per-pair BLEU delta plus an improved/regressed verdict.

This module NEVER runs fine-tuning itself (that is heavy / GPU work). It only
measures. See the "HOW TO PRODUCE A POST-FINE-TUNE translate_fn" section in
``make_translate_fn`` for the exact steps.

CLI
---
    python -m src.metrics.translation_quality

Prints base-model BLEU per language pair + aggregate, then writes a sample
human-eval template CSV (pre-filled with base-model output) to an output path it
prints. Degrades gracefully with a clear message if torch / transformers /
langdetect / sacrebleu are unavailable.

PUBLIC API
----------
    compute_bleu(translate_fn, pairs)              -> dict
    evaluate_translation_quality(directions=None, translate_fn=None) -> dict
    compare_pre_post(pre_result, post_result)      -> dict
    make_translate_fn(...)                          -> callable
    export_human_eval_template(csv_path, ...)      -> str
    aggregate_human_scores(csv_path)               -> dict
    HUMAN_EVAL_FIELDS                              (schema)
"""

import csv
import importlib
import importlib.util
import itertools
import statistics
import tempfile
from pathlib import Path

from src.metrics.translation_eval_data import (
    all_directions,
    eval_set_sizes,
    get_pairs,
)

# ---------------------------------------------------------------------------
# Optional heavy dependencies — imported lazily / defensively so the module can
# still be imported (and the human-eval aggregator still runs) when torch /
# transformers / sacrebleu are missing.
# ---------------------------------------------------------------------------
try:
    import sacrebleu as _sacrebleu

    _HAS_SACREBLEU = True
except ImportError:  # pragma: no cover - environment dependent
    _HAS_SACREBLEU = False


def _split_direction(direction: str):
    """'es-en' -> ('es', 'en')."""
    src, tgt = direction.split("-", 1)
    return src, tgt


# ---------------------------------------------------------------------------
# translate_fn factory
# ---------------------------------------------------------------------------
def make_translate_fn(model_dir: str = None):
    """
    Build a ``translate_fn(text, src_lang, tgt_lang) -> str`` for BLEU scoring.

    With no argument, returns the project's REAL translation function
    (``src.translation.translator.translate``), which resolves to the base
    MarianMT model unless a fine-tuned model already exists at the standard
    ``models/translation/<src>-<tgt>/`` path.

    HOW TO PRODUCE A POST-FINE-TUNE translate_fn
    --------------------------------------------
    Option A (standard path — zero code):
        1. Fine-tune, which saves to models/translation/<src>-<tgt>/:
               python src/translation/fine_tune.py --src es --tgt en
        2. Re-run evaluate_translation_quality(); translator._load_model()
           automatically loads the fine-tuned model from that directory.

    Option B (custom / experimental path — this factory):
        A fine-tuned model saved somewhere else can be scored without disturbing
        the global translator cache by loading it directly:

            from transformers import MarianMTModel, MarianTokenizer
            import torch

            def make_finetuned_fn(path):
                tok = MarianTokenizer.from_pretrained(path)
                mdl = MarianMTModel.from_pretrained(path); mdl.eval()
                def fn(text, src, tgt):
                    if src == tgt or not text.strip():
                        return text
                    enc = tok([text], return_tensors="pt", truncation=True,
                              max_length=512)
                    with torch.no_grad():
                        out = mdl.generate(**enc, num_beams=4, max_length=512)
                    return tok.decode(out[0], skip_special_tokens=True)
                return fn

            post = evaluate_translation_quality(
                translate_fn=make_finetuned_fn("path/to/ft-es-en"))

        (That per-pair loader only fits single-direction models; the default
        translate handles every direction. This module keeps fine-tuning out of
        scope — it only measures.)

    Args:
        model_dir: Unused hook kept for API symmetry; pass a custom translate_fn
            to evaluate_translation_quality() for non-standard model paths.

    Returns:
        The real translate callable from the translation module.
    """
    del model_dir  # compatibility hook retained for callers using the keyword
    # Lazy by design: importing this module must not require the heavy ML stack.
    translator = importlib.import_module("src.translation.translator")
    return translator.translate


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------
def compute_bleu(translate_fn, pairs: list) -> dict:
    """
    Score a translate function against reference translations with sacreBLEU.

    Runs ``translate_fn(src, src_lang, tgt_lang)`` over every pair, then computes
    corpus BLEU of the hypotheses vs the single references using the ``sacrebleu``
    library directly (the reproducible, tokenisation-standardised BLEU used in MT
    research and by ``fine_tune.py``).

    Args:
        translate_fn: Callable ``(text, src_lang, tgt_lang) -> str``. The pairs
            must therefore carry their direction; pass pairs from a single
            direction and set ``src_lang``/``tgt_lang`` via the ``pairs`` items,
            OR use ``evaluate_translation_quality`` which handles directions.
            Each pair is a dict with keys: "src", "ref", "src_lang", "tgt_lang".
        pairs: List of dicts. Required keys per item: "src", "ref". Optional:
            "src_lang", "tgt_lang" (needed when translate_fn needs a direction).

    Returns:
        dict with:
          "bleu"     : float corpus BLEU (0–100), rounded to 2 dp
          "n"        : number of sentences scored
          "details"  : list of per-sentence dicts
                       {"src", "ref", "hyp", "sentence_bleu"}
    """
    if not _HAS_SACREBLEU:
        raise RuntimeError(
            "sacrebleu is not installed — cannot compute BLEU. "
            "Install with: pip install sacrebleu"
        )
    hyps, refs, details = [], [], []
    for p in pairs:
        src_lang = p.get("src_lang")
        tgt_lang = p.get("tgt_lang")
        if src_lang and tgt_lang:
            hyp = translate_fn(p["src"], src_lang, tgt_lang)
        else:
            # translate_fn already knows its direction (single-arg style)
            hyp = translate_fn(p["src"])
        hyp = (hyp or "").strip()
        hyps.append(hyp)
        refs.append(p["ref"])
        s_bleu = _sacrebleu.sentence_bleu(hyp, [p["ref"]]).score
        details.append(
            {
                "src": p["src"],
                "ref": p["ref"],
                "hyp": hyp,
                "sentence_bleu": round(s_bleu, 2),
            }
        )

    # sacreBLEU corpus BLEU expects references as a list-of-lists (one list per
    # reference set); we have a single reference per sentence.
    corpus = _sacrebleu.corpus_bleu(hyps, [refs])
    return {
        "bleu": round(corpus.score, 2),
        "n": len(hyps),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Pre/post evaluation API
# ---------------------------------------------------------------------------
def evaluate_translation_quality(directions=None, translate_fn=None) -> dict:
    """
    Compute BLEU per language-pair direction + an aggregate.

    By default scores the CURRENT (base / pre-fine-tune) MarianMT model using the
    real ``translate`` function. Re-run after fine-tuning to score the fine-tuned
    model (see ``make_translate_fn``), or pass a custom ``translate_fn`` to score
    an experimental fine-tuned model directly.

    Args:
        directions: Optional list of direction keys ("es-en", ...). Defaults to
            every direction in the eval set.
        translate_fn: Optional ``(text, src, tgt) -> str`` callable. Defaults to
            the project's real translate function.

    Returns:
        dict with:
          "per_pair"  : {direction: {"bleu", "n", "details"}}
          "aggregate" : {"bleu": micro-averaged corpus BLEU over all sentences,
                         "n": total sentences}
          "sizes"     : {direction: pair_count}
    """
    if translate_fn is None:
        translate_fn = make_translate_fn()
    if directions is None:
        directions = all_directions()

    per_pair = {}
    all_hyps, all_refs = [], []

    for direction in directions:
        src_lang, tgt_lang = _split_direction(direction)
        pairs = [
            {**p, "src_lang": src_lang, "tgt_lang": tgt_lang}
            for p in get_pairs(direction)
        ]
        res = compute_bleu(translate_fn, pairs)
        per_pair[direction] = res
        for d in res["details"]:
            all_hyps.append(d["hyp"])
            all_refs.append(d["ref"])

    aggregate_score = (
        round(_sacrebleu.corpus_bleu(all_hyps, [all_refs]).score, 2)
        if all_hyps
        else 0.0
    )
    return {
        "per_pair": per_pair,
        "aggregate": {"bleu": aggregate_score, "n": len(all_hyps)},
        "sizes": {direction: result["n"] for direction, result in per_pair.items()},
    }


def compare_pre_post(pre_result: dict, post_result: dict) -> dict:
    """
    Diff two ``evaluate_translation_quality`` results (pre vs post fine-tuning).

    Args:
        pre_result:  result dict from the BASE model.
        post_result: result dict from the FINE-TUNED model.

    Returns:
        dict with:
          "per_pair"  : {direction: {"pre", "post", "delta", "verdict"}}
          "aggregate" : {"pre", "post", "delta", "verdict"}
        where verdict is "improved" | "regressed" | "unchanged".
    """

    def _verdict(delta: float) -> str:
        if delta > 0.01:
            return "improved"
        if delta < -0.01:
            return "regressed"
        return "unchanged"

    per_pair = {}
    directions = set(pre_result["per_pair"]) | set(post_result["per_pair"])
    for d in sorted(directions):
        pre = pre_result["per_pair"].get(d, {}).get("bleu", 0.0)
        post = post_result["per_pair"].get(d, {}).get("bleu", 0.0)
        delta = round(post - pre, 2)
        per_pair[d] = {
            "pre": pre,
            "post": post,
            "delta": delta,
            "verdict": _verdict(delta),
        }

    pre_agg = pre_result["aggregate"]["bleu"]
    post_agg = post_result["aggregate"]["bleu"]
    agg_delta = round(post_agg - pre_agg, 2)
    return {
        "per_pair": per_pair,
        "aggregate": {
            "pre": pre_agg,
            "post": post_agg,
            "delta": agg_delta,
            "verdict": _verdict(agg_delta),
        },
    }


# ---------------------------------------------------------------------------
# Human-evaluation scaffold
# ---------------------------------------------------------------------------
# CSV schema. Humans fill in the two *_1to5 columns and the rater column.
HUMAN_EVAL_FIELDS = [
    "id",  # stable row id, e.g. "es-en-03"
    "direction",  # "es-en"
    "source",  # source sentence
    "reference",  # human reference (for the rater's context)
    "mt_output",  # machine translation to be judged (pre-filled)
    "human_adequacy_1to5",  # 1 (meaning lost) .. 5 (meaning fully preserved)
    "human_fluency_1to5",  # 1 (unnatural) .. 5 (perfectly natural)
    "rater",  # rater id/name, e.g. "rater_a"
]


def export_human_eval_template(
    csv_path: str, directions=None, translate_fn=None, raters=None
) -> str:
    """
    Write a human-eval template CSV pre-filled with base-model MT output.

    Humans only need to fill ``human_adequacy_1to5``, ``human_fluency_1to5`` and
    ``rater``; everything else (source, reference, mt_output) is prepared here.

    Args:
        csv_path:    Output path for the template CSV.
        directions:  Direction keys to include (default: all).
        translate_fn: Callable producing the mt_output (default: real translate).
        raters:      Optional list of rater ids. If given, one row PER rater per
            sentence is emitted (blank scores) so multiple people can rate the
            same items — enabling inter-rater agreement. If None, a single blank
            ``rater`` column is emitted.

    Returns:
        The csv_path written.
    """
    if translate_fn is None:
        translate_fn = make_translate_fn()
    if directions is None:
        directions = all_directions()

    rows = []
    for direction in directions:
        src_lang, tgt_lang = _split_direction(direction)
        for i, p in enumerate(get_pairs(direction), start=1):
            mt = translate_fn(p["src"], src_lang, tgt_lang)
            base = {
                "id": f"{direction}-{i:02d}",
                "direction": direction,
                "source": p["src"],
                "reference": p["ref"],
                "mt_output": (mt or "").strip(),
            }
            if raters:
                rows.extend(
                    [
                        {
                            **base,
                            "human_adequacy_1to5": "",
                            "human_fluency_1to5": "",
                            "rater": rater,
                        }
                        for rater in raters
                    ]
                )
            else:
                rows.append(
                    {
                        **base,
                        "human_adequacy_1to5": "",
                        "human_fluency_1to5": "",
                        "rater": "",
                    }
                )

    output_path = Path(csv_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HUMAN_EVAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _write_sample_human_eval(csv_path: str) -> str:
    """
    Write a tiny FILLED sample so ``aggregate_human_scores`` is demonstrably
    runnable without any real human data. Two raters score three items.
    """
    sample = [
        {
            "id": "es-en-01",
            "direction": "es-en",
            "source": "Quiero una comedia romántica de los 90.",
            "reference": "I want a romantic comedy from the 90s.",
            "mt_output": "I want a romantic comedy from the 90s.",
            "human_adequacy_1to5": 5,
            "human_fluency_1to5": 5,
            "rater": "rater_a",
        },
        {
            "id": "es-en-01",
            "direction": "es-en",
            "source": "Quiero una comedia romántica de los 90.",
            "reference": "I want a romantic comedy from the 90s.",
            "mt_output": "I want a romantic comedy from the 90s.",
            "human_adequacy_1to5": 5,
            "human_fluency_1to5": 4,
            "rater": "rater_b",
        },
        {
            "id": "en-es-01",
            "direction": "en-es",
            "source": "I want a romantic comedy from the 90s.",
            "reference": "Quiero una comedia romántica de los 90.",
            "mt_output": "Quiero una comedia romántica de los 90.",
            "human_adequacy_1to5": 5,
            "human_fluency_1to5": 5,
            "rater": "rater_a",
        },
        {
            "id": "en-es-01",
            "direction": "en-es",
            "source": "I want a romantic comedy from the 90s.",
            "reference": "Quiero una comedia romántica de los 90.",
            "mt_output": "Quiero una comedia romántica de los 90.",
            "human_adequacy_1to5": 4,
            "human_fluency_1to5": 5,
            "rater": "rater_b",
        },
        {
            "id": "fr-en-03",
            "direction": "fr-en",
            "source": "Je cherche un film d'horreur qui fait vraiment peur.",
            "reference": "I'm looking for a horror movie that is really scary.",
            "mt_output": "I'm looking for a horror film that's really scary.",
            "human_adequacy_1to5": 5,
            "human_fluency_1to5": 4,
            "rater": "rater_a",
        },
        {
            "id": "fr-en-03",
            "direction": "fr-en",
            "source": "Je cherche un film d'horreur qui fait vraiment peur.",
            "reference": "I'm looking for a horror movie that is really scary.",
            "mt_output": "I'm looking for a horror film that's really scary.",
            "human_adequacy_1to5": 4,
            "human_fluency_1to5": 4,
            "rater": "rater_b",
        },
    ]
    output_path = Path(csv_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HUMAN_EVAL_FIELDS)
        writer.writeheader()
        writer.writerows(sample)
    return csv_path


def _read_human_scores(csv_path: str) -> list[dict]:
    """Read complete human-evaluation rows and coerce their numeric scores."""
    rows = []
    with Path(csv_path).open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            try:
                adequacy = float(row.get("human_adequacy_1to5", ""))
                fluency = float(row.get("human_fluency_1to5", ""))
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "id": row.get("id", ""),
                    "direction": row.get("direction", ""),
                    "rater": row.get("rater", ""),
                    "adequacy": adequacy,
                    "fluency": fluency,
                }
            )
    return rows


def _scores_by_direction(rows: list[dict]) -> dict:
    """Aggregate mean scores for each translation direction."""
    result = {}
    for direction in sorted({row["direction"] for row in rows}):
        subset = [row for row in rows if row["direction"] == direction]
        result[direction] = {
            "adequacy": round(statistics.mean(row["adequacy"] for row in subset), 3),
            "fluency": round(statistics.mean(row["fluency"] for row in subset), 3),
            "n": len(subset),
        }
    return result


def _inter_rater_summary(rows: list[dict], raters: list[str]) -> dict | None:
    """Calculate mean absolute differences for overlapping rater pairs."""
    if len(raters) < 2:
        return None
    by_item = {}
    for row in rows:
        by_item.setdefault(row["id"], {})[row["rater"]] = row
    adequacy_diffs, fluency_diffs = [], []
    for per_rater in by_item.values():
        present = [rater for rater in raters if rater in per_rater]
        for first, second in itertools.combinations(present, 2):
            score_a, score_b = per_rater[first], per_rater[second]
            adequacy_diffs.append(abs(score_a["adequacy"] - score_b["adequacy"]))
            fluency_diffs.append(abs(score_a["fluency"] - score_b["fluency"]))
    if not adequacy_diffs:
        return None
    return {
        "adequacy_mean_abs_diff": round(statistics.mean(adequacy_diffs), 3),
        "fluency_mean_abs_diff": round(statistics.mean(fluency_diffs), 3),
        "pairs_compared": len(adequacy_diffs),
    }


def aggregate_human_scores(csv_path: str) -> dict:
    """
    Aggregate a filled human-eval CSV into mean adequacy/fluency scores.

    Reads rows written in the ``HUMAN_EVAL_FIELDS`` schema (only rows with both
    score columns filled are counted) and returns overall and per-direction
    means, plus a simple inter-rater agreement when 2+ raters scored shared items.

    Inter-rater agreement is reported as the mean absolute difference between
    raters on items they both scored (0 = perfect agreement; larger = more
    disagreement, on the 1–5 scale) for adequacy and fluency separately. This is
    a lightweight, dependency-free proxy — not a chance-corrected kappa.

    Args:
        csv_path: Path to a filled template CSV.

    Returns:
        dict with:
          "n_ratings"      : number of scored rows
          "raters"         : sorted list of rater ids seen
          "overall"        : {"adequacy", "fluency"} mean floats
          "per_direction"  : {direction: {"adequacy", "fluency", "n"}}
          "inter_rater"    : {"adequacy_mean_abs_diff", "fluency_mean_abs_diff",
                              "pairs_compared"} or None if <2 raters overlap
    """

    rows = _read_human_scores(csv_path)

    if not rows:
        return {
            "n_ratings": 0,
            "raters": [],
            "overall": None,
            "per_direction": {},
            "inter_rater": None,
        }

    overall = {
        "adequacy": round(statistics.mean(r["adequacy"] for r in rows), 3),
        "fluency": round(statistics.mean(r["fluency"] for r in rows), 3),
    }

    raters = sorted({r["rater"] for r in rows if r["rater"]})
    return {
        "n_ratings": len(rows),
        "raters": raters,
        "overall": overall,
        "per_direction": _scores_by_direction(rows),
        "inter_rater": _inter_rater_summary(rows, raters),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _check_runtime() -> str:
    """Return an error string if a required dep is missing, else ''."""
    missing = []
    if not _HAS_SACREBLEU:
        missing.append("sacrebleu")
    if not all(importlib.util.find_spec(name) for name in ("torch", "transformers")):
        missing.append("torch/transformers")
    if importlib.util.find_spec("langdetect") is None:
        missing.append("langdetect")
    if missing:
        return (
            "Cannot run base-model BLEU: missing dependencies -> "
            + ", ".join(missing)
            + ".\nInstall them (e.g. pip install sacrebleu transformers torch "
            "langdetect) and re-run."
        )
    return ""


def main():
    print("=" * 64)
    print("  CineAssist — Translation Quality (BLEU + human-eval scaffold)")
    print("=" * 64)

    sizes = eval_set_sizes()
    print("\nEval set sizes (sentence pairs per direction):")
    for d, n in sizes.items():
        print(f"  {d}: {n}")
    print(f"  total: {sum(sizes.values())}")

    err = _check_runtime()
    if err:
        print("\n" + err)
        print(
            "\nThe human-eval aggregator does not need those deps; writing a "
            "sample and aggregating it below.\n"
        )
    else:
        print("\nScoring BASE (pre-fine-tune) MarianMT model with sacreBLEU...")
        print("(first call per language pair loads the model; cached after)\n")
        result = evaluate_translation_quality()

        header = f"{'Direction':<10} {'Pairs':>6} {'BLEU':>8}"
        print(header)
        print("-" * len(header))
        for d, res in result["per_pair"].items():
            print(f"{d:<10} {res['n']:>6} {res['bleu']:>8.2f}")
        print("-" * len(header))
        agg = result["aggregate"]
        print(f"{'AGGREGATE':<10} {agg['n']:>6} {agg['bleu']:>8.2f}")

        print(
            "\nPre/post fine-tuning: re-run after fine-tuning "
            "(src/translation/fine_tune.py) and call "
            "compare_pre_post(pre_result, post_result)."
        )

    # Human-eval scaffold demo — always runnable.
    out_dir = Path(tempfile.mkdtemp(prefix="cineassist_human_eval_"))
    sample_path = out_dir / "human_eval_sample_filled.csv"
    _write_sample_human_eval(sample_path)
    agg_human = aggregate_human_scores(sample_path)

    print("\n" + "=" * 64)
    print("  Human-evaluation scaffold")
    print("=" * 64)
    print(f"Schema columns: {', '.join(HUMAN_EVAL_FIELDS)}")
    print(f"\nSample FILLED human-eval CSV written to:\n  {sample_path}")
    print("\nAggregated sample human scores:")
    print(f"  n_ratings : {agg_human['n_ratings']}")
    print(f"  raters    : {agg_human['raters']}")
    print(f"  overall   : {agg_human['overall']}")
    print(f"  per_dir   : {agg_human['per_direction']}")
    print(f"  agreement : {agg_human['inter_rater']}")

    # Also export a BLANK template pre-filled with base-model output IF deps ok.
    if not err:
        blank_path = out_dir / "human_eval_template_blank.csv"
        try:
            export_human_eval_template(blank_path)
            print(
                f"\nBlank template (base-model mt_output pre-filled, scores "
                f"empty for raters to fill) written to:\n  {blank_path}"
            )
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
            print(f"\n[warn] could not write blank template: {exc}")

    print("\nDone.")


if __name__ == "__main__":
    main()
