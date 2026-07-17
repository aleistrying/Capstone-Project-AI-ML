# Metrics Module - Architecture & Technical Documentation

## Overview

The Metrics module provides a comprehensive evaluation framework for the CineAssist chatbot recommendation system. It measures recommendation quality using precision, recall, accuracy, F1 score, and Mean Reciprocal Rank (MRR) metrics.

---

## Folder Structure

```
src/metrics/
├── __init__.py                    # Package initialization & exports
├── metrics.py                     # Core metric calculation functions
├── test_data.py                   # Predefined test scenarios & ground truth
├── evaluator.py                   # Main Evaluator class orchestrator (ID-based ground-truth eval)
├── benchmark.py                   # Live-pipeline genre-overlap benchmark (David) — CLI: python -m src.metrics.benchmark
├── setup_and_run.py               # Helper script for exploration & evaluation
├── README.md                      # User-facing documentation
├── ARCHITECTURE.md                # This file (technical docs)
├── METRICS_FLOWCHART.pdf          # Visual architecture diagram
└── METRICS_FLOWCHART.png          # PNG preview of flowchart
```

---

## Module Descriptions

### 1. `metrics.py` - Core Metric Functions

**Purpose:** Implements all metric calculation algorithms.

**Functions:**

```python
def calculate_precision(predicted_ids, relevant_ids) -> float
```
- **What:** (True Positives) / (True Positives + False Positives)
- **Meaning:** Of the movies we recommended, what % were actually relevant?
- **Range:** 0.0 to 1.0 (higher is better)
- **Use Case:** Quality control - ensure recommendations are correct

```python
def calculate_recall(predicted_ids, relevant_ids) -> float
```
- **What:** (True Positives) / (True Positives + False Negatives)
- **Meaning:** Of all relevant movies, what % did we return?
- **Range:** 0.0 to 1.0 (higher is better)
- **Use Case:** Coverage - ensure we find all good movies

```python
def calculate_accuracy(predicted_ids, relevant_ids, total_items) -> float
```
- **What:** (True Positives + True Negatives) / Total Items
- **Meaning:** Overall % of correct classifications
- **Range:** 0.0 to 1.0 (higher is better)
- **Note:** Can be misleading with imbalanced datasets

```python
def calculate_f1_score(predicted_ids, relevant_ids) -> float
```
- **What:** 2 × (Precision × Recall) / (Precision + Recall)
- **Meaning:** Harmonic mean of precision and recall
- **Range:** 0.0 to 1.0 (higher is better)
- **Use Case:** When you need balanced precision AND recall

```python
def calculate_mean_reciprocal_rank(predicted_ids, relevant_ids) -> float
```
- **What:** 1 / (rank of first relevant item)
- **Meaning:** How early does the first relevant movie appear?
- **Range:** 0.0 to 1.0 (higher is better)
- **Use Case:** Ranking quality - reward putting relevant items first

---

### 2. `test_data.py` - Test Scenarios & Ground Truth

**Purpose:** Stores predefined test cases with expected movie recommendations.

**Structure:**
```python
TEST_SCENARIOS = [
    {
        "id": "test_01_action_movie",
        "user_input": "I want an action movie with lots of fights...",
        "relevant_movie_ids": [79132, 58559, 72998, 89745, 122904],
        "description": "Action movie with fighting and explosions"
    },
    # ... 10 scenarios total
]
```

**Key Functions:**
- `get_test_scenarios()` - Returns all test scenarios
- `get_test_scenario(test_id)` - Get specific scenario by ID
- `print_test_scenarios()` - Print all scenarios in readable format

**10 Predefined Scenarios:**
1. Action movies
2. Comedy movies
3. Drama movies
4. Sci-Fi movies
5. Horror movies
6. Romance movies
7. Animation movies
8. Year-based filtering (2000s)
9. Rating-based filtering (>8.0)
10. Language-based filtering (Spanish)

---

### 3. `evaluator.py` - Main Orchestrator

**Purpose:** Coordinates the evaluation process and aggregates results.

**Class: `Evaluator`**

**Constructor:**
```python
def __init__(self, recommender_func, movies_df, vectorizer, tfidf_matrix=None)
```
- `recommender_func`: Function that generates recommendations
- `movies_df`: DataFrame with all movies
- `vectorizer`: TF-IDF vectorizer for text processing
- `tfidf_matrix`: Pre-computed TF-IDF matrix (optional)

**Key Methods:**

```python
def evaluate_single_query(self, user_input, relevant_ids, top_n=5) -> dict
```
- Evaluates one user query
- Returns dict with metrics for that query
- Called for each test scenario

```python
def evaluate_all_scenarios(self, top_n=5) -> list
```
- Evaluates all 10 test scenarios
- Returns list of result dicts
- Each dict contains: predicted_ids, relevant_ids, metrics

```python
def get_average_metrics(self, results=None) -> dict
```
- Calculates average metrics across all tests
- Returns: avg_precision, avg_recall, avg_accuracy, avg_f1, avg_mrr, total_tests

```python
def print_results(self, results=None)
```
- Pretty-prints results to console
- Shows metrics for each scenario
- Shows average metrics

```python
def export_results_csv(self, filename, results=None)
```
- Exports results to CSV file
- One row per test scenario
- Columns: scenario_id, user_input, metrics, etc.

---

### 4. `setup_and_run.py` - Helper Script

**Purpose:** Command-line interface for exploring data and running evaluation.

**Three Actions:**

#### Action 1: `--action explore`
```bash
python setup_and_run.py --action explore
```
**What it does:**
1. Loads movie dataset (CSV or Databricks)
2. Analyzes available movies
3. Shows:
   - Sample movies from dataset
   - Genre distribution
   - MovieIds for each genre
   - High-rated movies (>8.0)
   - Movies from 2000s
   - Movies by language

**Output:** MovieIds grouped by category (used to populate test_data.py)

**Functions:**
- `load_data_local()` - Try loading from data/movies_final.csv
- `load_data_databricks()` - Try loading from Databricks workspace
- `load_movies_data()` - Load from available source
- `explore_movies()` - Analyze and display movie data

#### Action 2: `--action run`
```bash
python setup_and_run.py --action run
```
**What it does:**
1. Loads movie data
2. Checks that test_data.py is populated (no empty scenarios)
3. Loads ML models (vectorizer & matrix)
4. Runs evaluation on all test scenarios
5. Prints results to console
6. Exports results to CSV

**Functions:**
- `load_models()` - Load TF-IDF models from models/ folder
- `run_evaluation()` - Execute evaluation pipeline

#### Action 3: `--action setup` (default)
```bash
python setup_and_run.py --action setup
```
**What it does:**
- Prints step-by-step setup guide
- Shows how to use explore and run actions

---

### 5. Programmatic usage

**Usage:**
```python
from src.metrics import Evaluator
from src.recommender.recommender_engine import recommend_on_the_fly

evaluator = Evaluator(recommend_on_the_fly, movies_df, vectorizer, tfidf_matrix)
results = evaluator.evaluate_all_scenarios(top_n=5)
evaluator.print_results(results)
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │  setup_and_run.py   │
                    │  --action explore   │
                    └─────────────────────┘
                              ↓
                ┌─────────────────────────────┐
                │  Load Movie Data            │
                │  (CSV or Databricks)        │
                └─────────────────────────────┘
                              ↓
                ┌─────────────────────────────┐
                │  Analyze & Display:         │
                │  - Genres                   │
                │  - MovieIds                 │
                │  - Ratings                  │
                │  - Languages                │
                │  - Years                    │
                └─────────────────────────────┘
                              ↓
         ┌────────────────────────────────────────┐
         │ USER: Copy MovieIds to test_data.py    │
         └────────────────────────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │  setup_and_run.py   │
                    │   --action run      │
                    └─────────────────────┘
                              ↓
        ┌───────────────────────────────────────────┐
        │  Load:                                    │
        │  - Movie Dataset                          │
        │  - Test Scenarios (test_data.py)         │
        │  - ML Models (vectorizer, tfidf_matrix)  │
        └───────────────────────────────────────────┘
                              ↓
        ┌───────────────────────────────────────────┐
        │  Evaluator.evaluate_all_scenarios()       │
        └───────────────────────────────────────────┘
                              ↓
        ┌──────────────────────────────────────────────────────┐
        │  For Each Test Scenario:                             │
        │  1. Get user_input & relevant_movie_ids              │
        │  2. Call recommender_engine.recommend_on_the_fly()  │
        │  3. Get predicted movie IDs                          │
        │  4. Calculate metrics (precision, recall, etc.)      │
        │  5. Store result                                     │
        └──────────────────────────────────────────────────────┘
                              ↓
        ┌───────────────────────────────────────────┐
        │  Generate Output:                         │
        │  - Console: Formatted results             │
        │  - CSV: evaluation_results.csv            │
        │  - Summary: Average metrics               │
        └───────────────────────────────────────────┘
```

---

## Evaluation Workflow

### Step 1: Initialization
```
Evaluator.__init__()
  ├─ Store: recommender_func
  ├─ Store: movies_df (86,906 movies)
  ├─ Store: vectorizer (TF-IDF transformer)
  ├─ Store: tfidf_matrix (pre-computed)
  └─ Calculate: total_movies count
```

### Step 2: Query Evaluation
```
evaluate_single_query(user_input, relevant_ids, top_n=5)
  ├─ Call: recommender_func(query_text, movies_df, vectorizer, tfidf_matrix)
  │   └─ TF-IDF similarity matching
  │   └─ Return: top 5 recommended movies
  │
  ├─ Extract: predicted_movie_ids (movieId column)
  │
  ├─ Calculate Metrics:
  │   ├─ precision = |predicted ∩ relevant| / |predicted|
  │   ├─ recall = |predicted ∩ relevant| / |relevant|
  │   ├─ accuracy = (TP + TN) / total_movies
  │   ├─ f1_score = 2 × (precision × recall) / (precision + recall)
  │   └─ mrr = 1 / rank_of_first_relevant
  │
  └─ Return: dict with all metrics
```

### Step 3: Batch Processing
```
evaluate_all_scenarios(top_n=5)
  ├─ For test_01 to test_10:
  │   └─ Call: evaluate_single_query()
  │
  └─ Return: list of 10 result dicts
```

### Step 4: Aggregation
```
get_average_metrics(results)
  ├─ avg_precision = mean(all precisions)
  ├─ avg_recall = mean(all recalls)
  ├─ avg_accuracy = mean(all accuracies)
  ├─ avg_f1_score = mean(all f1_scores)
  ├─ avg_mrr = mean(all mrrs)
  └─ Return: summary dict
```

### Step 5: Output
```
Three output formats:
  1. Console: print_results() - formatted text display
  2. CSV: export_results_csv() - spreadsheet for analysis
  3. Summary: get_average_metrics() - aggregate statistics
```

---

## Integration with Existing Code

### Connection to `recommender_engine.py`
```python
# From: src/recommender/recommender_engine.py
def recommend_on_the_fly(query_text, movies_df, vectorizer, tfidf_matrix, state_dict=None, top_n=5):
    """
    Input: User query + movie data + ML models
    Process: TF-IDF similarity matching
    Output: Top N recommended movies (DataFrame)
    """
```

**How it's used:**
- Evaluator calls this function 10 times (once per test scenario)
- Returns recommendations DataFrame with columns:
  - title, genres_list, vote_average, similarity_score, overview
  - optionally: release_year

### Connection to `test_data.py`
```python
# Test scenarios contain:
#   - user_input: Natural language query
#   - relevant_movie_ids: Ground truth (expected correct answers)
#   - description: What this test measures
```

**How it's used:**
- Evaluator loads all 10 scenarios
- Passes user_input to recommender
- Compares predicted IDs against relevant_ids
- Calculates metrics from the comparison

---

## Metric Interpretation Guide

### Precision (Quality)
```
High Precision (>0.7):  Recommendations are usually correct
Low Precision (<0.4):   Many wrong recommendations
Trade-off:              Conservative but high-quality results
```

### Recall (Coverage)
```
High Recall (>0.7):     We find most relevant movies
Low Recall (<0.4):      We miss many relevant movies
Trade-off:              Comprehensive but may include errors
```

### F1 Score (Balance)
```
F1 = 2 × (P × R) / (P + R)

F1 > 0.65:  Good balance between precision & recall
F1 < 0.55:  Poor overall performance
```

### Accuracy
```
Be careful with accuracy!
- Works well: Balanced relevant/non-relevant split
- Misleading: Highly imbalanced (90% non-relevant)
```

### MRR (Ranking Quality)
```
MRR = 1.0:   First result is relevant (perfect!)
MRR = 0.5:   First relevant at rank 2
MRR = 0.33:  First relevant at rank 3
MRR = 0.0:   No relevant results found
```

---

## Performance Expectations

### Minimum MVP
- Precision: > 0.60
- Recall: > 0.50
- F1 Score: > 0.55

### Good Performance
- Precision: > 0.70
- Recall: > 0.60
- F1 Score: > 0.65

### Excellent Performance
- Precision: > 0.80
- Recall: > 0.75
- F1 Score: > 0.77

---

## Error Handling

### Common Errors

**1. "release_year not in index"**
- **Cause:** DataFrame has release_date, not release_year
- **Fix:** recommender_engine.py handles both (extracts year from date)

**2. "Could not find data/movies_final.csv"**
- **Cause:** Data not exported locally
- **Solutions:**
  - Run in Databricks notebook
  - Export: `spark.read.table(...).toPandas().to_csv(...)`

**3. "Empty recommended movies"**
- **Cause:** Test scenarios not populated with real movieIds
- **Fix:** Run `explore` action first, copy movieIds to test_data.py

**4. "UnicodeEncodeError"**
- **Cause:** Windows default encoding doesn't support special characters
- **Fix:** Use ASCII alternative (e.g., [OK] instead of ✓)

---

## Extending the Module

### Adding Custom Metrics

1. Add function to `metrics.py`:
```python
def calculate_custom_metric(predicted_ids, relevant_ids):
    """Your metric formula here"""
    return score
```

2. Call in `evaluator.py`:
```python
result['custom_metric'] = calculate_custom_metric(
    predicted_ids, relevant_ids
)
```

### Adding Test Scenarios

1. Update `test_data.py`:
```python
TEST_SCENARIOS.append({
    "id": "test_11_custom",
    "user_input": "Your query here",
    "relevant_movie_ids": [movie_id_1, movie_id_2, ...],
    "description": "Description"
})
```

2. Run evaluation:
```bash
python setup_and_run.py --action run
```

---

## Files Generated by Evaluation

| File | Purpose | Format |
|------|---------|--------|
| `evaluation_results.csv` | Detailed per-scenario results | CSV |
| Console output | Human-readable summary | Text |
| Memory (self.results) | Programmatic access to results | Python dict |

---

## Dependencies

```
Core Dependencies:
├── pandas         (Data manipulation)
├── numpy          (Numerical computing)
├── scikit-learn   (TF-IDF vectorization, metrics)
└── joblib         (Load ML models)

Optional (for flowchart generation):
└── matplotlib     (Visualization)
```

---

## Testing the Module

```bash
# Explore available data
python setup_and_run.py --action explore

# View test scenarios
python -c "from src.metrics.test_data import print_test_scenarios; print_test_scenarios()"

# Run full evaluation
python setup_and_run.py --action run

# View results
cat evaluation_results.csv
```

---

## Summary

The Metrics module provides a complete evaluation framework:

1. **metrics.py** - Pure calculation functions (no side effects)
2. **test_data.py** - Ground truth test cases
3. **evaluator.py** - Orchestration & aggregation
4. **benchmark.py** - Live-pipeline genre-overlap benchmark (CLI)
5. **setup_and_run.py** - CLI interface for common tasks

All modules work together to measure recommendation quality across 10 diverse test scenarios, providing precision, recall, accuracy, F1 score, and MRR metrics.
