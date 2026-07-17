# Metrics Evaluation Module for CineAssist

This module provides comprehensive evaluation tools for the CineAssist recommendation chatbot using standard information retrieval metrics.

## Overview

The metrics module evaluates how well the chatbot recommender system performs by comparing its predictions against ground-truth relevant movies.

## Metrics Explained

### Precision
**Formula:** (# of relevant items recommended) / (# of items recommended)

- **Meaning:** Of the movies we recommended, how many were actually relevant to the user?
- **Range:** 0.0 to 1.0 (higher is better)
- **Use case:** Important when you want high-quality recommendations (minimize bad recommendations)

**Example:**
- Recommended: [Movie1, Movie2, Movie3, Movie4, Movie5]
- Relevant: [Movie1, Movie2, Movie6, Movie7]
- True Positives: 2 (Movie1, Movie2)
- Precision = 2/5 = 0.4

### Recall
**Formula:** (# of relevant items recommended) / (# of all relevant items)

- **Meaning:** Of all the relevant movies, how many did we return?
- **Range:** 0.0 to 1.0 (higher is better)
- **Use case:** Important when you want comprehensive results (don't miss relevant movies)

**Example:**
- Recommended: [Movie1, Movie2, Movie3, Movie4, Movie5]
- Relevant: [Movie1, Movie2, Movie6, Movie7]
- True Positives: 2 (Movie1, Movie2)
- Recall = 2/4 = 0.5

### Accuracy
**Formula:** (True Positives + True Negatives) / Total Items

- **Meaning:** What fraction of all items were correctly classified as relevant/not relevant?
- **Range:** 0.0 to 1.0 (higher is better)
- **Note:** Can be misleading if dataset is highly imbalanced

### F1 Score
**Formula:** 2 × (Precision × Recall) / (Precision + Recall)

- **Meaning:** Harmonic mean of precision and recall
- **Range:** 0.0 to 1.0 (higher is better)
- **Use case:** Good for balanced evaluation when both precision and recall matter

### Mean Reciprocal Rank (MRR)
**Formula:** 1 / (rank of first relevant item)

- **Meaning:** How early does the first relevant item appear in the ranking?
- **Range:** 0.0 to 1.0 (higher is better)
- **Use case:** Measures ranking quality of results

## Module Structure

```
src/metrics/
├── __init__.py              # Package initialization
├── metrics.py               # Core metric calculation functions
├── test_data.py             # Predefined test scenarios
├── evaluator.py             # Main Evaluator class (ID-based ground-truth eval)
├── benchmark.py             # Live-pipeline genre-overlap benchmark (CLI: python -m src.metrics.benchmark)
└── README.md                # This file
```

## Usage

### Basic Usage

```python
from src.metrics import Evaluator
from src.recommender.recommender_engine import recommend_on_the_fly
import pandas as pd
import joblib

# Load your data and models
movies_df = pd.read_csv("data/movies_final.csv")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")

# Create evaluator
evaluator = Evaluator(
    recommender_func=recommend_on_the_fly,
    movies_df=movies_df,
    vectorizer=vectorizer,
    tfidf_matrix=tfidf_matrix
)

# Evaluate all test scenarios
results = evaluator.evaluate_all_scenarios(top_n=5)

# Print results
evaluator.print_results(results)

# Get average metrics
avg_metrics = evaluator.get_average_metrics(results)
print(f"Average Precision: {avg_metrics['avg_precision']:.3f}")
```

### Evaluate Single Query

```python
user_input = "I want an action movie with lots of explosions"
relevant_ids = [1, 2, 3, 5, 50]  # Ground truth

result = evaluator.evaluate_single_query(user_input, relevant_ids, top_n=5)

print(f"Precision: {result['precision']:.3f}")
print(f"Recall: {result['recall']:.3f}")
print(f"F1 Score: {result['f1_score']:.3f}")
```

### View Test Scenarios

```python
from src.metrics.test_data import print_test_scenarios, get_test_scenarios

# Print all scenarios
print_test_scenarios()

# Get specific scenario
scenarios = get_test_scenarios()
```

### Export Results to CSV

```python
evaluator.export_results_csv("evaluation_results.csv", results)
```

## Test Scenarios

The module includes 10 predefined test scenarios covering:

1. **test_01_action_movie** - Action movies with fighting/explosions
2. **test_02_comedy** - Funny comedy movies
3. **test_03_drama_emotional** - Emotional drama films
4. **test_04_scifi_future** - Sci-Fi with robots/space themes
5. **test_05_horror_scary** - Scary horror movies
6. **test_06_romance_love** - Romantic love stories
7. **test_07_animation** - Animated cartoons
8. **test_08_year_constraint** - Movies from specific era (2000s)
9. **test_09_rating_high** - Highly-rated movies (>8.0)
10. **test_10_multilingual** - Movies in Spanish language

Each scenario includes:
- `user_input`: Natural language user query
- `relevant_movie_ids`: Ground-truth movie IDs that should be recommended
- `description`: Scenario explanation

**Important:** You need to populate `relevant_movie_ids` in `test_data.py` with actual movie IDs from your dataset based on the scenario criteria.

## Interpretation Guide

### Good Results
- Precision: > 0.7 (70% of recommendations are relevant)
- Recall: > 0.6 (we find most relevant movies)
- F1: > 0.6 (balanced performance)

### Precision vs Recall Trade-off
- **High Precision, Low Recall:** System recommends few movies, but they're mostly relevant (conservative)
- **Low Precision, High Recall:** System recommends many movies, finds most relevant ones (aggressive)
- **Balanced:** Aim for F1 > 0.6 for good overall performance

## Adding Custom Test Scenarios

To add your own test cases, edit `test_data.py`:

```python
TEST_SCENARIOS.append({
    "id": "test_11_custom",
    "user_input": "Your custom user input",
    "relevant_movie_ids": [123, 456, 789],  # Actual movie IDs from your dataset
    "description": "Description of this test case"
})
```

## Limitations

1. **Manual Ground Truth:** Relevant movie IDs must be manually curated for each test scenario
2. **Limited to Top-N:** Evaluates only top-N recommendations (adjust with `top_n` parameter)
3. **No Ranking Weights:** All relevant movies weighted equally (no partial credit for close misses)
4. **Categorical Evaluation:** Movies are classified as relevant or not (no relevance scores)

## Future Enhancements

- Add ranking-aware metrics (NDCG - Normalized Discounted Cumulative Gain)
- Support weighted relevance (some movies more relevant than others)
- Interactive evaluation UI
- A/B testing framework
- Temporal evaluation (how metrics change over time)

## Files Generated

When you run evaluations:
- `evaluation_results.csv` - Detailed results for each test case
- Console output with metrics and averages
