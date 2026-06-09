# 🎬 CineAssist Metrics - Quick Start Guide

## 3-Step Process to Check Chatbot Metrics

### **STEP 1: Explore Your Movie Data** 🔍

Run this command to see what movies you have:

```bash
python src/metrics/setup_and_run.py --action explore
```

**Output will show:**
- Total number of movies
- Sample movies with their IDs
- Movies grouped by genre (with count)
- High-rated movies (rating > 8.0)
- Movies from specific decades
- Movies in different languages

**Example output:**
```
ACTION movies (first 5 movieIds):
  [1, 2, 3, 5, 50]

COMEDY movies (first 5 movieIds):
  [10, 15, 20, 25, 30]

DRAMA movies (first 5 movieIds):
  [40, 42, 45, 48, 52]
```

---

### **STEP 2: Update Test Data with Real MovieIds** ✏️

Edit: `src/metrics/test_data.py`

For each test scenario, replace `relevant_movie_ids` with actual movieIds from Step 1:

**Before (placeholder):**
```python
{
    "id": "test_01_action_movie",
    "user_input": "I want an action movie with lots of fights and explosions",
    "relevant_movie_ids": [1, 2, 3, 5, 50],  # ← Copy from explore output
    "description": "Action movie with fighting and explosions"
}
```

**After (updated):**
```python
{
    "id": "test_01_action_movie",
    "user_input": "I want an action movie with lots of fights and explosions",
    "relevant_movie_ids": [1, 2, 3, 5, 50],  # ← Real movieIds
    "description": "Action movie with fighting and explosions"
}
```

**Simple tip:** Copy-paste the movieIds from Step 1 output into the `relevant_movie_ids` list.

---

### **STEP 3: Run Evaluation** ▶️

```bash
python src/metrics/setup_and_run.py --action run
```

**Output will show:**
```
================================================================================
CINEASSIST RECOMMENDATION EVALUATION RESULTS
================================================================================

[test_01_action_movie] Action movie with fighting and explosions
User Input: I want an action movie with lots of fights and explosions

Recommended Movie IDs: [1, 2, 5, 10, 15]
Relevant Movie IDs:    [1, 2, 3, 5, 50]

Metrics:
  Precision:  0.800
  Recall:     0.600
  Accuracy:   0.950
  F1 Score:   0.686
  MRR:        1.000

...more scenarios...

================================================================================
AVERAGE METRICS
================================================================================
Average Precision:  0.750
Average Recall:     0.680
Average Accuracy:   0.920
Average F1 Score:   0.710
Average MRR:        0.850
Total Tests:        10
================================================================================
```

Results also saved to: `evaluation_results.csv`

---

## 📊 Understanding the Metrics

| Metric | What It Means | Good Score |
|--------|---|---|
| **Precision** | Of movies we recommended, what % were relevant? | > 0.70 |
| **Recall** | Of all relevant movies, what % did we return? | > 0.60 |
| **Accuracy** | Overall % of correct recommendations | > 0.70 |
| **F1 Score** | Balance between precision & recall | > 0.60 |
| **MRR** | How early does the first relevant movie appear? | > 0.50 |

### Example Interpretation:
- **Precision 0.800** = 80% of our recommendations were relevant (good quality)
- **Recall 0.600** = We found 60% of all relevant movies (decent coverage)
- **F1 0.686** = Balanced performance between precision and recall

---

## Troubleshooting

### ❌ "Could not find data/movies_final.csv"

**Solution:** The data is in Databricks. Two options:

**Option A:** Run in Databricks notebook
```python
%run /Users/your_username/path/to/setup_and_run.py
```

**Option B:** Export data as CSV first
1. In Databricks notebook:
   ```python
   movies_df = spark.read.table("workspace.datasets.movies_final").toPandas()
   movies_df.to_csv("/Workspace/path/to/project/data/movies_final.csv", index=False)
   ```
2. Then run `python src/metrics/setup_and_run.py --action explore`

---

### ❌ "Could not load tfidf_vectorizer.pkl"

Make sure these files exist in `models/` folder:
- `tfidf_vectorizer.pkl`
- `tfidf_matrix.pkl`

If not, run the vectorization notebook first:
```
notebooks/03_Vectorization.ipynb
```

---

### ❌ "Test scenarios have empty relevant_movie_ids"

You haven't updated `test_data.py` yet. Do Step 2 first:

1. Run: `python src/metrics/setup_and_run.py --action explore`
2. Copy movieIds from output
3. Update: `src/metrics/test_data.py`
4. Run: `python src/metrics/setup_and_run.py --action run`

---

## Complete Workflow (One Command Per Terminal)

```bash
# Terminal 1: Explore data
python src/metrics/setup_and_run.py --action explore

# (Copy movieIds from output, update test_data.py)

# Terminal 2: Run evaluation
python src/metrics/setup_and_run.py --action run
```

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `src/metrics/test_data.py` | ✏️ **YOU UPDATE THIS** with real movieIds |
| `src/metrics/setup_and_run.py` | Helper script (don't modify) |
| `src/metrics/metrics.py` | Metric calculations (don't modify) |
| `src/metrics/evaluator.py` | Main evaluator (don't modify) |
| `evaluation_results.csv` | Generated after running evaluation |

---

## Common Movieid Patterns

When you run `explore`, movieIds are usually sequential:
- Action: 1, 2, 3, 5, 50, ...
- Comedy: 10, 15, 20, 25, 30, ...
- Drama: 40, 42, 45, 48, 52, ...
- Horror: 100, 105, 110, 115, 120, ...

Just copy 5-6 movieIds per scenario as shown in the explore output.

---

## Need Help?

See full documentation: `src/metrics/README.md`
