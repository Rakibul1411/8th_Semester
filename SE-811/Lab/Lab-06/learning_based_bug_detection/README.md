# Lab 06 — Learning-Based Bug Detection

This implementation mines the local `commons-lang` Git history with PyDriller,
creates a labeled Java-source dataset, calculates three traditional software
metrics, and compares five classifiers.

## Labeling policy

- **Label 1 (Defective):** the commit message contains `fix`, `bug`, `issue`, or
  a `LANG-<number>` Jira ID. The saved source is `source_code_before`, i.e. the
  version immediately before the fixing change.
- **Label 0 (Clean):** the message does not match the rule. A uniformly random
  subset is retained, and the source saved is the version after that non-fix
  change.
- Only production `.java` files are included by default. Use `--include-tests`
  if test sources are also wanted.

Commit-message labeling is a practical heuristic, not ground truth: some fixes
have vague messages and some messages use these words in a non-defect context.

## Setup and execution

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Task 1: traverse commons-lang and create the required five-column CSV
python mine_repository.py

# Task 2a: calculate all metrics once (skip this if the advanced CSV already exists)
python calculate_metrics.py

# Task 2b: load the existing advanced metrics, perform an 80/20 split, and print reports
python train_models.py
```

The default repository path is `../commons-lang`. Outputs are:

- `data/mined_files.csv`: `Commit_ID`, `Commit_Date`, `File_Name`,
  `Source_Code`, `Label`
- `data/files_with_metrics.csv`: metadata, `Label`, `LOC`,
  `Cyclomatic_Complexity`, and `Number_of_Variables` (source text is omitted
  from this model-ready dataset)
- `data/files_with_advanced_metrics.csv`: the original metrics plus Halstead,
  comment-density, CK, and Git process metrics (source text is omitted)

The miner uses reservoir sampling with seed 42 and keeps at most 500 files per
label, producing a balanced and reproducible dataset without retaining every
candidate in memory. Change this with `--max-defective`, `--max-clean`, and
`--seed`. For a quick check, add `--max-commits 500`; omit it for the requested
full-history traversal.

The metric calculation is performed only by `calculate_metrics.py`. The
training script reads the existing advanced metrics CSV and does not recalculate
them. It prints separate comparisons for the original three features and the
expanded feature set. To train only on the original CSV, use
`python train_models.py --input data/files_with_metrics.csv`.

The cyclomatic metric is `1 +` the lexical count of Java `if`, `for`, `while`,
`case`, `catch`, ternary, `&&`, and `||` decision points. Variable count is an
explainable lexical approximation of fields, parameters, and local declaration
names. Comments and literals are removed before all metric calculations.

Expanded metrics include Halstead Volume, Difficulty, and Effort; comment
density; approximate CK metrics WMC, CBO, and LCOM; cumulative code churn; file
age in days; and prior fix-history count. Process metrics are derived from the
local Git history using PyDriller.

Models evaluated: Logistic Regression, Decision Tree, Random Forest, SVM,
K-Nearest Neighbors, Gaussian Naive Bayes, Gradient Boosting, and Extra Trees.
Logistic Regression, SVM, and K-Nearest Neighbors use standardized features.
For every model, the terminal output includes Precision, Recall, F1-score, and
ROC-AUC (with ROC-AUC calculated from probabilities or decision scores).

## Tests

The tests need only the Python standard library:

```bash
python -m unittest discover -s tests -v
```
