#!/usr/bin/env python3
"""Engineer Java metrics and evaluate traditional defect classifiers."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

FEATURE_NAMES = ["LOC", "Cyclomatic_Complexity", "Number_of_Variables"]
ADVANCED_FEATURE_NAMES = [
    "Halstead_Volume", "Halstead_Difficulty", "Halstead_Effort",
    "Comment_Density", "WMC", "CBO", "LCOM",
    "Code_Churn", "File_Age_Days", "Fix_History",
]


def allow_large_csv_fields() -> None:
    """Permit historical source files larger than csv's 128 KiB default field."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def load_metrics_dataset(
    input_path: Path,
) -> tuple[list[list[float]], list[list[float]], list[int], bool]:
    allow_large_csv_fields()
    original_features: list[list[float]] = []
    expanded_features: list[list[float]] = []
    labels: list[int] = []
    with input_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = set(FEATURE_NAMES + ["Label"]) - columns
        if missing:
            raise ValueError(
                f"Metrics CSV is missing columns: {sorted(missing)}. "
                "Run calculate_metrics.py first."
            )
        has_advanced = set(ADVANCED_FEATURE_NAMES).issubset(columns)
        for row in reader:
            label = int(row["Label"])
            original_features.append([float(row[name]) for name in FEATURE_NAMES])
            expanded_features.append(
                [float(row[name]) for name in FEATURE_NAMES + ADVANCED_FEATURE_NAMES]
                if has_advanced else []
            )
            labels.append(label)
    return original_features, expanded_features, labels, has_advanced


def evaluate(features: list[list[float]], labels: list[int], seed: int, title: str) -> None:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            classification_report,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    if len(features) < 10 or len(set(labels)) != 2:
        raise ValueError("Training needs at least 10 rows containing both labels")
    class_counts = {label: labels.count(label) for label in set(labels)}
    if min(class_counts.values()) < 2:
        raise ValueError("Each label needs at least two rows for a stratified split")

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.20, random_state=seed, stratify=labels
    )
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=5)
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, random_state=seed, n_jobs=-1
        ),
    }
    print(f"\n{'#' * 18} {title} {'#' * 18}")
    print(f"Training rows: {len(x_train)}; testing rows: {len(x_test)}")
    comparison: list[dict[str, float | str]] = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        # Use probabilities when a classifier exposes them; SVM uses its
        # decision function because probability=True is not enabled.
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(x_test)
        else:
            scores = predictions
        comparison.append(
            {
                "Model": name,
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1-Score": f1_score(y_test, predictions, zero_division=0),
                "AUC-ROC": roc_auc_score(y_test, scores),
            }
        )
        print(f"\n{'=' * 16} {name} {'=' * 16}")
        print(
            classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Clean (0)", "Defective (1)"],
                zero_division=0,
            )
        )

    print("\n" + "=" * 22 + f" {title} MODEL COMPARISON " + "=" * 22)
    print(f"{'Model':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'AUC-ROC':>10}")
    print("-" * 70)
    for result in comparison:
        print(
            f"{result['Model']:<25} {result['Precision']:>10.3f} "
            f"{result['Recall']:>10.3f} {result['F1-Score']:>10.3f} "
            f"{result['AUC-ROC']:>10.3f}"
        )


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=project_dir / "data" / "files_with_advanced_metrics.csv",
        help="Metrics CSV; use files_with_metrics.csv for original features only",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original, expanded, labels, has_advanced = load_metrics_dataset(args.input)
    evaluate(original, labels, args.seed, "ORIGINAL FEATURES")
    if has_advanced:
        evaluate(expanded, labels, args.seed, "EXPANDED FEATURES")
    else:
        print("Expanded metrics are not present; only original features were evaluated.")


if __name__ == "__main__":
    main()
