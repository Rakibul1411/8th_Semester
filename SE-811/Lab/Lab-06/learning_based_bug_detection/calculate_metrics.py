#!/usr/bin/env python3
"""Calculate Java software metrics once and save a model-ready CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from mine_repository import is_bug_fix
from software_metrics import calculate_advanced_metrics


FEATURE_NAMES = ["LOC", "Cyclomatic_Complexity", "Number_of_Variables"]
ADVANCED_FEATURE_NAMES = [
    "Halstead_Volume", "Halstead_Difficulty", "Halstead_Effort",
    "Comment_Density", "WMC", "CBO", "LCOM",
    "Code_Churn", "File_Age_Days", "Fix_History",
]
OUTPUT_FIELDS = [
    "Commit_ID", "Commit_Date", "File_Name", "Label",
    *FEATURE_NAMES, *ADVANCED_FEATURE_NAMES,
]


def allow_large_csv_fields() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def _path_key(path: str | None) -> str:
    return (path or "").replace("\\", "/")


def _process_metrics(repository_path: Path, requested: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, int | float]]:
    """Build commit/file process metrics in one PyDriller history traversal."""
    try:
        from pydriller import Repository
    except ImportError as exc:
        raise SystemExit(
            "PyDriller is required for process metrics. Run: python -m pip install -r requirements.txt"
        ) from exc

    first_seen: dict[str, datetime] = {}
    cumulative_churn: dict[str, int] = {}
    prior_fixes: dict[str, int] = {}
    results: dict[tuple[str, str], dict[str, int | float]] = {}
    for commit in Repository(str(repository_path), only_no_merge=True).traverse_commits():
        commit_date = commit.committer_date
        is_fix = is_bug_fix(commit.msg)
        for modified in commit.modified_files:
            paths = {_path_key(modified.new_path), _path_key(modified.old_path)} - {""}
            additions = int(modified.added_lines or 0)
            deletions = int(modified.deleted_lines or 0)
            for path in paths:
                first_seen.setdefault(path, commit_date)
                cumulative_churn[path] = cumulative_churn.get(path, 0) + additions + deletions
                age_days = max(0, (commit_date - first_seen[path]).total_seconds() / 86400)
                values = {
                    "Code_Churn": cumulative_churn[path],
                    "File_Age_Days": age_days,
                    "Fix_History": prior_fixes.get(path, 0),
                }
                key = (commit.hash, path)
                if key in requested:
                    results[key] = values
            if is_fix:
                for path in paths:
                    prior_fixes[path] = prior_fixes.get(path, 0) + 1
    return results


def calculate_dataset(input_path: Path, output_path: Path, repository_path: Path) -> int:
    allow_large_csv_fields()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    requested = {(row["Commit_ID"], _path_key(row["File_Name"])) for row in rows}
    process = _process_metrics(repository_path, requested)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as target:
        required = {"Commit_ID", "Commit_Date", "File_Name", "Source_Code", "Label"}
        missing = required - set(rows[0] if rows else [])
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")
        writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            metrics = calculate_advanced_metrics(row["Source_Code"])
            process_metrics = process.get(
                (row["Commit_ID"], _path_key(row["File_Name"])),
                {"Code_Churn": 0, "File_Age_Days": 0.0, "Fix_History": 0},
            )
            writer.writerow(
                {
                    "Commit_ID": row["Commit_ID"],
                    "Commit_Date": row["Commit_Date"],
                    "File_Name": row["File_Name"],
                    "Label": row["Label"],
                    **metrics,
                    **process_metrics,
                }
            )
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=project_dir / "data" / "mined_files.csv"
    )
    parser.add_argument(
        "--output", type=Path,
        default=project_dir / "data" / "files_with_advanced_metrics.csv",
        help="Output CSV containing original and expanded metrics",
    )
    parser.add_argument(
        "--repo", type=Path, default=project_dir.parent / "commons-lang",
        help="Local Git repository used for process metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = calculate_dataset(args.input, args.output, args.repo.resolve())
    print(f"Calculated metrics for {count} files")
    print(f"Wrote metrics dataset to {args.output.resolve()}")


if __name__ == "__main__":
    main()
