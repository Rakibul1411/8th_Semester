#!/usr/bin/env python3
"""Mine Java snapshots from a local Git repository using PyDriller."""

from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BUG_FIX_PATTERN = re.compile(
    r"\b(?:fix(?:ed|es|ing)?|bug(?:s|fix)?|issue(?:s)?)\b|\bLANG-\d+\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Snapshot:
    commit_id: str
    commit_date: str
    file_name: str
    source_code: str
    label: int


def is_bug_fix(message: str) -> bool:
    """Return whether a message has a bug keyword or Commons Lang Jira ID."""
    return bool(BUG_FIX_PATTERN.search(message or ""))


def reservoir_add(
    reservoir: list[Snapshot], item: Snapshot, seen: int, limit: int, rng: random.Random
) -> None:
    """Uniformly sample `limit` items without retaining all candidates in memory."""
    if len(reservoir) < limit:
        reservoir.append(item)
        return
    replacement = rng.randrange(seen)
    if replacement < limit:
        reservoir[replacement] = item


def _is_wanted_java(path: str | None, include_tests: bool) -> bool:
    if not path or not path.endswith(".java"):
        return False
    normalized = path.replace("\\", "/")
    return include_tests or "/test/" not in f"/{normalized}"


def mine(
    repository_path: Path,
    max_defective: int,
    max_clean: int,
    seed: int,
    include_tests: bool = False,
    max_commits: int | None = None,
) -> list[Snapshot]:
    try:
        from pydriller import Repository
    except ImportError as exc:
        raise SystemExit(
            "PyDriller is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    if not (repository_path / ".git").exists():
        raise ValueError(f"Not a Git repository: {repository_path}")
    if max_defective < 1 or max_clean < 1:
        raise ValueError("Sample limits must be positive")

    rng = random.Random(seed)
    defective: list[Snapshot] = []
    clean: list[Snapshot] = []
    seen_defective = 0
    seen_clean = 0

    for commit_number, commit in enumerate(
        Repository(str(repository_path), only_no_merge=True).traverse_commits(), start=1
    ):
        fix_commit = is_bug_fix(commit.msg)
        for modified in commit.modified_files:
            # Defective examples are the file immediately before its fixing change.
            # Clean examples represent the non-fix commit after its change.
            path = modified.old_path if fix_commit else modified.new_path
            source = modified.source_code_before if fix_commit else modified.source_code
            if not _is_wanted_java(path, include_tests) or not source:
                continue

            item = Snapshot(
                commit_id=commit.hash,
                commit_date=commit.committer_date.isoformat(),
                file_name=path,
                source_code=source,
                label=int(fix_commit),
            )
            if fix_commit:
                seen_defective += 1
                reservoir_add(defective, item, seen_defective, max_defective, rng)
            else:
                seen_clean += 1
                reservoir_add(clean, item, seen_clean, max_clean, rng)

        if max_commits is not None and commit_number >= max_commits:
            break

    rows = defective + clean
    rng.shuffle(rows)
    print(
        f"Candidates: defective={seen_defective}, clean={seen_clean}; "
        f"selected: defective={len(defective)}, clean={len(clean)}"
    )
    return rows


def write_csv(rows: Iterable[Snapshot], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Commit_ID", "Commit_Date", "File_Name", "Source_Code", "Label"])
        for row in rows:
            writer.writerow(
                [row.commit_id, row.commit_date, row.file_name, row.source_code, row.label]
            )


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=project_dir.parent / "commons-lang",
        help="Path to the local commons-lang Git repository",
    )
    parser.add_argument(
        "--output", type=Path, default=project_dir / "data" / "mined_files.csv"
    )
    parser.add_argument("--max-defective", type=int, default=500)
    parser.add_argument("--max-clean", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-tests", action="store_true")
    parser.add_argument(
        "--max-commits",
        type=int,
        default=None,
        help="Optional traversal limit, useful for a quick smoke test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = mine(
        args.repo.resolve(),
        args.max_defective,
        args.max_clean,
        args.seed,
        args.include_tests,
        args.max_commits,
    )
    if not rows:
        raise SystemExit("No eligible Java snapshots found")
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()

