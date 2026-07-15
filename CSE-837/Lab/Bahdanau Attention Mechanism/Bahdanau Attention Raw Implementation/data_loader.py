"""Load the repository's canonical XSum JSON splits using the standard library."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import json
from pathlib import Path
import warnings

from preprocessing import tokenize


_XSUM_FILENAMES = (
    "xsum_train.json",
    "xsum_validation.json",
    "xsum_test.json",
)
_REQUIRED_FIELDS = ("document", "summary", "id")
Tokenizer = Callable[[str], list[str]]


def _validate_limit(limit: int | None) -> None:
    if limit is None:
        return
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an integer or None")
    if limit <= 0:
        raise ValueError("limit must be positive or None")


def _record_label(record: object, index: int) -> str:
    if isinstance(record, dict):
        record_id = record.get("id")
        if isinstance(record_id, str) and record_id.strip():
            return record_id
    return f"<index:{index}>"


def _validation_error(record: object) -> str | None:
    if not isinstance(record, dict):
        return "record must be a JSON object"

    missing = [field for field in _REQUIRED_FIELDS if field not in record]
    if missing:
        return f"missing required field(s): {', '.join(missing)}"

    non_string = [
        field for field in _REQUIRED_FIELDS if not isinstance(record[field], str)
    ]
    if non_string:
        return f"field(s) must be strings: {', '.join(non_string)}"
    if not record["id"].strip():
        return "id must not be empty"
    if not tokenize(record["document"]):
        return "document has no tokens after preprocessing"
    if not tokenize(record["summary"]):
        return "summary has no tokens after preprocessing"
    return None


def _read_xsum_split(
    path: Path,
    limit: int | None,
    strict: bool,
) -> tuple[list[dict[str, str]], list[str]]:
    if path.name not in _XSUM_FILENAMES:
        allowed = ", ".join(_XSUM_FILENAMES)
        raise ValueError(f"XSum split filename must be one of: {allowed}")
    if not path.is_file():
        raise FileNotFoundError(f"XSum split not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a top-level JSON array")

    valid_samples: list[dict[str, str]] = []
    skipped_ids: list[str] = []
    for index, record in enumerate(payload):
        reason = _validation_error(record)
        if reason is not None:
            label = _record_label(record, index)
            if strict:
                raise ValueError(
                    f"Invalid XSum record {label!r} at index {index} in "
                    f"{path.name}: {reason}"
                )
            skipped_ids.append(label)
            continue

        # Discard any unrelated fields and expose one canonical pipeline schema.
        valid_samples.append(
            {
                "document": record["document"],
                "summary": record["summary"],
                "id": record["id"],
            }
        )
        # Development limits are intentionally applied while walking the
        # payload: parsing the JSON is unavoidable, but tokenizing tens of
        # thousands of unused records is not. Invalid records after the limit
        # are outside the requested sample and are therefore not reported.
        if limit is not None and len(valid_samples) >= limit:
            break
    return valid_samples, skipped_ids


def _warn_about_skipped(skipped: list[str]) -> None:
    if skipped:
        warnings.warn(
            f"Skipped {len(skipped)} invalid XSum sample(s): {', '.join(skipped)}",
            RuntimeWarning,
            stacklevel=3,
        )


def load_xsum_split(
    path: str | Path,
    limit: int | None = None,
    strict: bool = False,
) -> list[dict[str, str]]:
    """Load one canonical XSum split and return ``document/summary/id`` records.

    By default, malformed or token-empty records are omitted and their IDs are
    reported in one warning. With ``strict=True``, the first invalid record raises
    ``ValueError`` instead.
    """
    _validate_limit(limit)
    if not isinstance(strict, bool):
        raise TypeError("strict must be a boolean")
    split_path = Path(path).expanduser()
    samples, skipped_ids = _read_xsum_split(split_path, limit, strict)
    if not strict:
        _warn_about_skipped([f"{split_path.name}:{item}" for item in skipped_ids])
    return samples


def load_xsum_dataset(
    path: str | Path,
    limit: int | None = None,
    strict: bool = False,
) -> list[dict[str, str]]:
    """Explicit alias for :func:`load_xsum_split` used by simple callers."""
    return load_xsum_split(path, limit=limit, strict=strict)


def load_xsum_splits(
    dataset_directory: str | Path,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    test_limit: int | None = None,
    strict: bool = False,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    """Load the fixed train, validation, and test files without resplitting them."""
    for limit in (train_limit, validation_limit, test_limit):
        _validate_limit(limit)
    if not isinstance(strict, bool):
        raise TypeError("strict must be a boolean")

    directory = Path(dataset_directory).expanduser()
    if not directory.is_dir():
        raise NotADirectoryError(f"XSum dataset directory not found: {directory}")

    limits = (train_limit, validation_limit, test_limit)
    loaded_splits: list[list[dict[str, str]]] = []
    all_skipped: list[str] = []
    for filename, limit in zip(_XSUM_FILENAMES, limits):
        samples, skipped_ids = _read_xsum_split(
            directory / filename, limit, strict
        )
        loaded_splits.append(samples)
        all_skipped.extend(f"{filename}:{item}" for item in skipped_ids)

    if not strict:
        _warn_about_skipped(all_skipped)
    return loaded_splits[0], loaded_splits[1], loaded_splits[2]


def dataset_statistics(
    samples: Iterable[dict[str, str]],
    tokenizer: Tokenizer = tokenize,
) -> dict[str, float | int]:
    """Calculate XSum sample count and average document/summary token lengths."""
    sample_count = 0
    document_token_count = 0
    summary_token_count = 0
    for sample in samples:
        sample_count += 1
        document_token_count += len(tokenizer(sample["document"]))
        summary_token_count += len(tokenizer(sample["summary"]))

    if sample_count == 0:
        return {
            "samples": 0,
            "average_document_length": 0.0,
            "average_summary_length": 0.0,
        }
    return {
        "samples": sample_count,
        "average_document_length": document_token_count / sample_count,
        "average_summary_length": summary_token_count / sample_count,
    }


def display_dataset_summary(
    samples: list[dict[str, str]],
    tokenizer: Tokenizer = tokenize,
    preview_characters: int = 500,
) -> None:
    """Print XSum statistics and a bounded example document/reference pair."""
    if preview_characters <= 0:
        raise ValueError("preview_characters must be positive")
    stats = dataset_statistics(samples, tokenizer)
    print(f"Number of samples: {stats['samples']}")
    print(
        "Average document length: "
        f"{stats['average_document_length']:.2f} tokens"
    )
    print(f"Average summary length: {stats['average_summary_length']:.2f} tokens")
    if samples:
        document = samples[0]["document"]
        suffix = "..." if len(document) > preview_characters else ""
        print("\nExample document:\n", document[:preview_characters] + suffix)
        print("\nExample summary:\n", samples[0]["summary"])
