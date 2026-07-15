"""Load an XSum checkpoint and generate a summary with greedy decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from decoder import Decoder
from embedding import UNK_TOKEN, Vocabulary
from encoder import Encoder


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_DIRECTORY = SCRIPT_DIRECTORY / "checkpoints" / "xsum_model"


def _restore_parameters(
    archive_path: Path,
    named_parameters: list[tuple[str, np.ndarray, np.ndarray]],
    zero_if_missing: set[str] | None = None,
) -> None:
    """Copy saved arrays into model parameters after checking names and shapes.

    ``zero_if_missing`` is used only for explicitly compatible parameters added
    after an older checkpoint was written.  All other missing arrays remain a
    hard error.
    """
    optional_names = zero_if_missing or set()
    with np.load(archive_path, allow_pickle=False) as archive:
        expected_names = {name for name, _, _ in named_parameters}
        missing_names = expected_names.difference(archive.files)
        required_missing_names = missing_names.difference(optional_names)
        if required_missing_names:
            raise ValueError(
                f"Checkpoint {archive_path} is missing parameters: "
                f"{sorted(required_missing_names)}"
            )

        for name, parameter, _ in named_parameters:
            if name in missing_names:
                parameter.fill(0.0)
                continue
            stored = archive[name]
            if stored.shape != parameter.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: expected {parameter.shape}, "
                    f"got {stored.shape}"
                )
            parameter[...] = stored


def load_checkpoint(
    checkpoint_directory: str | Path,
) -> tuple[Vocabulary, Encoder, Decoder, dict[str, object]]:
    """Reconstruct an encoder-decoder model from an XSum checkpoint.

    Legacy checkpoints are rejected explicitly instead of being silently loaded with
    incompatible dataset assumptions.
    """
    directory = Path(checkpoint_directory).expanduser()
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"XSum checkpoint not found at {directory}; run `python train.py` first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset") != "xsum":
        raise ValueError(
            f"Checkpoint {directory} is not an XSum checkpoint: "
            "metadata.json must contain dataset='xsum'"
        )

    vocabulary = Vocabulary.load(directory / "vocabulary.json")
    embedding_dimension = int(metadata["embedding_dimension"])
    hidden_size = int(metadata["hidden_size"])
    attention_size = int(metadata["attention_size"])
    rng = np.random.default_rng(int(metadata.get("seed", 42)))

    encoder = Encoder(
        len(vocabulary),
        embedding_dimension,
        hidden_size,
        pad_id=vocabulary.pad_id,
        rng=rng,
    )
    decoder = Decoder(
        len(vocabulary),
        embedding_dimension,
        hidden_size,
        hidden_size,
        attention_size,
        pad_id=vocabulary.pad_id,
        rng=rng,
    )
    _restore_parameters(directory / "encoder.npz", encoder.named_parameters())
    _restore_parameters(
        directory / "decoder.npz",
        decoder.named_parameters(),
        # Format-v2 checkpoints predate the direct source-context output path.
        # Zero initialization preserves their original predictions and permits
        # a neural warm start that learns the new path.
        zero_if_missing={"decoder.output.W_context"},
    )
    return vocabulary, encoder, decoder, metadata


def _normalize_literal_line_breaks(text: str) -> str:
    """Interpret shell-passed ``\\n``/``\\r`` sequences as real line breaks."""
    return (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
    )


def _blocked_ngram_completions(
    generated_ids: list[int], ngram_size: int
) -> set[int]:
    """Return tokens that would repeat an already generated n-gram."""
    if ngram_size <= 0 or len(generated_ids) < ngram_size - 1:
        return set()
    if ngram_size == 1:
        return set(generated_ids)

    prefix = tuple(generated_ids[-(ngram_size - 1) :])
    blocked: set[int] = set()
    for start in range(len(generated_ids) - ngram_size + 1):
        if tuple(generated_ids[start : start + ngram_size - 1]) == prefix:
            blocked.add(generated_ids[start + ngram_size - 1])
    return blocked


def generate_summary(
    article: str,
    vocabulary: Vocabulary,
    encoder: Encoder,
    decoder: Decoder,
    max_article_length: int = 256,
    max_summary_length: int = 40,
    no_repeat_ngram_size: int = 3,
    repetition_penalty: float = 1.1,
    minimum_summary_length: int = 3,
) -> dict[str, object]:
    """Generate one abstractive summary by pure greedy autoregressive decoding.

    The encoder consumes source IDs with shape ``(source_length,)``. At decoder
    step ``t``, the previously predicted ID and states with shape ``(hidden_size,)``
    produce a vocabulary distribution. Its argmax becomes the next decoder input.
    """
    if not isinstance(article, str):
        raise TypeError("article must be a string")
    if max_article_length <= 0:
        raise ValueError("max_article_length must be positive")
    if max_summary_length <= 0:
        raise ValueError("max_summary_length must be positive")
    if (
        isinstance(no_repeat_ngram_size, bool)
        or not isinstance(no_repeat_ngram_size, int)
        or no_repeat_ngram_size < 0
    ):
        raise ValueError("no_repeat_ngram_size must be a non-negative integer")
    if not np.isfinite(repetition_penalty) or repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1.0")
    if (
        isinstance(minimum_summary_length, bool)
        or not isinstance(minimum_summary_length, int)
        or minimum_summary_length < 0
    ):
        raise ValueError("minimum_summary_length must be a non-negative integer")
    # Keep small diagnostic calls such as max_summary_length=1 usable even with
    # the normal three-token minimum; the effective minimum cannot exceed the
    # requested generation budget.
    minimum_summary_length = min(minimum_summary_length, max_summary_length)

    # In POSIX shells, "...\\n..." in ordinary double quotes contains a
    # literal backslash and ``n``.  Without this normalization, names after the
    # separator become artificial tokens such as ``ncooke`` and ``nleague``.
    article = _normalize_literal_line_breaks(article)

    article_ids = vocabulary.encode(
        article,
        add_eos=True,
        max_length=max_article_length,
    )
    if not article_ids:
        article_ids = [vocabulary.eos_id]

    encoder_cache = encoder.forward(article_ids)
    encoder_hidden_states = encoder_cache["hidden_states"]
    hidden = np.asarray(encoder_cache["final_hidden"], dtype=np.float64).copy()
    cell = np.asarray(encoder_cache["final_cell"], dtype=np.float64).copy()

    previous_word_id = vocabulary.sos_id
    generated_ids: list[int] = []
    attention_rows: list[np.ndarray] = []
    termination_reason = "maximum length reached"

    for _ in range(max_summary_length):
        step = decoder.step_forward(
            previous_word_id,
            hidden,
            cell,
            encoder_hidden_states,
        )

        # Greedy selection is made from the neural logits after constraints.
        # These masks do not insert text or copy a source sentence.
        scores = np.asarray(step["logits"], dtype=np.float64).copy()
        scores[~np.isfinite(scores)] = -np.inf
        if repetition_penalty > 1.0:
            for word_id in set(generated_ids):
                scores[word_id] = (
                    scores[word_id] / repetition_penalty
                    if scores[word_id] >= 0.0
                    else scores[word_id] * repetition_penalty
                )
        scores[vocabulary.pad_id] = -np.inf
        scores[vocabulary.sos_id] = -np.inf
        scores[vocabulary.unk_id] = -np.inf
        if len(generated_ids) < minimum_summary_length:
            scores[vocabulary.eos_id] = -np.inf
        for word_id in _blocked_ngram_completions(
            generated_ids, no_repeat_ngram_size
        ):
            scores[word_id] = -np.inf
        if not np.any(np.isfinite(scores)):
            # A toy vocabulary can contain no legal non-EOS token (for example,
            # only the four special IDs).  Relax only the minimum-length mask in
            # that impossible case and still let the neural EOS score terminate.
            eos_score = float(np.asarray(step["logits"])[vocabulary.eos_id])
            if not np.isfinite(eos_score):
                raise RuntimeError("all neural decoder candidates were masked")
            scores[vocabulary.eos_id] = eos_score
        next_word_id = int(np.argmax(scores))

        if next_word_id == vocabulary.eos_id:
            termination_reason = "<EOS> predicted"
            break

        generated_ids.append(next_word_id)
        attention_rows.append(
            np.asarray(step["attention_weights"], dtype=np.float64).copy()
        )

        # Autoregression: y_t becomes the decoder input at step t+1.
        previous_word_id = next_word_id
        hidden = np.asarray(step["hidden"], dtype=np.float64)
        cell = np.asarray(step["cell"], dtype=np.float64)

    generated_words = [
        vocabulary.index_to_word.get(word_id, UNK_TOKEN)
        for word_id in generated_ids
    ]
    source_words = [
        vocabulary.index_to_word.get(int(word_id), UNK_TOKEN)
        for word_id in article_ids
    ]
    attention_matrix = (
        np.vstack(attention_rows)
        if attention_rows
        else np.zeros((0, len(article_ids)), dtype=np.float64)
    )

    return {
        "summary": vocabulary.decode(generated_ids, skip_special=True),
        "generated_ids": generated_ids,
        "generated_words": generated_words,
        "source_words": source_words,
        "attention_weights": attention_matrix,
        "termination_reason": termination_reason,
        "decoding_method": "greedy",
        "decoding_constraints": {
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repetition_penalty": repetition_penalty,
            "minimum_summary_length": minimum_summary_length,
        },
    }


def generate_summary_greedy(
    article: str,
    vocabulary: Vocabulary,
    encoder: Encoder,
    decoder: Decoder,
    max_article_length: int = 256,
    max_summary_length: int = 40,
    no_repeat_ngram_size: int = 3,
    repetition_penalty: float = 1.1,
    minimum_summary_length: int = 3,
) -> dict[str, object]:
    """Compatibility alias for the sole supported greedy decoder."""
    return generate_summary(
        article,
        vocabulary,
        encoder,
        decoder,
        max_article_length=max_article_length,
        max_summary_length=max_summary_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        minimum_summary_length=minimum_summary_length,
    )


def format_generation_report(result: dict[str, object]) -> str:
    """Format a concise report containing only the neural decoder output."""
    # Keep an EOS-only result genuinely empty; do not substitute a canned
    # sentence or diagnostic phrase for the model's output.
    summary = str(result["summary"])
    generated_words = list(result["generated_words"])
    constraints = dict(result.get("decoding_constraints", {}))
    constraint_report = (
        f"no-repeat {constraints.get('no_repeat_ngram_size', 0)}-gram, "
        f"repetition penalty {constraints.get('repetition_penalty', 1.0):g}"
    )
    return "\n".join(
        [
            "Generated summary:",
            summary,
            "",
            f"Generated tokens: {len(generated_words)}",
            f"Decoder stopped because: {result['termination_reason']}",
            f"Decoding method: {result['decoding_method']}",
            f"Neural constraints: {constraint_report}",
        ]
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface for XSum checkpoint inference."""
    parser = argparse.ArgumentParser(
        description="Generate an abstractive summary with a raw NumPy XSum model"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIRECTORY,
        help=f"Checkpoint directory (default: {DEFAULT_CHECKPOINT_DIRECTORY})",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--article", type=str, help="Article text to summarize")
    input_group.add_argument(
        "--article-file",
        type=Path,
        help="UTF-8 text file containing the article to summarize",
    )
    parser.add_argument(
        "--max-article-length",
        type=int,
        default=None,
        help="Override the checkpoint's maximum encoded article length",
    )
    parser.add_argument(
        "--max-summary-length",
        type=int,
        default=None,
        help="Override the checkpoint's maximum generated length",
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=3,
        help="Block a neural candidate that would repeat an n-gram (0 disables)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Penalize already generated neural tokens (1.0 disables)",
    )
    parser.add_argument(
        "--minimum-summary-length",
        type=int,
        default=3,
        help="Do not accept <EOS> before this many generated tokens",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optionally save the generation report to this file",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    vocabulary, encoder, decoder, metadata = load_checkpoint(args.checkpoint_dir)

    completed_epochs = int(
        metadata.get("epochs_completed", len(metadata.get("training_loss", [])))
    )
    if completed_epochs < 3:
        print(
            f"Warning: this checkpoint records only {completed_epochs} completed "
            "epoch(s); repeated generic text is likely until it is trained longer.",
            file=sys.stderr,
        )

    if args.article_file is not None:
        article = args.article_file.read_text(encoding="utf-8", errors="replace")
    elif args.article is not None:
        article = args.article
    else:
        article = input("Enter an article to summarize: ")
    if not article.strip():
        raise ValueError("article text must not be empty")

    max_summary_length = (
        args.max_summary_length
        if args.max_summary_length is not None
        else int(metadata["max_summary_length"])
    )
    max_article_length = (
        args.max_article_length
        if args.max_article_length is not None
        else int(metadata["max_article_length"])
    )
    result = generate_summary(
        article,
        vocabulary,
        encoder,
        decoder,
        max_article_length=max_article_length,
        max_summary_length=max_summary_length,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        minimum_summary_length=args.minimum_summary_length,
    )
    report = format_generation_report(result)
    print(report)

    if args.output_file is not None:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved inference report to: {args.output_file.resolve()}")


if __name__ == "__main__":
    main()
