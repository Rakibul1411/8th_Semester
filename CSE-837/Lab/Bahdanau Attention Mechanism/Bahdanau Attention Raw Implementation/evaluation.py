"""ROUGE metrics and dependency-free SVG visualizations."""

from __future__ import annotations

import argparse
from collections import Counter
from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np

from data_loader import load_xsum_split
from preprocessing import tokenize


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_DATA_DIRECTORY = SCRIPT_DIRECTORY
DEFAULT_CHECKPOINT_DIRECTORY = SCRIPT_DIRECTORY / "checkpoints" / "xsum_model"


def _f1(overlap: int, predicted_count: int, reference_count: int) -> dict[str, float]:
    precision = overlap / predicted_count if predicted_count else 0.0
    recall = overlap / reference_count if reference_count else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def rouge_n(prediction: str, reference: str, n: int) -> dict[str, float]:
    """Calculate count-aware ROUGE-N precision, recall, and F1."""
    if n <= 0:
        raise ValueError("n must be positive")
    predicted_ngrams = _ngrams(tokenize(prediction), n)
    reference_ngrams = _ngrams(tokenize(reference), n)
    # overlap_n = sum_g(min(count_pred(g), count_ref(g)))
    overlap = sum((predicted_ngrams & reference_ngrams).values())
    return _f1(overlap, sum(predicted_ngrams.values()), sum(reference_ngrams.values()))


def _lcs_length(first: list[str], second: list[str]) -> int:
    """Memory-efficient dynamic programming for longest common subsequence."""
    previous = [0] * (len(second) + 1)
    # LCS(i,j) = LCS(i-1,j-1)+1 if x_i=y_j else max(LCS(i-1,j),LCS(i,j-1))
    for first_token in first:
        current = [0]
        for index, second_token in enumerate(second, start=1):
            if first_token == second_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> dict[str, float]:
    predicted_tokens = tokenize(prediction)
    reference_tokens = tokenize(reference)
    lcs = _lcs_length(predicted_tokens, reference_tokens)
    return _f1(lcs, len(predicted_tokens), len(reference_tokens))


def calculate_rouge(prediction: str, reference: str) -> dict[str, dict[str, float]]:
    return {
        "rouge-1": rouge_n(prediction, reference, 1),
        "rouge-2": rouge_n(prediction, reference, 2),
        "rouge-l": rouge_l(prediction, reference),
    }


def average_rouge(
    scored_examples: Iterable[dict[str, dict[str, float]]]
) -> dict[str, dict[str, float]]:
    scores = list(scored_examples)
    if not scores:
        return {
            name: {metric: 0.0 for metric in ("precision", "recall", "f1")}
            for name in ("rouge-1", "rouge-2", "rouge-l")
        }
    return {
        name: {
            metric: float(np.mean([score[name][metric] for score in scores]))
            for metric in ("precision", "recall", "f1")
        }
        for name in ("rouge-1", "rouge-2", "rouge-l")
    }


def save_loss_curve_svg(
    training_losses: list[float],
    validation_losses: list[float],
    output_path: str | Path,
) -> None:
    """Write a loss plot as SVG using only Python strings and arithmetic."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 440
    left, right, top, bottom = 70, 30, 35, 60
    plot_width, plot_height = width - left - right, height - top - bottom
    all_values = training_losses + validation_losses
    maximum = max(all_values, default=1.0)
    minimum = min(all_values, default=0.0)
    if maximum == minimum:
        maximum = minimum + 1.0
    epochs = max(len(training_losses), len(validation_losses), 1)

    def points(values: list[float]) -> str:
        coordinates = []
        for index, value in enumerate(values):
            x = left + (index / max(1, epochs - 1)) * plot_width
            y = top + ((maximum - value) / (maximum - minimum)) * plot_height
            coordinates.append(f"{x:.2f},{y:.2f}")
        return " ".join(coordinates)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">Training loss</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#222"/>',
        f'<text x="18" y="{top+plot_height/2}" transform="rotate(-90 18 {top+plot_height/2})" text-anchor="middle" font-family="sans-serif">Cross-entropy</text>',
        f'<text x="{left+plot_width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif">Epoch</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        y = top + fraction * plot_height
        value = maximum - fraction * (maximum - minimum)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_width}" y2="{y:.2f}" stroke="#ddd"/>')
        elements.append(f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.3f}</text>')
    if training_losses:
        elements.append(f'<polyline points="{points(training_losses)}" fill="none" stroke="#2563eb" stroke-width="3"/>')
    if validation_losses:
        elements.append(f'<polyline points="{points(validation_losses)}" fill="none" stroke="#dc2626" stroke-width="3"/>')
    elements.extend([
        f'<line x1="{width-205}" y1="22" x2="{width-180}" y2="22" stroke="#2563eb" stroke-width="3"/><text x="{width-174}" y="26" font-family="sans-serif" font-size="12">train</text>',
        f'<line x1="{width-115}" y1="22" x2="{width-90}" y2="22" stroke="#dc2626" stroke-width="3"/><text x="{width-84}" y="26" font-family="sans-serif" font-size="12">validation</text>',
        '</svg>',
    ])
    path.write_text("\n".join(elements), encoding="utf-8")


def save_attention_heatmap_svg(
    attention_weights: np.ndarray,
    source_words: list[str],
    generated_words: list[str],
    output_path: str | Path,
) -> None:
    """Write generated-word × source-word attention weights as an SVG heatmap."""
    matrix = np.asarray(attention_weights, dtype=np.float64)
    if matrix.shape != (len(generated_words), len(source_words)):
        raise ValueError("attention matrix shape must match generated and source words")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cell_width, cell_height = 38, 30
    left, top, right, bottom = 150, 135, 30, 35
    width = left + cell_width * max(1, len(source_words)) + right
    height = top + cell_height * max(1, len(generated_words)) + bottom
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="18">Bahdanau attention</text>',
    ]
    for column, word in enumerate(source_words):
        x = left + column * cell_width + cell_width / 2
        elements.append(
            f'<text x="{x:.1f}" y="{top-8}" transform="rotate(-60 {x:.1f} {top-8})" '
            f'text-anchor="start" font-family="sans-serif" font-size="11">{escape(word)}</text>'
        )
    for row, word in enumerate(generated_words):
        y = top + row * cell_height
        elements.append(
            f'<text x="{left-8}" y="{y+cell_height*0.68:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{escape(word)}</text>'
        )
        for column in range(len(source_words)):
            value = float(np.clip(matrix[row, column], 0.0, 1.0))
            red = int(239 - 190 * value)
            green = int(246 - 120 * value)
            blue = int(255 - 20 * value)
            x = left + column * cell_width
            elements.append(
                f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" '
                f'fill="rgb({red},{green},{blue})" stroke="white"><title>{value:.6f}</title></rect>'
            )
    elements.append('</svg>')
    path.write_text("\n".join(elements), encoding="utf-8")


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the XSum test-set evaluation command-line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate raw NumPy XSum summaries using manual ROUGE"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIRECTORY,
        help=f"Checkpoint directory (default: {DEFAULT_CHECKPOINT_DIRECTORY})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIRECTORY,
        help="Directory containing xsum_test.json",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Optionally evaluate only the first N XSum test examples",
    )
    parser.add_argument(
        "--heatmap",
        type=Path,
        default=None,
        help="Optionally save the first example's attention heatmap as SVG",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.samples is not None and args.samples <= 0:
        raise ValueError("--samples must be positive")

    # Local import keeps the metric/plot functions independently reusable by train.py.
    from inference import generate_summary, load_checkpoint

    vocabulary, encoder, decoder, metadata = load_checkpoint(args.checkpoint_dir)
    test_path = args.data_dir / "xsum_test.json"
    samples = load_xsum_split(test_path, limit=args.samples)
    print(f"Evaluating {len(samples)} examples from {test_path.resolve()}.")

    scores: list[dict[str, dict[str, float]]] = []
    first_result: dict[str, object] | None = None
    for sample in samples:
        result = generate_summary(
            sample["document"],
            vocabulary,
            encoder,
            decoder,
            max_article_length=int(metadata["max_article_length"]),
            max_summary_length=int(metadata["max_summary_length"]),
        )
        scores.append(calculate_rouge(result["summary"], sample["summary"]))
        if first_result is None:
            first_result = result
            print("Document:\n", sample["document"])
            print("\nReference summary:\n", sample["summary"])
            print("\nGenerated summary:\n", result["summary"])

    averaged = average_rouge(scores)
    print("\nAverage ROUGE scores:")
    for metric_name, metric_scores in averaged.items():
        print(
            f"{metric_name}: precision={metric_scores['precision']:.4f}, "
            f"recall={metric_scores['recall']:.4f}, f1={metric_scores['f1']:.4f}"
        )
    if first_result is not None and args.heatmap is not None:
        save_attention_heatmap_svg(
            first_result["attention_weights"],
            first_result["source_words"],
            first_result["generated_words"],
            args.heatmap,
        )
        print(f"Attention heatmap saved to {args.heatmap.resolve()}")


if __name__ == "__main__":
    main()
