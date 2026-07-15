"""Train a raw-NumPy XSum encoder-decoder with Bahdanau attention.

The recurrent modules operate on one variable-length example at a time.  A
mini-batch is therefore implemented by accumulating each example's analytical
gradients, averaging them, and applying one optimizer update.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Callable, Iterable, Iterator

import numpy as np

from data_loader import display_dataset_summary, load_xsum_split
from decoder import Decoder
from embedding import Vocabulary
from encoder import Encoder
from evaluation import save_loss_curve_svg
from loss import CrossEntropyLoss
from optimizer import Adam, SGD
from preprocessing import prepare_summary_sequences


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_DATA_DIRECTORY = PROJECT_DIRECTORY
DEFAULT_CHECKPOINT_DIRECTORY = PROJECT_DIRECTORY / "checkpoints" / "xsum_model"


def build_models(
    vocabulary_size: int,
    embedding_dimension: int,
    hidden_size: int,
    attention_size: int,
    pad_id: int,
    seed: int = 42,
) -> tuple[Encoder, Decoder]:
    """Create equally sized encoder/decoder states using one seeded RNG."""
    rng = np.random.default_rng(seed)
    encoder = Encoder(
        vocabulary_size,
        embedding_dimension,
        hidden_size,
        pad_id=pad_id,
        rng=rng,
    )
    decoder = Decoder(
        vocabulary_size,
        embedding_dimension,
        hidden_size,
        hidden_size,
        attention_size,
        pad_id=pad_id,
        rng=rng,
    )
    return encoder, decoder


def all_named_parameters(
    encoder: Encoder, decoder: Decoder
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return every ``(name, parameter, gradient)`` triple in the model."""
    return encoder.named_parameters() + decoder.named_parameters()


def prepare_training_pair(
    sample: dict[str, str],
    vocabulary: Vocabulary,
    max_article_length: int,
    max_summary_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert one XSum record into source, teacher input, and target IDs.

    Shapes are ``(S,)``, ``(T,)``, and ``(T,)``.  Given summary words
    ``w1 ... wn``, decoder input is ``<SOS> w1 ... wn`` and the aligned target
    is ``w1 ... wn <EOS>``.  Truncation always preserves the final ``<EOS>``.
    """
    if max_article_length <= 0 or max_summary_length <= 0:
        raise ValueError("maximum sequence lengths must be positive")
    if "document" not in sample or "summary" not in sample:
        raise KeyError("each XSum sample must contain document and summary")

    article_ids = vocabulary.encode(
        sample["document"], add_eos=True, max_length=max_article_length
    )
    if not article_ids:
        raise ValueError("document produced no source IDs")
    decoder_input_ids, decoder_target_ids = prepare_summary_sequences(
        sample["summary"], vocabulary, max_summary_length
    )

    return (
        np.asarray(article_ids, dtype=np.int64),
        np.asarray(decoder_input_ids, dtype=np.int64),
        np.asarray(decoder_target_ids, dtype=np.int64),
    )


def forward_sample(
    article_ids: np.ndarray,
    decoder_input_ids: np.ndarray,
    decoder_target_ids: np.ndarray,
    encoder: Encoder,
    decoder: Decoder,
    criterion: CrossEntropyLoss,
) -> tuple[float, dict[str, object], list[dict[str, object]], np.ndarray]:
    """Run encoder, teacher-forced decoder, and stable cross-entropy."""
    encoder_cache = encoder.forward(article_ids)
    logits, probabilities, decoder_caches = decoder.forward_teacher_forcing(
        decoder_input_ids,
        encoder_cache["hidden_states"],
        encoder_cache["final_hidden"],
        encoder_cache["final_cell"],
    )
    loss_value = criterion.forward(
        logits, decoder_target_ids, from_logits=True
    )
    return loss_value, encoder_cache, decoder_caches, probabilities


def backward_sample(
    encoder: Encoder,
    decoder: Decoder,
    criterion: CrossEntropyLoss,
    encoder_cache: dict[str, object],
    decoder_caches: list[dict[str, object]],
) -> np.ndarray:
    """Backpropagate loss through output, decoder, attention, and encoder."""
    # softmax-cross-entropy: dL/dlogits = probabilities - one_hot(target)
    gradient_logits = criterion.backward()
    gradient_encoder_states, gradient_initial_hidden, gradient_initial_cell = (
        decoder.backward(
            gradient_logits,
            decoder_caches,
            encoder_cache["hidden_states"],
        )
    )
    # Decoder initial state is the encoder final state; attention also supplies
    # a direct gradient to every encoder hidden state.
    encoder.backward(
        gradient_encoder_states,
        encoder_cache,
        gradient_final_hidden=gradient_initial_hidden,
        gradient_final_cell=gradient_initial_cell,
    )
    return gradient_logits


def create_batches(
    samples: list[dict[str, str]],
    batch_size: int,
    random_generator: random.Random,
    shuffle: bool = True,
) -> Iterator[list[dict[str, str]]]:
    """Yield lists of examples; tensors stay unpadded and variable length."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = list(range(len(samples)))
    if shuffle:
        random_generator.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [samples[index] for index in indices[start : start + batch_size]]


def evaluate_teacher_forced_loss(
    samples: Iterable[dict[str, str]],
    vocabulary: Vocabulary,
    encoder: Encoder,
    decoder: Decoder,
    max_article_length: int,
    max_summary_length: int,
) -> float:
    """Return mean token-normalized teacher-forced loss on validation data."""
    losses: list[float] = []
    # Inference forbids <UNK>, so training must not reward predicting it.
    criterion = CrossEntropyLoss(
        vocabulary.pad_id, ignored_ids={vocabulary.unk_id}
    )
    for sample in samples:
        article_ids, decoder_inputs, targets = prepare_training_pair(
            sample, vocabulary, max_article_length, max_summary_length
        )
        loss_value, _, _, _ = forward_sample(
            article_ids, decoder_inputs, targets, encoder, decoder, criterion
        )
        losses.append(loss_value)
    return float(np.mean(losses)) if losses else 0.0


def greedy_decode_sample(
    document: str,
    vocabulary: Vocabulary,
    encoder: Encoder,
    decoder: Decoder,
    max_article_length: int = 256,
    max_summary_length: int = 40,
) -> str:
    """Generate the same constrained-neural preview used by inference."""
    # Local import avoids coupling checkpoint loading into the training module.
    from inference import generate_summary

    return str(
        generate_summary(
            document,
            vocabulary,
            encoder,
            decoder,
            max_article_length=max_article_length,
            max_summary_length=max_summary_length,
        )["summary"]
    )


def _assert_finite_gradients(
    named_parameters: list[tuple[str, np.ndarray, np.ndarray]],
) -> None:
    for name, _, gradient in named_parameters:
        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError(f"non-finite gradient detected in {name}")


def _print_first_sample_trace(
    article_ids: np.ndarray,
    encoder_cache: dict[str, object],
    decoder_caches: list[dict[str, object]],
    loss_value: float,
    gradient_logits: np.ndarray,
) -> None:
    """Print compact values proving the requested forward/backward path ran."""
    first_step = decoder_caches[0]
    weights = np.asarray(first_step["attention_weights"])
    print("\n========== FIRST SAMPLE TRACE ==========")
    print("source IDs shape:", article_ids.shape)
    print("encoder states shape:", np.asarray(encoder_cache["hidden_states"]).shape)
    print("attention weights shape:", weights.shape, "sum:", float(weights.sum()))
    print("context shape:", np.asarray(first_step["context"]).shape)
    print("loss:", loss_value)
    print("dL/dlogits shape:", gradient_logits.shape)
    print("========================================\n")


def train_model(
    training_samples: list[dict[str, str]],
    validation_samples: list[dict[str, str]] | None,
    vocabulary: Vocabulary,
    encoder: Encoder,
    decoder: Decoder,
    epochs: int = 8,
    learning_rate: float = 0.001,
    max_article_length: int = 256,
    max_summary_length: int = 40,
    gradient_clip: float = 5.0,
    seed: int = 42,
    debug_first_sample: bool = False,
    optimizer_name: str = "adam",
    batch_size: int = 4,
    log_every: int = 100,
    early_stopping_patience: int | None = 3,
    epoch_callback: Callable[[int, list[float], list[float]], None] | None = None,
) -> tuple[list[float], list[float]]:
    """Train with teacher forcing and complete manually coded BPTT.

    The defaults are deliberately longer than the original one-epoch/120-token
    smoke configuration: XSum article and summary length audits show that the
    old source cap discarded most of each document, while one epoch left the
    decoder at a high-frequency language-model solution.
    """
    if not training_samples:
        raise ValueError("training_samples must not be empty")
    if epochs <= 0 or batch_size <= 0 or log_every <= 0:
        raise ValueError("epochs, batch_size, and log_every must be positive")
    if optimizer_name not in {"adam", "sgd"}:
        raise ValueError("optimizer_name must be 'adam' or 'sgd'")
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive or None")

    parameters = all_named_parameters(encoder, decoder)
    optimizer_class = Adam if optimizer_name == "adam" else SGD
    optimizer = optimizer_class(
        parameters,
        learning_rate=learning_rate,
        max_gradient_norm=gradient_clip,
    )
    criterion = CrossEntropyLoss(
        vocabulary.pad_id, ignored_ids={vocabulary.unk_id}
    )
    random_generator = random.Random(seed)
    training_history: list[float] = []
    validation_history: list[float] = []
    trace_pending = debug_first_sample
    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        sample_losses: list[float] = []
        processed = 0

        for batch in create_batches(
            training_samples, batch_size, random_generator, shuffle=True
        ):
            encoder.zero_grad()
            decoder.zero_grad()

            for sample in batch:
                article_ids, decoder_inputs, targets = prepare_training_pair(
                    sample, vocabulary, max_article_length, max_summary_length
                )
                loss_value, encoder_cache, decoder_caches, _ = forward_sample(
                    article_ids,
                    decoder_inputs,
                    targets,
                    encoder,
                    decoder,
                    criterion,
                )
                if not np.isfinite(loss_value):
                    raise FloatingPointError("non-finite training loss detected")
                gradient_logits = backward_sample(
                    encoder, decoder, criterion, encoder_cache, decoder_caches
                )
                if trace_pending:
                    _print_first_sample_trace(
                        article_ids,
                        encoder_cache,
                        decoder_caches,
                        loss_value,
                        gradient_logits,
                    )
                    trace_pending = False
                sample_losses.append(loss_value)

            # Each example loss is token-normalized.  Average accumulated
            # gradients so update magnitude does not grow with batch size.
            for _, _, gradient in parameters:
                gradient /= len(batch)
            _assert_finite_gradients(parameters)
            gradient_norm = optimizer.step()
            if not np.isfinite(gradient_norm):
                raise FloatingPointError("non-finite global gradient norm detected")

            processed += len(batch)
            if processed % log_every < len(batch) or processed == len(training_samples):
                print(
                    f"Epoch {epoch}/{epochs} - sample {processed}/{len(training_samples)} "
                    f"- running loss {np.mean(sample_losses):.4f}",
                    flush=True,
                )

        training_loss = float(np.mean(sample_losses))
        training_history.append(training_loss)
        elapsed = time.perf_counter() - started

        if validation_samples:
            validation_loss = evaluate_teacher_forced_loss(
                validation_samples,
                vocabulary,
                encoder,
                decoder,
                max_article_length,
                max_summary_length,
            )
            validation_history.append(validation_loss)
            print(
                f"Epoch {epoch}/{epochs} complete - train loss {training_loss:.4f} "
                f"- validation loss {validation_loss:.4f} - {elapsed:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch}/{epochs} complete - train loss {training_loss:.4f} "
                f"- {elapsed:.1f}s"
            )

        generated = greedy_decode_sample(
            training_samples[0]["document"],
            vocabulary,
            encoder,
            decoder,
            max_article_length,
            max_summary_length,
        )
        print(f"  Generated: {generated or '[EOS immediately]'}")
        print(f"  Reference: {training_samples[0]['summary'][:160]}")

        if epoch_callback is not None:
            epoch_callback(
                epoch, training_history.copy(), validation_history.copy()
            )

        if validation_samples and early_stopping_patience is not None:
            if validation_history[-1] < best_validation_loss - 1e-4:
                best_validation_loss = validation_history[-1]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    print(
                        "Early stopping: validation loss did not improve for "
                        f"{early_stopping_patience} epoch(s)."
                    )
                    break

    return training_history, validation_history


def save_checkpoint(
    checkpoint_directory: str | Path,
    vocabulary: Vocabulary,
    encoder: Encoder,
    decoder: Decoder,
    metadata: dict[str, object],
) -> None:
    """Save vocabulary, dimensions, and all trainable NumPy arrays."""
    directory = Path(checkpoint_directory)
    directory.mkdir(parents=True, exist_ok=True)
    vocabulary.save(directory / "vocabulary.json")
    np.savez_compressed(
        directory / "encoder.npz",
        **{name: parameter for name, parameter, _ in encoder.named_parameters()},
    )
    np.savez_compressed(
        directory / "decoder.npz",
        **{name: parameter for name, parameter, _ in decoder.named_parameters()},
    )
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def summary_unknown_rate(
    samples: Iterable[dict[str, str]], vocabulary: Vocabulary
) -> tuple[int, int, float]:
    """Return unknown/total reference-summary tokens and the corresponding rate."""
    unknown = 0
    total = 0
    for sample in samples:
        token_ids = vocabulary.encode(sample["summary"])
        unknown += sum(token_id == vocabulary.unk_id for token_id in token_ids)
        total += len(token_ids)
    return unknown, total, (unknown / total if total else 0.0)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train raw NumPy Seq2Seq LSTM + Bahdanau attention on XSum"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIRECTORY
    )
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--strict-data", action="store_true")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--attention-size", type=int, default=48)
    parser.add_argument("--vocabulary-size", type=int, default=120000)
    parser.add_argument("--minimum-frequency", type=int, default=2)
    parser.add_argument("--max-article-length", type=int, default=256)
    parser.add_argument("--max-summary-length", type=int, default=40)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--learning-rate",
        type=float,
        help=(
            "Defaults to 0.001 for fresh Adam, 0.0003 for Adam warm starts, "
            "and 0.01 for SGD"
        ),
    )
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop after this many non-improving validation epochs; 0 disables",
    )
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help=(
            "Load model/vocabulary weights from a checkpoint and start a fresh "
            "optimizer (useful for the existing one-epoch checkpoint)"
        ),
    )
    parser.add_argument("--debug-first-sample", action="store_true")
    parser.add_argument("--show-dataset", action="store_true")
    parser.add_argument(
        "--overfit-test",
        action="store_true",
        help="Use at most eight training records and no validation",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.early_stopping_patience < 0:
        raise ValueError("early-stopping-patience must be non-negative")
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else (
            (0.0003 if args.warm_start is not None else 0.001)
            if args.optimizer == "adam"
            else 0.01
        )
    )

    train_path = args.data_dir / "xsum_train.json"
    validation_path = args.data_dir / "xsum_validation.json"
    training_samples = load_xsum_split(
        train_path, limit=args.train_limit, strict=args.strict_data
    )
    validation_samples = (
        []
        if args.skip_validation
        else load_xsum_split(
            validation_path,
            limit=args.validation_limit,
            strict=args.strict_data,
        )
    )
    if args.overfit_test:
        training_samples = training_samples[:8]
        validation_samples = []
        print(f"Overfit test: using {len(training_samples)} XSum training records")
    if not training_samples:
        raise ValueError("no valid XSum training records were loaded")

    if args.show_dataset:
        display_dataset_summary(training_samples)
    print(
        f"Loaded XSum: {len(training_samples)} train / "
        f"{len(validation_samples)} validation"
    )

    previous_metadata: dict[str, object] = {}
    if args.warm_start is not None:
        from inference import load_checkpoint

        vocabulary, encoder, decoder, previous_metadata = load_checkpoint(
            args.warm_start
        )
        print(
            f"Warm-started {args.warm_start.resolve()} with a fresh "
            f"{args.optimizer.upper()} optimizer at learning rate {learning_rate:g}"
        )
    else:
        # Validation and test text never contribute to vocabulary statistics.
        vocabulary = Vocabulary().build(
            (
                text
                for sample in training_samples
                for text in (sample["document"], sample["summary"])
            ),
            min_frequency=args.minimum_frequency,
            max_size=args.vocabulary_size,
        )
        encoder, decoder = build_models(
            len(vocabulary),
            args.embedding_dim,
            args.hidden_size,
            args.attention_size,
            vocabulary.pad_id,
            seed=args.seed,
        )
    print(f"Vocabulary: {len(vocabulary)} tokens")
    unknown_tokens, summary_tokens, unknown_rate = summary_unknown_rate(
        training_samples, vocabulary
    )
    print(
        f"Reference-summary <UNK> rate: {unknown_tokens}/{summary_tokens} "
        f"({100.0 * unknown_rate:.2f}%); those targets are excluded from loss"
    )

    previous_epochs = int(
        previous_metadata.get(
            "epochs_completed", len(previous_metadata.get("training_loss", []))
        )
    )
    best_validation: dict[str, float] = {"loss": float("inf")}

    def make_metadata(
        training_history: list[float],
        validation_history: list[float],
        checkpoint_role: str,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "format_version": 3,
            "dataset": "xsum",
            "dataset_files": {
                "train": "xsum_train.json",
                "validation": "xsum_validation.json",
                "test": "xsum_test.json",
            },
            "architecture": "lstm_bahdanau_direct_context_output",
            "embedding_dimension": encoder.embedding_dimension,
            "hidden_size": encoder.hidden_size,
            "attention_size": decoder.attention_size,
            "max_article_length": args.max_article_length,
            "max_summary_length": args.max_summary_length,
            "vocabulary_size": len(vocabulary),
            "minimum_frequency": int(
                previous_metadata.get(
                    "minimum_frequency", args.minimum_frequency
                )
            ),
            "seed": int(previous_metadata.get("seed", args.seed)),
            "optimizer": args.optimizer,
            "learning_rate": learning_rate,
            "batch_size": args.batch_size,
            "gradient_clip": args.gradient_clip,
            "training_samples": len(training_samples),
            "validation_samples": len(validation_samples),
            "epochs_completed": previous_epochs + len(training_history),
            "current_run_epochs": len(training_history),
            "training_loss": training_history,
            "validation_loss": validation_history,
            "loss_ignored_tokens": ["<PAD>", "<UNK>"],
            "summary_unknown_tokens": unknown_tokens,
            "summary_token_count": summary_tokens,
            "summary_unknown_rate": unknown_rate,
            "checkpoint_role": checkpoint_role,
        }
        if args.warm_start is not None:
            metadata["warm_started_from"] = str(args.warm_start.resolve())
            metadata["previous_checkpoint_epochs"] = previous_epochs
            metadata["optimizer_state_restored"] = False
        if np.isfinite(best_validation["loss"]):
            metadata["best_validation_loss"] = best_validation["loss"]
        return metadata

    def save_completed_epoch(
        epoch: int,
        training_history: list[float],
        validation_history: list[float],
    ) -> None:
        if validation_history:
            latest_directory = args.checkpoint_dir / "last"
            save_checkpoint(
                latest_directory,
                vocabulary,
                encoder,
                decoder,
                make_metadata(training_history, validation_history, "latest"),
            )
            save_loss_curve_svg(
                training_history,
                validation_history,
                latest_directory / "training_loss.svg",
            )
            if validation_history[-1] < best_validation["loss"] - 1e-4:
                best_validation["loss"] = validation_history[-1]
                save_checkpoint(
                    args.checkpoint_dir,
                    vocabulary,
                    encoder,
                    decoder,
                    make_metadata(training_history, validation_history, "best"),
                )
                save_loss_curve_svg(
                    training_history,
                    validation_history,
                    args.checkpoint_dir / "training_loss.svg",
                )
                print(
                    f"  Saved new best checkpoint (validation "
                    f"{validation_history[-1]:.4f})"
                )
            else:
                print(f"  Saved completed epoch {epoch} to checkpoints/last")
        else:
            save_checkpoint(
                args.checkpoint_dir,
                vocabulary,
                encoder,
                decoder,
                make_metadata(training_history, validation_history, "latest"),
            )
            save_loss_curve_svg(
                training_history,
                validation_history,
                args.checkpoint_dir / "training_loss.svg",
            )
            print(f"  Saved completed epoch {epoch}")

    training_history, validation_history = train_model(
        training_samples,
        validation_samples or None,
        vocabulary,
        encoder,
        decoder,
        epochs=args.epochs,
        learning_rate=learning_rate,
        max_article_length=args.max_article_length,
        max_summary_length=args.max_summary_length,
        gradient_clip=args.gradient_clip,
        seed=args.seed,
        debug_first_sample=args.debug_first_sample,
        optimizer_name=args.optimizer,
        batch_size=args.batch_size,
        log_every=args.log_every,
        early_stopping_patience=(
            args.early_stopping_patience
            if args.early_stopping_patience > 0
            else None
        ),
        epoch_callback=save_completed_epoch,
    )
    if validation_history:
        print(f"Best XSum checkpoint: {args.checkpoint_dir.resolve()}")
        print(f"Latest XSum checkpoint: {(args.checkpoint_dir / 'last').resolve()}")
    else:
        print(f"Saved XSum checkpoint to {args.checkpoint_dir.resolve()}")


if __name__ == "__main__":
    main()
