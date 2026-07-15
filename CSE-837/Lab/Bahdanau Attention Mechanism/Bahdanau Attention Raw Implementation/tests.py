"""Dependency-free correctness checks for the complete raw-NumPy XSum pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import random
import tempfile
import warnings

import numpy as np

from data_loader import load_xsum_split, load_xsum_splits
from embedding import (
    EOS_TOKEN,
    PAD_TOKEN,
    SOS_TOKEN,
    UNK_TOKEN,
    Embedding,
    Vocabulary,
)
from evaluation import calculate_rouge, save_attention_heatmap_svg, save_loss_curve_svg
from inference import (
    _blocked_ngram_completions,
    format_generation_report,
    generate_summary,
    load_checkpoint,
)
from loss import CrossEntropyLoss
from optimizer import Adam, SGD
from preprocessing import tokenize
from train import (
    backward_sample,
    build_models,
    create_batches,
    forward_sample,
    prepare_training_pair,
    save_checkpoint,
    train_model,
)


PROJECT_DIRECTORY = Path(__file__).resolve().parent
ARTICLE_IDS = np.asarray([4, 5, 2], dtype=np.int64)
DECODER_INPUTS = np.asarray([1, 6], dtype=np.int64)
TARGETS = np.asarray([6, 2], dtype=np.int64)


def _assert_raises(error_type: type[BaseException], function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except error_type:
        return
    raise AssertionError(f"expected {error_type.__name__} from {function.__name__}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _loss(encoder, decoder) -> float:
    criterion = CrossEntropyLoss(pad_id=0)
    value, _, _, _ = forward_sample(
        ARTICLE_IDS, DECODER_INPUTS, TARGETS, encoder, decoder, criterion
    )
    return value


def _numeric_gradient(
    encoder,
    decoder,
    parameter: np.ndarray,
    index: tuple[int, ...],
) -> float:
    epsilon = 1e-5
    original = float(parameter[index])
    parameter[index] = original + epsilon
    positive = _loss(encoder, decoder)
    parameter[index] = original - epsilon
    negative = _loss(encoder, decoder)
    parameter[index] = original
    return (positive - negative) / (2.0 * epsilon)


def test_gradients() -> tuple[object, object]:
    """Compare representative gradients along every major path to finite differences."""
    encoder, decoder = build_models(8, 3, 4, 3, pad_id=0, seed=7)
    criterion = CrossEntropyLoss(pad_id=0)
    encoder.zero_grad()
    decoder.zero_grad()
    _, encoder_cache, decoder_caches, _ = forward_sample(
        ARTICLE_IDS, DECODER_INPUTS, TARGETS, encoder, decoder, criterion
    )
    backward_sample(encoder, decoder, criterion, encoder_cache, decoder_caches)

    checks = [
        ("output.W_y", decoder.W_y, decoder.dW_y, (1, 2)),
        (
            "output.W_context",
            decoder.W_context,
            decoder.dW_context,
            (1, 2),
        ),
        ("attention.W_h", decoder.attention.W_h, decoder.attention.dW_h, (0, 1)),
        ("attention.W_s", decoder.attention.W_s, decoder.attention.dW_s, (1, 2)),
        ("attention.v", decoder.attention.v_a, decoder.attention.dv_a, (1,)),
        ("decoder.lstm.W_i", decoder.lstm.W_i, decoder.lstm.dW_i, (0, 1)),
        ("encoder.lstm.W_f", encoder.lstm.W_f, encoder.lstm.dW_f, (0, 1)),
        ("encoder.embedding.E", encoder.embedding.E, encoder.embedding.dE, (4, 1)),
        ("decoder.embedding.E", decoder.embedding.E, decoder.embedding.dE, (1, 1)),
    ]
    for name, parameter, analytical_gradient, index in checks:
        analytical = float(analytical_gradient[index])
        numerical = _numeric_gradient(encoder, decoder, parameter, index)
        denominator = max(1e-8, abs(analytical) + abs(numerical))
        relative_error = abs(analytical - numerical) / denominator
        absolute_error = abs(analytical - numerical)
        assert absolute_error < 1e-7 or relative_error < 2e-4, (
            f"{name}{index}: analytical={analytical}, numerical={numerical}, "
            f"absolute error={absolute_error}, relative error={relative_error}"
        )
    return encoder, decoder


def test_xsum_loader_tokenizer_vocabulary_and_pairs() -> None:
    records = [
        {
            "document": "Müller's café costs £20 after a 10% rise.",
            "summary": "Café prices rose.",
            "id": "valid-1",
            "ignored": "discard me",
        },
        {"document": "", "summary": "Invalid source.", "id": "empty-source"},
        {"document": "Words exist.", "summary": ".", "id": "empty-target"},
        {
            "document": "A company announced new investment.",
            "summary": "Company announced investment.",
            "id": "valid-2",
        },
    ]
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        train_path = directory / "xsum_train.json"
        _write_json(train_path, records)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = load_xsum_split(train_path)
        assert [sample["id"] for sample in loaded] == ["valid-1", "valid-2"]
        assert set(loaded[0]) == {"document", "summary", "id"}
        assert len(caught) == 1 and "empty-source" in str(caught[0].message)
        _assert_raises(ValueError, load_xsum_split, train_path, strict=True)

        valid_only = [records[0], records[3]]
        for filename in (
            "xsum_train.json",
            "xsum_validation.json",
            "xsum_test.json",
        ):
            _write_json(directory / filename, valid_only)
        train, validation, test = load_xsum_splits(
            directory, train_limit=1, validation_limit=1, test_limit=1
        )
        assert len(train) == len(validation) == len(test) == 1

        vocabulary = Vocabulary().build(
            [loaded[0]["document"], loaded[0]["summary"]]
        )
        assert (
            vocabulary.pad_id,
            vocabulary.sos_id,
            vocabulary.eos_id,
            vocabulary.unk_id,
        ) == (0, 1, 2, 3)
        assert [
            vocabulary.index_to_word[index] for index in range(4)
        ] == [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        tokens = tokenize(records[0]["document"])
        assert "müller's" in tokens and "£" in tokens and "%" in tokens

        source, decoder_input, target = prepare_training_pair(
            loaded[0], vocabulary, max_article_length=20, max_summary_length=8
        )
        assert source[-1] == vocabulary.eos_id
        assert decoder_input[0] == vocabulary.sos_id
        assert target[-1] == vocabulary.eos_id
        assert len(decoder_input) == len(target)

        vocabulary.save(directory / "vocabulary.json")
        restored = Vocabulary.load(directory / "vocabulary.json")
        assert restored.word_to_index == vocabulary.word_to_index


def test_supplied_xsum_files() -> None:
    """Exercise all three real split artifacts without loading them into training."""
    for filename in (
        "xsum_train.json",
        "xsum_validation.json",
        "xsum_test.json",
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            samples = load_xsum_split(PROJECT_DIRECTORY / filename, limit=2)
        assert len(samples) == 2
        assert set(samples[0]) == {"document", "summary", "id"}


def test_embedding_loss_and_optimizers() -> None:
    embedding = Embedding(7, 3, pad_id=0, rng=np.random.default_rng(4))
    ids = np.asarray([4, 4, 5])
    embedding.backward(ids, np.ones((3, 3)))
    assert np.allclose(embedding.dE[4], 2.0)
    assert np.allclose(embedding.dE[5], 1.0)
    assert np.all(embedding.dE[0] == 0.0)
    _assert_raises(ValueError, embedding.forward, [-1])

    # Stable log-sum-exp must remain finite for extreme logits. Target 2 is PAD.
    criterion = CrossEntropyLoss(pad_id=2)
    logits = np.asarray([[1000.0, -1000.0, 0.0], [0.0, 0.0, 0.0]])
    loss = criterion.forward(logits, np.asarray([0, 2]))
    gradient = criterion.backward()
    assert np.isfinite(loss) and loss < 1e-10
    assert np.all(np.isfinite(gradient)) and np.all(gradient[1] == 0.0)
    probability_loss = CrossEntropyLoss(pad_id=2).forward(
        np.asarray([[0.1, 0.7, 0.2]]), np.asarray([1]), from_logits=False
    )
    assert abs(probability_loss + np.log(0.7)) < 1e-12
    ignored_unknown = CrossEntropyLoss(pad_id=0, ignored_ids={3})
    ignored_unknown.forward(
        np.zeros((2, 5)), np.asarray([3, 2]), from_logits=True
    )
    ignored_gradient = ignored_unknown.backward()
    assert np.all(ignored_gradient[0] == 0.0)
    assert not np.all(ignored_gradient[1] == 0.0)

    adam_parameter = np.asarray([1.0, -1.0])
    adam_gradient = np.asarray([0.5, -0.25])
    Adam(
        [("parameter", adam_parameter, adam_gradient)], learning_rate=0.01
    ).step()
    assert np.allclose(adam_parameter, [0.99, -0.99], atol=1e-8)

    sgd_parameter = np.asarray([1.0, 1.0])
    sgd_gradient = np.asarray([3.0, 4.0])
    norm = SGD(
        [("parameter", sgd_parameter, sgd_gradient)],
        learning_rate=0.1,
        max_gradient_norm=1.0,
    ).step()
    assert abs(norm - 5.0) < 1e-12
    assert np.allclose(sgd_parameter, [0.94, 0.92])


def test_checkpoint_inference_metrics_and_visualizations(encoder, decoder) -> None:
    vocabulary = Vocabulary().build(["alpha beta gamma delta"])
    assert len(vocabulary) == 8
    metadata = {
        "format_version": 3,
        "dataset": "xsum",
        "embedding_dimension": 3,
        "hidden_size": 4,
        "attention_size": 3,
        "max_article_length": 5,
        "max_summary_length": 3,
        "seed": 7,
    }
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        save_checkpoint(directory, vocabulary, encoder, decoder, metadata)
        loaded_vocabulary, loaded_encoder, loaded_decoder, loaded_metadata = (
            load_checkpoint(directory)
        )
        result = generate_summary(
            "alpha beta",
            loaded_vocabulary,
            loaded_encoder,
            loaded_decoder,
            max_article_length=int(loaded_metadata["max_article_length"]),
            max_summary_length=int(loaded_metadata["max_summary_length"]),
        )
        assert result["attention_weights"].shape == (
            len(result["generated_words"]),
            len(result["source_words"]),
        )
        assert result["summary"] == " ".join(result["generated_words"])
        assert result["decoding_method"] == "greedy"
        assert result["decoding_constraints"]["no_repeat_ngram_size"] == 3
        report = format_generation_report(result)
        assert "Generated summary:" in report
        assert "extractive" not in report.lower()
        assert "fallback" not in report.lower()

        escaped = generate_summary(
            "alpha\\nbeta",
            loaded_vocabulary,
            loaded_encoder,
            loaded_decoder,
            max_article_length=5,
            max_summary_length=3,
        )
        actual = generate_summary(
            "alpha\nbeta",
            loaded_vocabulary,
            loaded_encoder,
            loaded_decoder,
            max_article_length=5,
            max_summary_length=3,
        )
        assert escaped["source_words"] == actual["source_words"]

        assert _blocked_ngram_completions([4, 5, 4, 5], 2) == {4}
        assert _blocked_ngram_completions([4, 5, 6, 4, 5], 3) == {6}

        # A format-v2 decoder archive has no direct-context matrix. Loading it
        # must zero only that new compatible parameter, not reject the model.
        decoder_path = directory / "decoder.npz"
        with np.load(decoder_path, allow_pickle=False) as archive:
            legacy_arrays = {
                name: archive[name]
                for name in archive.files
                if name != "decoder.output.W_context"
            }
        np.savez_compressed(decoder_path, **legacy_arrays)
        _, _, legacy_decoder, _ = load_checkpoint(directory)
        assert np.all(legacy_decoder.W_context == 0.0)

        save_loss_curve_svg([2.0, 1.5], [2.1, 1.7], directory / "loss.svg")
        save_attention_heatmap_svg(
            result["attention_weights"],
            result["source_words"],
            result["generated_words"],
            directory / "attention.svg",
        )
        assert (directory / "loss.svg").is_file()
        assert (directory / "attention.svg").is_file()

        metadata["dataset"] = "legacy"
        (directory / "metadata.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        _assert_raises(ValueError, load_checkpoint, directory)

    scores = calculate_rouge("the cat sat", "the cat sat")
    assert all(abs(metric["f1"] - 1.0) < 1e-12 for metric in scores.values())


def test_end_to_end_xsum_training_smoke() -> None:
    samples = [
        {
            "document": "Company announces a new solar investment.",
            "summary": "Company invests in solar.",
            "id": "one",
        },
        {
            "document": "Council opens a new city library.",
            "summary": "New library opens.",
            "id": "two",
        },
        {
            "document": "Team wins the final after extra time.",
            "summary": "Team wins final.",
            "id": "three",
        },
    ]
    vocabulary = Vocabulary().build(
        text for sample in samples for text in (sample["document"], sample["summary"])
    )
    encoder, decoder = build_models(
        len(vocabulary), 4, 6, 5, vocabulary.pad_id, seed=11
    )
    before = decoder.W_y.copy()
    batches = list(create_batches(samples, 2, random.Random(3), shuffle=False))
    assert [len(batch) for batch in batches] == [2, 1]
    training_history, validation_history = train_model(
        samples,
        samples[:1],
        vocabulary,
        encoder,
        decoder,
        epochs=1,
        learning_rate=0.01,
        max_article_length=10,
        max_summary_length=6,
        batch_size=2,
        optimizer_name="adam",
        log_every=100,
        seed=3,
    )
    assert len(training_history) == len(validation_history) == 1
    assert np.isfinite(training_history[0]) and np.isfinite(validation_history[0])
    assert not np.array_equal(before, decoder.W_y)

    with tempfile.TemporaryDirectory() as temporary_directory:
        metadata = {
            "format_version": 3,
            "dataset": "xsum",
            "embedding_dimension": 4,
            "hidden_size": 6,
            "attention_size": 5,
            "max_article_length": 10,
            "max_summary_length": 6,
            "seed": 11,
        }
        save_checkpoint(
            temporary_directory, vocabulary, encoder, decoder, metadata
        )
        loaded_vocabulary, loaded_encoder, loaded_decoder, loaded_metadata = (
            load_checkpoint(temporary_directory)
        )
        result = generate_summary(
            samples[0]["document"],
            loaded_vocabulary,
            loaded_encoder,
            loaded_decoder,
            max_article_length=int(loaded_metadata["max_article_length"]),
            max_summary_length=int(loaded_metadata["max_summary_length"]),
        )
        scores = calculate_rouge(result["summary"], samples[0]["summary"])
        assert set(scores) == {"rouge-1", "rouge-2", "rouge-l"}


def test_no_forbidden_or_legacy_code_paths() -> None:
    python_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in PROJECT_DIRECTORY.glob("*.py")
        if path.name != "tests.py"
    ).lower()
    forbidden_imports = (
        "import torch",
        "from torch",
        "import tensorflow",
        "from tensorflow",
        "import keras",
        "from keras",
        "import sklearn",
        "from sklearn",
        "import transformers",
        "from transformers",
        "from datasets",
    )
    assert not any(item in python_sources for item in forbidden_imports)
    assert "category_split" not in python_sources
    assert "sentence_rank" not in python_sources
    assert "fallback_summary" not in python_sources
    assert not (PROJECT_DIRECTORY / "download_xsum.py").exists()


def main() -> None:
    encoder, decoder = test_gradients()
    test_xsum_loader_tokenizer_vocabulary_and_pairs()
    test_supplied_xsum_files()
    test_embedding_loss_and_optimizers()
    test_checkpoint_inference_metrics_and_visualizations(encoder, decoder)
    test_end_to_end_xsum_training_smoke()
    test_no_forbidden_or_legacy_code_paths()
    print("All raw-NumPy XSum tests passed, including finite-difference gradients.")


if __name__ == "__main__":
    main()
