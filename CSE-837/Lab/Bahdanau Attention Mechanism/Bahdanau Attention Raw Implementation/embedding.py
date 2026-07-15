"""Vocabulary and trainable NumPy embedding lookup table."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from preprocessing import tokenize


PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"
SPECIAL_TOKENS = (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN)


class Vocabulary:
    """Deterministic token/id mapping with four fixed special-token IDs.

    The reserved mapping is always ``PAD=0, SOS=1, EOS=2, UNK=3``. Corpus
    tokens are ordered first by descending frequency and then alphabetically,
    making independently repeated builds reproducible.
    """

    def __init__(self) -> None:
        self.word_to_index: dict[str, int] = {
            token: index for index, token in enumerate(SPECIAL_TOKENS)
        }
        self.index_to_word: dict[int, str] = {
            index: token for index, token in enumerate(SPECIAL_TOKENS)
        }
        self.counts: Counter[str] = Counter()

    @property
    def token_to_id(self) -> dict[str, int]:
        """Alias for the token-to-integer mapping."""
        return self.word_to_index

    @property
    def id_to_token(self) -> dict[int, str]:
        """Alias for the integer-to-token mapping."""
        return self.index_to_word

    @property
    def pad_id(self) -> int:
        return self.word_to_index[PAD_TOKEN]

    @property
    def sos_id(self) -> int:
        return self.word_to_index[SOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.word_to_index[EOS_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.word_to_index[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.word_to_index)

    def build(
        self,
        texts: Iterable[str],
        min_frequency: int = 1,
        max_size: int | None = None,
    ) -> "Vocabulary":
        """Count tokens and rebuild the mapping from training text only."""
        if isinstance(min_frequency, bool) or not isinstance(min_frequency, int):
            raise TypeError("min_frequency must be an integer")
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        if max_size is not None:
            if isinstance(max_size, bool) or not isinstance(max_size, int):
                raise TypeError("max_size must be an integer or None")
            if max_size < len(SPECIAL_TOKENS):
                raise ValueError(
                    f"max_size must be at least {len(SPECIAL_TOKENS)} to hold "
                    "the special tokens"
                )

        counts: Counter[str] = Counter()
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("every vocabulary input must be a string")
            counts.update(tokenize(text))
        self.counts = counts

        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        words = [word for word, count in ordered if count >= min_frequency]
        if max_size is not None:
            words = words[: max_size - len(SPECIAL_TOKENS)]

        all_tokens = list(SPECIAL_TOKENS) + words
        self.word_to_index = {
            token: index for index, token in enumerate(all_tokens)
        }
        self.index_to_word = {
            index: token for token, index in self.word_to_index.items()
        }
        return self

    def encode(
        self,
        text_or_tokens: str | list[str],
        add_sos: bool = False,
        add_eos: bool = False,
        max_length: int | None = None,
    ) -> list[int]:
        """Map text/tokens to IDs and optionally add decoder boundary tokens."""
        if max_length is not None:
            if isinstance(max_length, bool) or not isinstance(max_length, int):
                raise TypeError("max_length must be an integer or None")
            if max_length <= 0:
                raise ValueError("max_length must be positive")
            required_boundaries = int(add_sos) + int(add_eos)
            if max_length < required_boundaries:
                raise ValueError(
                    "max_length is too small for the requested boundary tokens"
                )

        if isinstance(text_or_tokens, str):
            tokens = tokenize(text_or_tokens)
        else:
            tokens = list(text_or_tokens)
            if any(not isinstance(token, str) for token in tokens):
                raise TypeError("all tokens must be strings")

        ids = [self.word_to_index.get(token, self.unk_id) for token in tokens]
        if add_sos:
            ids.insert(0, self.sos_id)
        if add_eos:
            ids.append(self.eos_id)
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            if add_eos:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Map IDs to whitespace-joined tokens, stopping at the first EOS."""
        words: list[str] = []
        for word_id in ids:
            try:
                index = int(word_id)
            except (TypeError, ValueError) as error:
                raise TypeError("vocabulary IDs must be integers") from error
            word = self.index_to_word.get(index, UNK_TOKEN)
            if word == EOS_TOKEN:
                break
            if skip_special and word in SPECIAL_TOKENS:
                continue
            words.append(word)
        return " ".join(words)

    def save(self, path: str | Path) -> None:
        """Persist the mapping and frequency counts as UTF-8 JSON."""
        payload = {
            "special_tokens": list(SPECIAL_TOKENS),
            "word_to_index": self.word_to_index,
            "counts": dict(self.counts),
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load a vocabulary and reject corrupt or incompatible mappings."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("vocabulary file must contain a JSON object")
        raw_mapping = payload.get("word_to_index")
        if not isinstance(raw_mapping, dict):
            raise ValueError("vocabulary file is missing word_to_index")

        mapping: dict[str, int] = {}
        for token, index in raw_mapping.items():
            if not isinstance(token, str):
                raise ValueError("vocabulary tokens must be strings")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("vocabulary IDs must be integers")
            mapping[token] = index

        expected_special_ids = {
            token: index for index, token in enumerate(SPECIAL_TOKENS)
        }
        if any(mapping.get(token) != index for token, index in expected_special_ids.items()):
            raise ValueError(
                "vocabulary special-token IDs must be PAD=0, SOS=1, EOS=2, UNK=3"
            )
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise ValueError("vocabulary IDs must be unique and contiguous from zero")

        raw_counts = payload.get("counts", {})
        if not isinstance(raw_counts, dict):
            raise ValueError("vocabulary counts must be a JSON object")
        counts: Counter[str] = Counter()
        for token, count in raw_counts.items():
            if not isinstance(token, str):
                raise ValueError("counted vocabulary tokens must be strings")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("vocabulary counts must be non-negative integers")
            counts[token] = count

        vocabulary = cls()
        vocabulary.word_to_index = mapping
        vocabulary.index_to_word = {
            index: token for token, index in mapping.items()
        }
        vocabulary.counts = counts
        return vocabulary


class Embedding:
    """Trainable lookup matrix ``E`` with shape ``(vocabulary_size, dimension)``."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        pad_id: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if isinstance(vocabulary_size, (bool, np.bool_)) or not isinstance(
            vocabulary_size, (int, np.integer)
        ):
            raise TypeError("vocabulary_size must be an integer")
        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        if isinstance(embedding_dimension, (bool, np.bool_)) or not isinstance(
            embedding_dimension, (int, np.integer)
        ):
            raise TypeError("embedding_dimension must be an integer")
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")
        if isinstance(pad_id, (bool, np.bool_)) or not isinstance(
            pad_id, (int, np.integer)
        ):
            raise TypeError("pad_id must be an integer")
        vocabulary_size = int(vocabulary_size)
        embedding_dimension = int(embedding_dimension)
        pad_id = int(pad_id)
        if not 0 <= pad_id < vocabulary_size:
            raise ValueError("pad_id must be within the vocabulary")

        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.pad_id = pad_id
        generator = np.random.default_rng() if rng is None else rng

        # E in R^(vocabulary_size x embedding_dimension).
        self.E = generator.normal(
            0.0,
            1.0 / np.sqrt(embedding_dimension),
            size=(vocabulary_size, embedding_dimension),
        )
        self.E[pad_id] = 0.0
        self.dE = np.zeros_like(self.E)
        # Descriptive aliases make the lookup-table role explicit while the
        # short names remain compatible with the optimizer/checkpoint format.
        self.embedding_matrix = self.E
        self.d_embedding_matrix = self.dE

    def _validate_ids(
        self, word_ids: np.ndarray | list[int] | tuple[int, ...] | int
    ) -> np.ndarray:
        raw_ids = np.asarray(word_ids)
        if raw_ids.size and raw_ids.dtype.kind not in {"i", "u"}:
            raise TypeError("word_ids must contain integers")
        if raw_ids.size and (
            np.any(raw_ids < 0) or np.any(raw_ids >= self.vocabulary_size)
        ):
            raise ValueError("word_ids must be within the vocabulary")
        return raw_ids.astype(np.int64, copy=False)

    def forward(
        self, word_ids: np.ndarray | list[int] | tuple[int, ...] | int
    ) -> np.ndarray:
        """Look up IDs with shape ``S`` and return embeddings of shape ``S + (D,)``."""
        ids = self._validate_ids(word_ids)
        # x_t = E[word_id]. NumPy indexing supports scalar, sequence, and batch IDs.
        return self.E[ids]

    def backward(
        self,
        word_ids: np.ndarray | list[int] | tuple[int, ...] | int,
        gradient_embeddings: np.ndarray,
    ) -> None:
        """Accumulate ``dL/dE`` from a gradient shaped ``ids.shape + (D,)``.

        Repeated IDs are accumulated with ``np.add.at``. This method computes
        gradients only; the shared SGD/Adam optimizer is the sole parameter-update
        path.
        """
        ids = self._validate_ids(word_ids)
        gradients = np.asarray(gradient_embeddings, dtype=np.float64)
        expected_shape = ids.shape + (self.embedding_dimension,)
        if gradients.shape != expected_shape:
            raise ValueError(
                f"gradient_embeddings must have shape {expected_shape}, "
                f"not {gradients.shape}"
            )

        # dL/dE[word_id] += dL/dx_t.
        np.add.at(self.dE, ids, gradients)
        self.dE[self.pad_id] = 0.0

    def named_parameters(
        self, prefix: str = "embedding"
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Expose ``(name, parameter, gradient)`` to the shared optimizer."""
        return [(f"{prefix}.E", self.E, self.dE)]

    def zero_grad(self) -> None:
        """Reset the accumulated embedding gradient before a new batch."""
        self.dE.fill(0.0)
