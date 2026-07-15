"""Embedding plus LSTM encoder and the corresponding backward flow."""

from __future__ import annotations

import numpy as np

from embedding import Embedding
from lstm import LSTM


class Encoder:
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        hidden_size: int,
        pad_id: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize an encoder with token IDs ``(S,)`` and states ``(S,H)``."""
        dimensions = {
            "vocabulary_size": vocabulary_size,
            "embedding_dimension": embedding_dimension,
            "hidden_size": hidden_size,
        }
        for name, value in dimensions.items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(pad_id, (bool, np.bool_))
            or not isinstance(pad_id, (int, np.integer))
            or not 0 <= pad_id < vocabulary_size
        ):
            raise ValueError("pad_id must be an integer in the vocabulary range")
        self.vocabulary_size = int(vocabulary_size)
        self.embedding_dimension = int(embedding_dimension)
        self.hidden_size = int(hidden_size)
        rng = rng or np.random.default_rng()
        self.embedding = Embedding(
            vocabulary_size, embedding_dimension, pad_id=pad_id, rng=rng
        )
        self.lstm = LSTM(embedding_dimension, hidden_size, rng=rng)

    def forward(self, article_ids: np.ndarray | list[int]) -> dict[str, object]:
        """Map source IDs ``(S,)`` to embeddings ``(S,E)`` and states ``(S,H)``."""
        raw_ids = np.asarray(article_ids)
        if raw_ids.ndim != 1 or len(raw_ids) == 0:
            raise ValueError("article_ids must be a non-empty one-dimensional sequence")
        if not np.issubdtype(raw_ids.dtype, np.integer):
            raise ValueError("article_ids must contain integers")
        ids = raw_ids.astype(np.int64, copy=False)
        if np.any(ids < 0) or np.any(ids >= self.vocabulary_size):
            raise ValueError("article_ids contain a token outside the vocabulary range")
        # x_i = E_encoder[article_id_i]
        embeddings = self.embedding.forward(ids)
        hidden_states, final_hidden, final_cell, lstm_caches = self.lstm.forward(
            embeddings
        )
        return {
            "article_ids": ids,
            "embeddings": embeddings,
            "hidden_states": hidden_states,
            "final_hidden": final_hidden,
            "final_cell": final_cell,
            "lstm_caches": lstm_caches,
        }

    def backward(
        self,
        gradient_hidden_states: np.ndarray,
        cache: dict[str, object],
        gradient_final_hidden: np.ndarray | None = None,
        gradient_final_cell: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Backpropagate ``dhidden (S,H)`` into source embeddings and initial states."""
        gradients = np.asarray(gradient_hidden_states, dtype=np.float64)
        article_ids = np.asarray(cache["article_ids"], dtype=np.int64)
        expected_shape = (len(article_ids), self.hidden_size)
        if gradients.shape != expected_shape:
            raise ValueError(
                f"gradient_hidden_states must have shape {expected_shape}"
            )
        gradient_embeddings, gradient_initial_hidden, gradient_initial_cell = (
            self.lstm.backward(
                gradients,
                cache["lstm_caches"],
                gradient_final_hidden=gradient_final_hidden,
                gradient_final_cell=gradient_final_cell,
            )
        )
        # dL/dE_encoder[article_id_i] += dL/dx_i
        self.embedding.backward(cache["article_ids"], gradient_embeddings)
        return gradient_initial_hidden, gradient_initial_cell

    def named_parameters(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Return encoder parameter and accumulated-gradient triples."""
        return self.embedding.named_parameters("encoder.embedding") + self.lstm.named_parameters(
            "encoder.lstm"
        )

    def zero_grad(self) -> None:
        """Reset all accumulated encoder gradients to zero."""
        self.embedding.zero_grad()
        self.lstm.zero_grad()
