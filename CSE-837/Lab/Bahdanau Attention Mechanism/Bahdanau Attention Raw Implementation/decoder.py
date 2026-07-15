"""Attention-driven LSTM decoder, output layer, and full manual backward pass."""

from __future__ import annotations

import numpy as np

from attention import BahdanauAttention, softmax
from embedding import Embedding
from lstm import LSTM


class Decoder:
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        attention_size: int,
        pad_id: int = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize a single-sequence attention decoder with hidden size ``H``."""
        dimensions = {
            "vocabulary_size": vocabulary_size,
            "embedding_dimension": embedding_dimension,
            "encoder_hidden_size": encoder_hidden_size,
            "decoder_hidden_size": decoder_hidden_size,
            "attention_size": attention_size,
        }
        for name, value in dimensions.items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
        if encoder_hidden_size != decoder_hidden_size:
            raise ValueError(
                "encoder_hidden_size must equal decoder_hidden_size because encoder "
                "final states directly initialize the decoder"
            )
        if (
            isinstance(pad_id, (bool, np.bool_))
            or not isinstance(pad_id, (int, np.integer))
            or not 0 <= pad_id < vocabulary_size
        ):
            raise ValueError("pad_id must be an integer in the vocabulary range")
        self.vocabulary_size = int(vocabulary_size)
        self.embedding_dimension = int(embedding_dimension)
        self.encoder_hidden_size = int(encoder_hidden_size)
        self.hidden_size = int(decoder_hidden_size)
        self.attention_size = int(attention_size)
        rng = rng or np.random.default_rng()

        self.embedding = Embedding(
            vocabulary_size, embedding_dimension, pad_id=pad_id, rng=rng
        )
        self.attention = BahdanauAttention(
            encoder_hidden_size, decoder_hidden_size, attention_size, rng=rng
        )
        self.lstm = LSTM(
            embedding_dimension + encoder_hidden_size, decoder_hidden_size, rng=rng
        )
        self.W_y = rng.normal(
            0.0,
            1.0 / np.sqrt(max(1, decoder_hidden_size)),
            (vocabulary_size, decoder_hidden_size),
        )
        # Give the word distribution a short, direct path from the attended
        # source.  Sending context only through the decoder LSTM makes it easy
        # for a small model to ignore the article and behave like a generic
        # language model.
        self.W_context = rng.normal(
            0.0,
            1.0 / np.sqrt(max(1, encoder_hidden_size)),
            (vocabulary_size, encoder_hidden_size),
        )
        self.b_y = np.zeros(vocabulary_size)
        self.dW_y = np.zeros_like(self.W_y)
        self.dW_context = np.zeros_like(self.W_context)
        self.db_y = np.zeros_like(self.b_y)

    def step_forward(
        self,
        previous_word_id: int,
        previous_hidden: np.ndarray,
        previous_cell: np.ndarray,
        encoder_hidden_states: np.ndarray,
    ) -> dict[str, object]:
        """Decode one token from ID/scalar states into logits and probabilities ``(V,)``."""
        if (
            isinstance(previous_word_id, (bool, np.bool_))
            or not isinstance(previous_word_id, (int, np.integer))
        ):
            raise ValueError("previous_word_id must be an integer")
        previous_word_id = int(previous_word_id)
        if not 0 <= previous_word_id < self.vocabulary_size:
            raise ValueError("previous_word_id is outside the vocabulary range")
        previous_hidden = np.asarray(previous_hidden, dtype=np.float64)
        previous_cell = np.asarray(previous_cell, dtype=np.float64)
        if previous_hidden.shape != (self.hidden_size,):
            raise ValueError(f"previous_hidden must have shape ({self.hidden_size},)")
        if previous_cell.shape != (self.hidden_size,):
            raise ValueError(f"previous_cell must have shape ({self.hidden_size},)")
        # x_t(word) = E_decoder[previous_word_id]
        word_embedding = self.embedding.forward(previous_word_id)
        context, attention_weights, energy_scores, attention_cache = (
            self.attention.forward(encoder_hidden_states, previous_hidden)
        )
        # x_t(decoder) = [word_embedding ; context_vector]
        decoder_input = np.concatenate((word_embedding, context))
        hidden, cell, lstm_cache = self.lstm.step_forward(
            decoder_input, previous_hidden, previous_cell
        )
        # y_t = W_y s_t + W_context c_t + b_y.  This is still a fully neural
        # abstractive distribution; it does not copy or fall back to source text.
        logits = self.W_y @ hidden + self.W_context @ context + self.b_y
        # P(word=k | word_<t, article) = exp(y_(t,k)) / sum_j(exp(y_(t,j)))
        probabilities = softmax(logits)
        return {
            "previous_word_id": int(previous_word_id),
            "word_embedding": word_embedding,
            "context": context,
            "attention_weights": attention_weights,
            "energy_scores": energy_scores,
            "attention_cache": attention_cache,
            "lstm_cache": lstm_cache,
            "hidden": hidden,
            "cell": cell,
            "logits": logits,
            "probabilities": probabilities,
        }

    def forward_teacher_forcing(
        self,
        decoder_input_ids: np.ndarray | list[int],
        encoder_hidden_states: np.ndarray,
        initial_hidden: np.ndarray,
        initial_cell: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
        """Decode reference inputs ``(T,)`` into logits/probabilities ``(T,V)`` and caches."""
        raw_ids = np.asarray(decoder_input_ids)
        if raw_ids.ndim != 1 or len(raw_ids) == 0:
            raise ValueError(
                "decoder_input_ids must be a non-empty one-dimensional sequence"
            )
        if not np.issubdtype(raw_ids.dtype, np.integer):
            raise ValueError("decoder_input_ids must contain integers")
        ids = raw_ids.astype(np.int64, copy=False)
        if np.any(ids < 0) or np.any(ids >= self.vocabulary_size):
            raise ValueError(
                "decoder_input_ids contain a token outside the vocabulary range"
            )
        hidden = np.asarray(initial_hidden, dtype=np.float64).copy()
        cell = np.asarray(initial_cell, dtype=np.float64).copy()
        if hidden.shape != (self.hidden_size,):
            raise ValueError(f"initial_hidden must have shape ({self.hidden_size},)")
        if cell.shape != (self.hidden_size,):
            raise ValueError(f"initial_cell must have shape ({self.hidden_size},)")
        encoder_states = np.asarray(encoder_hidden_states, dtype=np.float64)
        if (
            encoder_states.ndim != 2
            or encoder_states.shape[1] != self.encoder_hidden_size
            or len(encoder_states) == 0
        ):
            raise ValueError(
                "encoder_hidden_states must have shape "
                f"(nonzero_time, {self.encoder_hidden_size})"
            )
        logits = np.empty((len(ids), self.vocabulary_size))
        probabilities = np.empty((len(ids), self.vocabulary_size))
        step_caches: list[dict[str, object]] = []

        # Teacher forcing: decoder input_t = correct target word_(t-1).
        for time_step, previous_word_id in enumerate(ids):
            cache = self.step_forward(
                int(previous_word_id), hidden, cell, encoder_states
            )
            hidden = cache["hidden"]
            cell = cache["cell"]
            logits[time_step] = cache["logits"]
            probabilities[time_step] = cache["probabilities"]
            step_caches.append(cache)
        return logits, probabilities, step_caches

    def backward(
        self,
        gradient_logits: np.ndarray,
        step_caches: list[dict[str, object]],
        encoder_hidden_states: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backpropagate ``dlogits (T,V)`` to encoder states and decoder initial states."""
        gradients = np.asarray(gradient_logits, dtype=np.float64)
        if gradients.shape != (len(step_caches), self.vocabulary_size):
            raise ValueError("gradient_logits shape does not match decoder caches")

        encoder_states = np.asarray(encoder_hidden_states, dtype=np.float64)
        if (
            encoder_states.ndim != 2
            or encoder_states.shape[1] != self.encoder_hidden_size
            or len(encoder_states) == 0
        ):
            raise ValueError(
                "encoder_hidden_states must have shape "
                f"(nonzero_time, {self.encoder_hidden_size})"
            )
        gradient_encoder_states = np.zeros_like(encoder_states)
        gradient_hidden_next = np.zeros(self.hidden_size)
        gradient_cell_next = np.zeros(self.hidden_size)

        # BPTT path: loss -> output -> decoder -> attention -> encoder.
        for time_step in range(len(step_caches) - 1, -1, -1):
            cache = step_caches[time_step]
            d_logits = gradients[time_step]
            hidden = cache["hidden"]

            # dL/dW_y += dL/dy_t outer s_t
            self.dW_y += np.outer(d_logits, hidden)
            # dL/dW_context += dL/dy_t outer c_t
            self.dW_context += np.outer(d_logits, cache["context"])
            # dL/db_y += dL/dy_t
            self.db_y += d_logits
            # dL/ds_t = W_y^T dL/dy_t + dL/ds_t|future
            gradient_hidden = self.W_y.T @ d_logits + gradient_hidden_next

            gradient_decoder_input, gradient_hidden_previous, gradient_cell_previous = (
                self.lstm.step_backward(
                    gradient_hidden, gradient_cell_next, cache["lstm_cache"]
                )
            )
            gradient_word_embedding = gradient_decoder_input[
                : self.embedding_dimension
            ]
            gradient_context = (
                gradient_decoder_input[self.embedding_dimension :]
                + self.W_context.T @ d_logits
            )
            # dL/dE_decoder[word_(t-1)] += dL/dx_t(word)
            self.embedding.backward(cache["previous_word_id"], gradient_word_embedding)

            d_encoder_from_attention, d_query = self.attention.backward(
                gradient_context, cache["attention_cache"]
            )
            gradient_encoder_states += d_encoder_from_attention
            # s_(t-1) influences both the decoder recurrence and attention query.
            gradient_hidden_next = gradient_hidden_previous + d_query
            gradient_cell_next = gradient_cell_previous

        return gradient_encoder_states, gradient_hidden_next, gradient_cell_next

    def named_parameters(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Return decoder parameter and accumulated-gradient triples."""
        return (
            self.embedding.named_parameters("decoder.embedding")
            + self.attention.named_parameters("decoder.attention")
            + self.lstm.named_parameters("decoder.lstm")
            + [
                ("decoder.output.W_y", self.W_y, self.dW_y),
                ("decoder.output.W_context", self.W_context, self.dW_context),
                ("decoder.output.b_y", self.b_y, self.db_y),
            ]
        )

    def zero_grad(self) -> None:
        """Reset all accumulated decoder gradients to zero."""
        self.embedding.zero_grad()
        self.attention.zero_grad()
        self.lstm.zero_grad()
        self.dW_y.fill(0.0)
        self.dW_context.fill(0.0)
        self.db_y.fill(0.0)
