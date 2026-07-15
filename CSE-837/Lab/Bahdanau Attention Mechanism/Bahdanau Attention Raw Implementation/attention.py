"""Bahdanau additive attention with an explicit analytical backward pass."""

from __future__ import annotations

import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    """Stable one-dimensional softmax."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("softmax values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("softmax values must be finite")
    shifted = array - np.max(array)
    exponentials = np.exp(shifted)
    # alpha_(t,i) = exp(e_(t,i)) / sum_j(exp(e_(t,j)))
    return exponentials / np.sum(exponentials)


class BahdanauAttention:
    def __init__(
        self,
        encoder_hidden_size: int,
        decoder_hidden_size: int,
        attention_size: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize additive attention from ``(S,H_e)`` and ``(H_d,)`` inputs."""
        dimensions = {
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
        self.encoder_hidden_size = int(encoder_hidden_size)
        self.decoder_hidden_size = int(decoder_hidden_size)
        self.attention_size = int(attention_size)
        rng = rng or np.random.default_rng()

        self.W_h = rng.normal(
            0.0,
            1.0 / np.sqrt(max(1, encoder_hidden_size)),
            (attention_size, encoder_hidden_size),
        )
        self.W_s = rng.normal(
            0.0,
            1.0 / np.sqrt(max(1, decoder_hidden_size)),
            (attention_size, decoder_hidden_size),
        )
        # The paper denotes these trainable vectors as ``v`` and ``b``.  The
        # ``*_a`` aliases preserve the names used by older checkpoints/tests;
        # both names point to the same NumPy arrays and are updated once.
        self.v = rng.normal(
            0.0, 1.0 / np.sqrt(max(1, attention_size)), attention_size
        )
        self.b = np.zeros(attention_size)

        self.dW_h = np.zeros_like(self.W_h)
        self.dW_s = np.zeros_like(self.W_s)
        self.dv = np.zeros_like(self.v)
        self.db = np.zeros_like(self.b)
        self.v_a = self.v
        self.b_a = self.b
        self.dv_a = self.dv
        self.db_a = self.db

    def forward(
        self, encoder_hidden_states: np.ndarray, previous_decoder_hidden: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Return context ``(H_e,)``, weights/scores ``(S,)``, and a backward cache."""
        hidden_states = np.asarray(encoder_hidden_states, dtype=np.float64)
        query = np.asarray(previous_decoder_hidden, dtype=np.float64)
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.encoder_hidden_size:
            raise ValueError(
                f"encoder_hidden_states must have shape (time, {self.encoder_hidden_size})"
            )
        if len(hidden_states) == 0:
            raise ValueError("encoder_hidden_states must contain at least one time step")
        if query.shape != (self.decoder_hidden_size,):
            raise ValueError(
                f"previous_decoder_hidden must have shape ({self.decoder_hidden_size},)"
            )

        # u_(t,i) = tanh(W_h h_i + W_s s_(t-1) + b_a)
        attention_features = np.tanh(
            hidden_states @ self.W_h.T + query @ self.W_s.T + self.b
        )
        # e_(t,i) = v_a^T tanh(W_h h_i + W_s s_(t-1) + b_a)
        energy_scores = attention_features @ self.v
        # alpha_(t,i) = exp(e_(t,i)) / sum_j(exp(e_(t,j)))
        attention_weights = softmax(energy_scores)
        # c_t = sum_i(alpha_(t,i) * h_i)
        context_vector = attention_weights @ hidden_states

        cache = {
            "encoder_hidden_states": hidden_states,
            "previous_decoder_hidden": query,
            "attention_features": attention_features,
            "attention_weights": attention_weights,
        }
        return context_vector, attention_weights, energy_scores, cache

    def backward(
        self, gradient_context: np.ndarray, cache: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map ``dcontext (H_e,)`` to encoder states ``(S,H_e)`` and query ``(H_d,)``."""
        hidden_states = cache["encoder_hidden_states"]
        query = cache["previous_decoder_hidden"]
        features = cache["attention_features"]
        weights = cache["attention_weights"]
        d_context = np.asarray(gradient_context, dtype=np.float64)
        if d_context.shape != (self.encoder_hidden_size,):
            raise ValueError(
                f"gradient_context must have shape ({self.encoder_hidden_size},)"
            )

        # dc_t/dh_i (direct path) = alpha_(t,i) I
        d_hidden_states = weights[:, None] * d_context[None, :]
        # dL/dalpha_(t,i) = dL/dc_t dot h_i
        d_attention_weights = hidden_states @ d_context
        # dL/de_i = alpha_i * (dL/dalpha_i - sum_j(alpha_j*dL/dalpha_j))
        d_energy = weights * (
            d_attention_weights - np.dot(weights, d_attention_weights)
        )

        # dL/dv_a = sum_i(dL/de_i * u_(t,i))
        self.dv += features.T @ d_energy
        # dL/du_(t,i) = dL/de_i * v_a
        d_features = d_energy[:, None] * self.v[None, :]
        # dL/dz_(t,i) = dL/du_(t,i) * (1 - tanh(z_(t,i))^2)
        d_pre_activation = d_features * (1.0 - features**2)

        # dL/dW_h = sum_i(dL/dz_(t,i) outer h_i)
        self.dW_h += d_pre_activation.T @ hidden_states
        # dL/dW_s = sum_i(dL/dz_(t,i)) outer s_(t-1)
        self.dW_s += np.outer(np.sum(d_pre_activation, axis=0), query)
        # dL/db_a = sum_i(dL/dz_(t,i))
        self.db += np.sum(d_pre_activation, axis=0)

        # dL/dh_i += W_h^T dL/dz_(t,i)
        d_hidden_states += d_pre_activation @ self.W_h
        # dL/ds_(t-1) = sum_i(W_s^T dL/dz_(t,i))
        d_previous_decoder_hidden = np.sum(d_pre_activation, axis=0) @ self.W_s
        return d_hidden_states, d_previous_decoder_hidden

    def named_parameters(
        self, prefix: str = "attention"
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Return attention parameter and accumulated-gradient triples."""
        return [
            (f"{prefix}.W_h", self.W_h, self.dW_h),
            (f"{prefix}.W_s", self.W_s, self.dW_s),
            (f"{prefix}.v_a", self.v_a, self.dv_a),
            (f"{prefix}.b_a", self.b_a, self.db_a),
        ]

    def zero_grad(self) -> None:
        """Reset all accumulated attention gradients to zero."""
        for _, _, gradient in self.named_parameters():
            gradient.fill(0.0)
