"""Numerically stable token-level cross-entropy computed directly from logits."""

from __future__ import annotations

import numpy as np


class CrossEntropyLoss:
    def __init__(self, pad_id: int = 0, ignored_ids: set[int] | None = None) -> None:
        """Configure ignored target IDs; padding is ignored by default."""
        candidate_ids = {pad_id, *(ignored_ids or set())}
        if any(
            isinstance(token_id, (bool, np.bool_))
            or not isinstance(token_id, (int, np.integer))
            or token_id < 0
            for token_id in candidate_ids
        ):
            raise ValueError("pad_id and ignored_ids must be non-negative integers")
        self.pad_id = int(pad_id)
        self.ignored_ids = {int(token_id) for token_id in candidate_ids}
        self._probabilities: np.ndarray | None = None
        self._targets: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._normalizer = 1

    def forward(
        self,
        logits: np.ndarray,
        target_ids: np.ndarray | list[int],
        *,
        from_logits: bool | None = None,
    ) -> float:
        """Return mean cross-entropy for score rows ``(T,V)`` and targets ``(T,)``.

        Training passes raw logits (the numerically stable default).  For
        compatibility with small educational callers, rows that are explicitly
        identified as probabilities can be used with ``from_logits=False``;
        when omitted, probability-like non-negative rows summing to one are
        recognized conservatively.
        """
        scores = np.asarray(logits, dtype=np.float64)
        raw_targets = np.asarray(target_ids)
        if scores.ndim != 2 or scores.shape[1] == 0:
            raise ValueError("logits must have shape (time, nonzero_vocabulary)")
        if raw_targets.shape != (scores.shape[0],):
            raise ValueError("target_ids must have shape (time,)")
        if not np.issubdtype(raw_targets.dtype, np.integer):
            raise ValueError("target_ids must contain integers")
        if not np.all(np.isfinite(scores)):
            raise ValueError("logits must be finite")
        targets = raw_targets.astype(np.int64, copy=False)
        vocabulary_size = scores.shape[1]
        if np.any(targets < 0) or np.any(targets >= vocabulary_size):
            raise ValueError("target_ids contain a token outside the vocabulary range")
        if any(token_id >= vocabulary_size for token_id in self.ignored_ids):
            raise ValueError("an ignored token ID is outside the vocabulary range")

        if from_logits is not None and not isinstance(from_logits, bool):
            raise TypeError("from_logits must be True, False, or None")
        if from_logits is None:
            from_logits = not (
                np.all(scores >= 0.0)
                and np.allclose(np.sum(scores, axis=1), 1.0, atol=1e-7)
            )

        # Structural/OOV targets can be excluded from both loss and gradients.
        mask = ~np.isin(targets, list(self.ignored_ids))
        self._normalizer = max(1, int(np.sum(mask)))
        if from_logits:
            # logsumexp(y_t) - y_t[target] is stable even for very large logits.
            row_maximum = np.max(scores, axis=1, keepdims=True)
            shifted_exponentials = np.exp(scores - row_maximum)
            exponential_sums = np.sum(shifted_exponentials, axis=1, keepdims=True)
            log_normalizers = np.log(exponential_sums[:, 0]) + row_maximum[:, 0]
            token_losses = log_normalizers - scores[
                np.arange(len(targets)), targets
            ]
            probabilities = shifted_exponentials / exponential_sums
        else:
            probabilities = np.clip(scores, 1e-12, 1.0)
            token_losses = -np.log(
                probabilities[np.arange(len(targets)), targets]
            )
        loss = np.sum(token_losses[mask])
        self._probabilities = probabilities
        self._targets = targets
        self._mask = mask
        return float(loss / self._normalizer)

    def backward(self) -> np.ndarray:
        """Return the cached loss gradient with respect to logits, shape ``(T,V)``."""
        if self._probabilities is None or self._targets is None or self._mask is None:
            raise RuntimeError("forward must be called before backward")
        # For softmax + cross entropy: dL/dy_t = P_t - one_hot(correct_word_t)
        gradient_logits = self._probabilities.copy()
        gradient_logits[np.arange(len(self._targets)), self._targets] -= 1.0
        # Padding tokens contribute no loss or gradient.
        gradient_logits[~self._mask] = 0.0
        gradient_logits /= self._normalizer
        return gradient_logits
