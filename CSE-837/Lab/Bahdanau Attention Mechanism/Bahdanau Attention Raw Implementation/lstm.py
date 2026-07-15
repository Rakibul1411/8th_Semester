"""A complete one-layer LSTM with explicit forward and BPTT equations."""

from __future__ import annotations

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Return the elementwise sigmoid without overflow, preserving its derivative."""
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    non_negative = array >= 0.0
    # These equivalent branches avoid evaluating exp() on a large positive value.
    result[non_negative] = 1.0 / (1.0 + np.exp(-array[non_negative]))
    negative_exponential = np.exp(array[~non_negative])
    result[~non_negative] = negative_exponential / (1.0 + negative_exponential)
    return result


class LSTM:
    """Manual LSTM supporting individual steps and full-sequence BPTT."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize an LSTM mapping inputs ``(D,)`` to hidden states ``(H,)``."""
        if (
            isinstance(input_size, (bool, np.bool_))
            or not isinstance(input_size, (int, np.integer))
            or input_size <= 0
        ):
            raise ValueError("input_size must be a positive integer")
        if (
            isinstance(hidden_size, (bool, np.bool_))
            or not isinstance(hidden_size, (int, np.integer))
            or hidden_size <= 0
        ):
            raise ValueError("hidden_size must be a positive integer")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        rng = rng or np.random.default_rng()
        fan_in = input_size + hidden_size
        scale = np.sqrt(1.0 / max(1, fan_in))

        self.W_f = rng.normal(0.0, scale, (hidden_size, fan_in))
        self.W_i = rng.normal(0.0, scale, (hidden_size, fan_in))
        self.W_c = rng.normal(0.0, scale, (hidden_size, fan_in))
        self.W_o = rng.normal(0.0, scale, (hidden_size, fan_in))
        self.b_f = np.ones(hidden_size)  # Positive forget bias aids early training.
        self.b_i = np.zeros(hidden_size)
        self.b_c = np.zeros(hidden_size)
        self.b_o = np.zeros(hidden_size)

        for name in ("W_f", "W_i", "W_c", "W_o", "b_f", "b_i", "b_c", "b_o"):
            setattr(self, f"d{name}", np.zeros_like(getattr(self, name)))

    def step_forward(
        self, x_t: np.ndarray, h_previous: np.ndarray, c_previous: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Advance one step: ``(D,), (H,), (H,) -> (H,), (H,), cache``."""
        x_t = np.asarray(x_t, dtype=np.float64)
        h_previous = np.asarray(h_previous, dtype=np.float64)
        c_previous = np.asarray(c_previous, dtype=np.float64)
        if x_t.shape != (self.input_size,):
            raise ValueError(f"x_t must have shape ({self.input_size},)")
        if h_previous.shape != (self.hidden_size,):
            raise ValueError(f"h_previous must have shape ({self.hidden_size},)")
        if c_previous.shape != (self.hidden_size,):
            raise ValueError(f"c_previous must have shape ({self.hidden_size},)")
        # [h_(t-1), x_t]
        combined = np.concatenate((h_previous, x_t))

        # f_t = sigma(W_f [h_(t-1), x_t] + b_f)
        forget_gate = sigmoid(self.W_f @ combined + self.b_f)
        # i_t = sigma(W_i [h_(t-1), x_t] + b_i)
        input_gate = sigmoid(self.W_i @ combined + self.b_i)
        # C~_t = tanh(W_c [h_(t-1), x_t] + b_c)
        candidate_memory = np.tanh(self.W_c @ combined + self.b_c)
        # C_t = f_t * C_(t-1) + i_t * C~_t
        cell_state = forget_gate * c_previous + input_gate * candidate_memory
        # o_t = sigma(W_o [h_(t-1), x_t] + b_o)
        output_gate = sigmoid(self.W_o @ combined + self.b_o)
        # h_t = o_t * tanh(C_t)
        hidden_state = output_gate * np.tanh(cell_state)

        cache = {
            "combined": combined,
            "h_previous": h_previous,
            "c_previous": c_previous,
            "forget_gate": forget_gate,
            "input_gate": input_gate,
            "candidate_memory": candidate_memory,
            "output_gate": output_gate,
            "cell_state": cell_state,
        }
        return hidden_state, cell_state, cache

    def forward(
        self,
        inputs: np.ndarray,
        initial_hidden: np.ndarray | None = None,
        initial_cell: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, np.ndarray]]]:
        """Encode ``inputs (T,D)`` and return states ``(T,H)``, final states, caches."""
        sequence = np.asarray(inputs, dtype=np.float64)
        if sequence.ndim != 2 or sequence.shape[1] != self.input_size:
            raise ValueError(f"inputs must have shape (time, {self.input_size})")
        hidden = (
            np.zeros(self.hidden_size)
            if initial_hidden is None
            else np.asarray(initial_hidden, dtype=np.float64).copy()
        )
        cell = (
            np.zeros(self.hidden_size)
            if initial_cell is None
            else np.asarray(initial_cell, dtype=np.float64).copy()
        )
        if hidden.shape != (self.hidden_size,):
            raise ValueError(f"initial_hidden must have shape ({self.hidden_size},)")
        if cell.shape != (self.hidden_size,):
            raise ValueError(f"initial_cell must have shape ({self.hidden_size},)")
        hidden_states = np.empty((len(sequence), self.hidden_size))
        caches: list[dict[str, np.ndarray]] = []
        for time_step, x_t in enumerate(sequence):
            hidden, cell, cache = self.step_forward(x_t, hidden, cell)
            hidden_states[time_step] = hidden
            caches.append(cache)
        return hidden_states, hidden, cell, caches

    def step_backward(
        self,
        gradient_hidden: np.ndarray,
        gradient_cell_from_future: np.ndarray,
        cache: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backpropagate one step and return ``dx_t (D,), dh_prev (H,), dc_prev (H,)``."""
        gradient_hidden = np.asarray(gradient_hidden, dtype=np.float64)
        gradient_cell_from_future = np.asarray(
            gradient_cell_from_future, dtype=np.float64
        )
        if gradient_hidden.shape != (self.hidden_size,):
            raise ValueError(f"gradient_hidden must have shape ({self.hidden_size},)")
        if gradient_cell_from_future.shape != (self.hidden_size,):
            raise ValueError(
                f"gradient_cell_from_future must have shape ({self.hidden_size},)"
            )
        combined = cache["combined"]
        c_previous = cache["c_previous"]
        forget_gate = cache["forget_gate"]
        input_gate = cache["input_gate"]
        candidate_memory = cache["candidate_memory"]
        output_gate = cache["output_gate"]
        cell_state = cache["cell_state"]

        tanh_cell = np.tanh(cell_state)
        # dL/do_t = dL/dh_t * tanh(C_t)
        d_output_gate = gradient_hidden * tanh_cell
        # dL/dC_t = dL/dh_t * o_t * (1 - tanh(C_t)^2) + dL/dC_t|future
        d_cell = (
            gradient_hidden * output_gate * (1.0 - tanh_cell**2)
            + gradient_cell_from_future
        )
        # dL/df_t = dL/dC_t * C_(t-1)
        d_forget_gate = d_cell * c_previous
        # dL/di_t = dL/dC_t * C~_t
        d_input_gate = d_cell * candidate_memory
        # dL/dC~_t = dL/dC_t * i_t
        d_candidate = d_cell * input_gate
        # dL/dC_(t-1) = dL/dC_t * f_t
        d_cell_previous = d_cell * forget_gate

        # dL/dz_f = dL/df_t * f_t * (1 - f_t)
        dz_f = d_forget_gate * forget_gate * (1.0 - forget_gate)
        # dL/dz_i = dL/di_t * i_t * (1 - i_t)
        dz_i = d_input_gate * input_gate * (1.0 - input_gate)
        # dL/dz_c = dL/dC~_t * (1 - C~_t^2)
        dz_c = d_candidate * (1.0 - candidate_memory**2)
        # dL/dz_o = dL/do_t * o_t * (1 - o_t)
        dz_o = d_output_gate * output_gate * (1.0 - output_gate)

        # dL/dW_gate += dL/dz_gate outer [h_(t-1), x_t]
        self.dW_f += np.outer(dz_f, combined)
        self.dW_i += np.outer(dz_i, combined)
        self.dW_c += np.outer(dz_c, combined)
        self.dW_o += np.outer(dz_o, combined)
        # dL/db_gate += dL/dz_gate
        self.db_f += dz_f
        self.db_i += dz_i
        self.db_c += dz_c
        self.db_o += dz_o

        # dL/d[h_(t-1),x_t] = sum(W_gate^T dL/dz_gate)
        d_combined = (
            self.W_f.T @ dz_f
            + self.W_i.T @ dz_i
            + self.W_c.T @ dz_c
            + self.W_o.T @ dz_o
        )
        d_hidden_previous = d_combined[: self.hidden_size]
        d_input = d_combined[self.hidden_size :]
        return d_input, d_hidden_previous, d_cell_previous

    def backward(
        self,
        gradient_hidden_states: np.ndarray,
        caches: list[dict[str, np.ndarray]],
        gradient_final_hidden: np.ndarray | None = None,
        gradient_final_cell: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run BPTT: ``dhidden (T,H) -> dinputs (T,D), dh_0 (H,), dc_0 (H,)``."""
        gradients = np.asarray(gradient_hidden_states, dtype=np.float64)
        if gradients.shape != (len(caches), self.hidden_size):
            raise ValueError("gradient_hidden_states shape does not match the caches")
        gradient_inputs = np.zeros((len(caches), self.input_size))
        gradient_hidden_next = (
            np.zeros(self.hidden_size)
            if gradient_final_hidden is None
            else np.asarray(gradient_final_hidden, dtype=np.float64).copy()
        )
        gradient_cell_next = (
            np.zeros(self.hidden_size)
            if gradient_final_cell is None
            else np.asarray(gradient_final_cell, dtype=np.float64).copy()
        )
        if gradient_hidden_next.shape != (self.hidden_size,):
            raise ValueError(
                f"gradient_final_hidden must have shape ({self.hidden_size},)"
            )
        if gradient_cell_next.shape != (self.hidden_size,):
            raise ValueError(
                f"gradient_final_cell must have shape ({self.hidden_size},)"
            )

        # BPTT: dL/dh_t(total) = dL/dh_t(direct) + dL/dh_t(from t+1)
        for time_step in range(len(caches) - 1, -1, -1):
            gradient_total_hidden = gradients[time_step] + gradient_hidden_next
            gradient_inputs[time_step], gradient_hidden_next, gradient_cell_next = (
                self.step_backward(
                    gradient_total_hidden, gradient_cell_next, caches[time_step]
                )
            )
        return gradient_inputs, gradient_hidden_next, gradient_cell_next

    def named_parameters(self, prefix: str = "lstm") -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Return ``(name, parameter, accumulated_gradient)`` triples."""
        names = ("W_f", "W_i", "W_c", "W_o", "b_f", "b_i", "b_c", "b_o")
        return [
            (f"{prefix}.{name}", getattr(self, name), getattr(self, f"d{name}"))
            for name in names
        ]

    def zero_grad(self) -> None:
        """Reset all accumulated parameter gradients to zero."""
        for _, _, gradient in self.named_parameters():
            gradient.fill(0.0)
