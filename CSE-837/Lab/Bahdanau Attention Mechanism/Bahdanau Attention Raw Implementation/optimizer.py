"""Framework-free SGD/Adam optimizers with global-norm clipping."""

from __future__ import annotations

import numpy as np


class SGD:
    """Stochastic gradient descent over explicit NumPy parameter arrays."""

    def __init__(
        self,
        named_parameters: list[tuple[str, np.ndarray, np.ndarray]],
        learning_rate: float = 0.01,
        max_gradient_norm: float | None = 5.0,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if max_gradient_norm is not None and max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive or None")
        self.named_parameters = named_parameters
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm

    def gradient_norm(self) -> float:
        """Return the global L2 norm over every parameter-gradient tensor."""
        # ||g||_2 = sqrt(sum_parameters(sum_elements(g^2)))
        squared_norm = sum(float(np.sum(gradient**2)) for _, _, gradient in self.named_parameters)
        return float(np.sqrt(squared_norm))

    def step(self) -> float:
        """Clip the global gradient if needed and update parameters in place."""
        norm = self.gradient_norm()
        scale = 1.0
        if self.max_gradient_norm is not None and norm > self.max_gradient_norm:
            # g_clipped = g * max_norm / ||g||_2
            scale = self.max_gradient_norm / (norm + 1e-12)
        for _, parameter, gradient in self.named_parameters:
            # parameter = parameter - learning_rate * gradient
            parameter -= self.learning_rate * scale * gradient
        return norm


class Adam:
    """Manual Adam optimizer; no framework or autograd dependency."""

    def __init__(
        self,
        named_parameters: list[tuple[str, np.ndarray, np.ndarray]],
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_gradient_norm: float | None = 5.0,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 < beta1 < 1.0 or not 0.0 < beta2 < 1.0:
            raise ValueError("Adam beta values must be between 0 and 1")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if max_gradient_norm is not None and max_gradient_norm <= 0.0:
            raise ValueError("max_gradient_norm must be positive or None")
        self.named_parameters = named_parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.max_gradient_norm = max_gradient_norm
        self.time_step = 0
        self.first_moments = [np.zeros_like(parameter) for _, parameter, _ in named_parameters]
        self.second_moments = [np.zeros_like(parameter) for _, parameter, _ in named_parameters]

    def gradient_norm(self) -> float:
        """Return the global L2 norm before clipping."""
        squared_norm = sum(
            float(np.sum(gradient**2))
            for _, _, gradient in self.named_parameters
        )
        return float(np.sqrt(squared_norm))

    def step(self) -> float:
        """Apply one bias-corrected Adam update and return pre-clip norm."""
        norm = self.gradient_norm()
        scale = 1.0
        if self.max_gradient_norm is not None and norm > self.max_gradient_norm:
            scale = self.max_gradient_norm / (norm + 1e-12)

        self.time_step += 1
        first_correction = 1.0 - self.beta1**self.time_step
        second_correction = 1.0 - self.beta2**self.time_step
        for index, (_, parameter, gradient) in enumerate(self.named_parameters):
            clipped_gradient = gradient * scale
            first = self.first_moments[index]
            second = self.second_moments[index]
            # m_t = beta1*m_(t-1) + (1-beta1)*g_t
            first *= self.beta1
            first += (1.0 - self.beta1) * clipped_gradient
            # v_t = beta2*v_(t-1) + (1-beta2)*g_t^2
            second *= self.beta2
            second += (1.0 - self.beta2) * clipped_gradient**2
            first_hat = first / first_correction
            second_hat = second / second_correction
            # parameter -= learning_rate*m_hat/(sqrt(v_hat)+epsilon)
            parameter -= self.learning_rate * first_hat / (
                np.sqrt(second_hat) + self.epsilon
            )
        return norm
