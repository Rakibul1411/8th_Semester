import numpy as np


class RawLSTMNextNumberPredictor:

    def __init__(
        self,
        input_size=1,
        hidden_size=16,
        output_size=1,
        learning_rate=0.01,
        epochs=400,
        seed=42,
        scale=250.0
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.scale = scale

        self.concat_size = self.input_size + self.hidden_size

        if seed is not None:
            np.random.seed(seed)

        self.rng = np.random.default_rng(seed)

        self.params = self.initialize_parameters()

        self.X = None
        self.Y = None

    # 1. Helper functions
    def sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, y):
        return y * (1 - y)

    def dtanh(self, y):
        return 1 - y * y

    # 2. Generalized dataset creation
    def create_dataset(self, num_samples=5000):
        X = []
        Y = []

        # Model will see different sequence lengths
        sequence_lengths = [3, 4, 5, 6]

        # Model will see both increasing and decreasing patterns
        steps = [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1,
                  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]

        for _ in range(num_samples):
            seq_len = int(self.rng.choice(sequence_lengths))
            step = int(self.rng.choice(steps))
            start = int(self.rng.integers(-150, 151))

            sequence = []

            for k in range(seq_len):
                sequence.append(start + k * step)

            target = start + seq_len * step

            # Normalize input and target
            X.append(np.array(sequence, dtype=float) / self.scale)
            Y.append(target / self.scale)

        self.X = X
        self.Y = np.array(Y, dtype=float).reshape(-1, 1)

        print("Dataset created.")
        print("Total samples:", len(self.X))
        print("Example input:", self.X[0])
        print("Example target:", self.Y[0])
        print("Variable sequence lengths included:", sequence_lengths)
        print("Steps included:", steps)

    # 3. Parameter initialization
    def initialize_parameters(self):
        params = {}

        # Forget gate parameters
        params["Wf"] = np.random.randn(self.hidden_size, self.concat_size) * 0.1

        # Positive forget bias helps LSTM remember at the beginning
        params["bf"] = np.ones((self.hidden_size, 1)) * 1.0

        # Input gate parameters
        params["Wi"] = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        params["bi"] = np.zeros((self.hidden_size, 1))

        # Candidate cell state parameters
        params["Wc"] = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        params["bc"] = np.zeros((self.hidden_size, 1))

        # Output gate parameters
        params["Wo"] = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        params["bo"] = np.zeros((self.hidden_size, 1))

        # Final output layer parameters
        params["Wy"] = np.random.randn(self.output_size, self.hidden_size) * 0.1
        params["by"] = np.zeros((self.output_size, 1))

        return params

    # 4. Forward pass
    def forward(self, sequence):
        h_prev = np.zeros((self.hidden_size, 1))
        c_prev = np.zeros((self.hidden_size, 1))

        caches = []

        for value in sequence:
            x_t = np.array([[value]])

            combined = np.vstack((h_prev, x_t))

            # Forget gate
            f_t = self.sigmoid(self.params["Wf"] @ combined + self.params["bf"])

            # Input gate
            i_t = self.sigmoid(self.params["Wi"] @ combined + self.params["bi"])

            # Candidate memory
            c_candidate = np.tanh(self.params["Wc"] @ combined + self.params["bc"])

            # Cell state update
            c_t = f_t * c_prev + i_t * c_candidate

            # Output gate
            o_t = self.sigmoid(self.params["Wo"] @ combined + self.params["bo"])

            # Hidden state
            h_t = o_t * np.tanh(c_t)

            cache = {
                "combined": combined,
                "f_t": f_t,
                "i_t": i_t,
                "c_candidate": c_candidate,
                "c_t": c_t,
                "c_prev": c_prev,
                "o_t": o_t,
                "h_t": h_t,
                "h_prev": h_prev,
                "x_t": x_t
            }

            caches.append(cache)

            h_prev = h_t
            c_prev = c_t

        y_hat = self.params["Wy"] @ h_prev + self.params["by"]

        return y_hat, caches

    # 5. Loss function
    def compute_loss(self, y_hat, y_true):
        return 0.5 * np.mean((y_hat - y_true) ** 2)

    # 6. Backward pass / BPTT
    def backward(self, y_hat, y_true, caches):
        grads = {}

        for key in self.params:
            grads[key] = np.zeros_like(self.params[key])

        dy = y_hat - y_true

        last_h = caches[-1]["h_t"]

        grads["Wy"] += dy @ last_h.T
        grads["by"] += dy

        dh_next = self.params["Wy"].T @ dy
        dc_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(caches))):
            cache = caches[t]

            combined = cache["combined"]
            f_t = cache["f_t"]
            i_t = cache["i_t"]
            c_candidate = cache["c_candidate"]
            c_t = cache["c_t"]
            c_prev = cache["c_prev"]
            o_t = cache["o_t"]

            dh = dh_next

            tanh_c = np.tanh(c_t)

            # Output gate gradient
            do = dh * tanh_c
            do_pre = do * self.dsigmoid(o_t)

            # Cell state gradient
            dc = dh * o_t * (1 - tanh_c ** 2) + dc_next

            # Forget gate gradient
            df = dc * c_prev
            df_pre = df * self.dsigmoid(f_t)

            # Input gate gradient
            di = dc * c_candidate
            di_pre = di * self.dsigmoid(i_t)

            # Candidate memory gradient
            dcandidate = dc * i_t
            dcandidate_pre = dcandidate * self.dtanh(c_candidate)

            # Parameter gradients
            grads["Wf"] += df_pre @ combined.T
            grads["bf"] += df_pre

            grads["Wi"] += di_pre @ combined.T
            grads["bi"] += di_pre

            grads["Wc"] += dcandidate_pre @ combined.T
            grads["bc"] += dcandidate_pre

            grads["Wo"] += do_pre @ combined.T
            grads["bo"] += do_pre

            # Gradient wrt combined = [h_prev, x_t]
            dcombined = (
                self.params["Wf"].T @ df_pre +
                self.params["Wi"].T @ di_pre +
                self.params["Wc"].T @ dcandidate_pre +
                self.params["Wo"].T @ do_pre
            )

            dh_next = dcombined[:self.hidden_size, :]
            dc_next = dc * f_t

        return grads

    # 7. Parameter update
    def update_parameters(self, grads):
        for key in grads:
            grads[key] = np.clip(grads[key], -1, 1)

        for key in self.params:
            self.params[key] -= self.learning_rate * grads[key]

    # 8. Training
    def train(self):
        if self.X is None or self.Y is None:
            raise ValueError("Dataset not created. Call create_dataset() first.")

        for epoch in range(self.epochs):
            total_loss = 0.0

            indices = np.random.permutation(len(self.X))

            for idx in indices:
                sequence = self.X[idx]
                y_true = self.Y[idx].reshape(1, 1)

                y_hat, caches = self.forward(sequence)

                loss = self.compute_loss(y_hat, y_true)
                total_loss += loss

                grads = self.backward(y_hat, y_true, caches)

                self.update_parameters(grads)

            if (epoch + 1) % 50 == 0:
                avg_loss = total_loss / len(self.X)
                print(f"Epoch {epoch + 1}, Loss: {float(avg_loss):.8f}")

    # 9. Prediction using only LSTM, no rule-based fallback
    def predict_next_number(self, seq):
        seq = np.array(seq, dtype=float) / self.scale
        y_hat, _ = self.forward(seq)

        return y_hat.item() * self.scale

    def test(self, test_sequences):
        print("\nTesting model:")

        for seq in test_sequences:
            prediction = self.predict_next_number(seq)
            print("Input:", seq, "Predicted next number:", round(prediction, 2))


def main():
    model = RawLSTMNextNumberPredictor(
        input_size=1,
        hidden_size=16,
        output_size=1,
        learning_rate=0.01,
        epochs=400,
        seed=42,
        scale=250.0
    )

    model.create_dataset(num_samples=5000)

    model.train()

    test_sequences = [
        [1, 2, 3, 4],
        [4, 3, 2, 1],
        [4, 8, 12, 16],
        [100, 101, 102],
        [20, 21, 22, 23],
        [50, 51, 52, 53],
        [-10, -5, 0, 5]
    ]

    model.test(test_sequences)


if __name__ == "__main__":
    main()