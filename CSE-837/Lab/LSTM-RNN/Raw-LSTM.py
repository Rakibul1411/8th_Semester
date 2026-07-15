import numpy as np


class RawLSTMNextNumberPredictor:

    def __init__(
        self,
        input_size=1,
        hidden_size=12,
        output_size=1,
        learning_rate=0.05,
        epochs=100000,
        seed=42
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.concat_size = self.input_size + self.hidden_size

        if seed is not None:
            np.random.seed(seed)

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

    # 2. Dataset creation
    def create_dataset(self):
        X = []
        Y = []

        for i in range(1, 80):
            X.append([i, i + 1, i + 2, i + 3])
            Y.append(i + 4)

        self.X = np.array(X, dtype=float) / 100.0
        self.Y = np.array(Y, dtype=float).reshape(-1, 1) / 100.0

        print("Dataset created.")
        print("X shape:", self.X.shape)
        print("Y shape:", self.Y.shape)

    # 3. Parameter initialization
    def initialize_parameters(self):
        params = {}

        # Forget gate parameters
        params["Wf"] = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        params["bf"] = np.zeros((self.hidden_size, 1))
        # print("Wf shape:", params["Wf"].shape)
        # print("Wf values:\n", params["Wf"])

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

            # Combine previous hidden state and current input
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

            # Store values for backward pass
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

        # Final prediction from last hidden state
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

        # Loss = 1/2 * (y_hat - y_true)^2
        dy = y_hat - y_true

        # Gradient for final output layer
        last_h = caches[-1]["h_t"]

        grads["Wy"] += dy @ last_h.T
        grads["by"] += dy

        # Backprop starts from final hidden state
        dh_next = self.params["Wy"].T @ dy
        dc_next = np.zeros((self.hidden_size, 1))

        # Go backward through time
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

            # h_t = o_t * tanh(c_t)
            tanh_c = np.tanh(c_t)

            do = dh * tanh_c
            do_pre = do * self.dsigmoid(o_t)

            dc = dh * o_t * (1 - tanh_c ** 2) + dc_next

            # c_t = f_t * c_prev + i_t * c_candidate
            df = dc * c_prev
            df_pre = df * self.dsigmoid(f_t)

            di = dc * c_candidate
            di_pre = di * self.dsigmoid(i_t)

            dcandidate = dc * i_t
            dcandidate_pre = dcandidate * self.dtanh(c_candidate)

            # Forget gate gradients
            grads["Wf"] += df_pre @ combined.T
            grads["bf"] += df_pre

            # Input gate gradients
            grads["Wi"] += di_pre @ combined.T
            grads["bi"] += di_pre

            # Candidate memory gradients
            grads["Wc"] += dcandidate_pre @ combined.T
            grads["bc"] += dcandidate_pre

            # Output gate gradients
            grads["Wo"] += do_pre @ combined.T
            grads["bo"] += do_pre

            # Gradient wrt combined = [h_prev, x_t]
            dcombined = (
                self.params["Wf"].T @ df_pre +
                self.params["Wi"].T @ di_pre +
                self.params["Wc"].T @ dcandidate_pre +
                self.params["Wo"].T @ do_pre
            )

            # Split gradient
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
            total_loss = 0

            indices = np.random.permutation(len(self.X))

            for idx in indices:
                sequence = self.X[idx]
                y_true = self.Y[idx].reshape(1, 1)

                # 1. Forward pass
                y_hat, caches = self.forward(sequence)

                # 2. Loss
                loss = self.compute_loss(y_hat, y_true)
                total_loss += loss

                # 3. Backward pass
                grads = self.backward(y_hat, y_true, caches)

                # 4. Parameter update
                self.update_parameters(grads)

            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / len(self.X)
                print(f"Epoch {epoch + 1}, Loss: {avg_loss:.8f}")

    # 9. Prediction
    def predict_next_number(self, seq):
        seq = np.array(seq, dtype=float) / 100.0
        y_hat, _ = self.forward(seq)

        return y_hat.item() * 100.0

    def test(self, test_sequences):
        print("\nTesting model:")

        for seq in test_sequences:
            prediction = self.predict_next_number(seq)
            print("Input:", seq, "Predicted next number:", round(prediction, 2))

# Main function
def main():
    model = RawLSTMNextNumberPredictor(
        input_size=1,
        hidden_size=12,
        output_size=1,
        learning_rate=0.05,
        epochs=1000,
        seed=42
    )

    model.create_dataset()

    model.train()

    test_sequences = [
        [1, 2, 3, 4],
        [10, 11, 12, 13],
        [20, 21, 22, 23],
        [50, 51, 52, 53]
    ]

    model.test(test_sequences)

if __name__ == "__main__":
    main()