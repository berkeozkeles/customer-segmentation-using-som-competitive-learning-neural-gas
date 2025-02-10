import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralGas:
    def __init__(self, n_neurons, n_features, learning_rate=0.5, lambda_decay=0.99, epochs=500):
        self.n_neurons = n_neurons
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.lambda_decay = lambda_decay
        self.epochs = epochs
        self.weights = np.random.rand(n_neurons, n_features)
        self.loss_history = []

    def train(self, X):
        lambda_value = self.n_neurons / 2
        plot_epochs = [100, 200, 300, 400, 499]

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        for epoch in range(self.epochs):
            np.random.shuffle(X)
            epoch_loss = 0

            for x in X:
                distances = np.linalg.norm(self.weights - x, axis=1)
                rankings = np.argsort(distances)
                epoch_loss += np.min(distances)

                for rank, neuron_idx in enumerate(rankings):
                    influence = np.exp(-rank / lambda_value)
                    self.weights[neuron_idx] += self.learning_rate * influence * (x - self.weights[neuron_idx])

            self.loss_history.append(epoch_loss / len(X))
            self.learning_rate *= self.lambda_decay
            lambda_value *= self.lambda_decay

            if epoch in plot_epochs:
                idx = plot_epochs.index(epoch)
                ax = axes[idx // 3, idx % 3]
                self.plot_neurons(ax, X, epoch)

        self.plot_loss(axes[1, 2])

        plt.tight_layout()
        plt.show()

    def predict(self, X):
        return np.array([np.argmin(np.linalg.norm(self.weights - x, axis=1)) for x in X])

    def plot_neurons(self, ax, X, epoch):
        ax.scatter(X[:, 0], X[:, 1], color="lightblue", alpha=0.6)
        ax.scatter(self.weights[:, 0], self.weights[:, 1], color="red", marker="X", s=100)
        ax.set_title(f"Epoch {epoch}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    def plot_loss(self, ax):
        ax.plot(range(1, len(self.loss_history) + 1), self.loss_history, label="Loss", color="b", linewidth=2)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss Value")
        ax.set_title("Loss Over Epochs")
        ax.legend()
        ax.grid(True)

def load_and_preprocess_data():
    df = pd.read_csv("../data/Mall_Customers.csv")
    X = df.iloc[:, 2:].values
    return X

if __name__ == "__main__":
    X = load_and_preprocess_data()
    ng = NeuralGas(n_neurons=5, n_features=X.shape[1], learning_rate=0.5, epochs=500)
    ng.train(X)
    y_pred = ng.predict(X)
