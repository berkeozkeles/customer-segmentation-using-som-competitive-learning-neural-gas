import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


class SelfOrganizingMap:
    def __init__(self, grid_size, n_features, learning_rate=0.5, sigma=1.0, epochs=500):
        self.grid_size = grid_size
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.epochs = epochs
        self.weights = np.random.rand(grid_size, grid_size, n_features)  # Initialize random weights

    def find_winner(self, x):
        """Find the best matching unit (BMU) for input x"""
        distances = np.linalg.norm(self.weights - x, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def fit(self, X):
        """Train the SOM and visualize learning at different epochs"""
        fig, axes = plt.subplots(1, 6, figsize=(18, 5))
        plot_epochs = [0, self.epochs // 5, 2 * self.epochs // 5, 3 * self.epochs // 5, 4 * self.epochs // 5,
                       self.epochs - 1]

        for epoch in range(self.epochs):
            for x in X:
                winner = self.find_winner(x)

                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        dist_to_winner = np.linalg.norm(np.array([i, j]) - np.array(winner))
                        influence = np.exp(-dist_to_winner ** 2 / (2 * self.sigma ** 2))
                        self.weights[i, j] += self.learning_rate * influence * (x - self.weights[i, j])

            self.learning_rate *= 0.99  # Reduce learning rate over time
            self.sigma *= 0.99  # Reduce neighborhood size

            if epoch in plot_epochs:
                ax = axes[plot_epochs.index(epoch)]
                self.plot_som_grid(ax, X, epoch)

        plt.tight_layout()
        plt.show()

    def predict(self, X):
        """Assign each data point to its closest neuron"""
        return np.array([self.find_winner(x) for x in X])

    def plot_som_grid(self, ax, X, epoch):
        """Plot SOM grid and mapped data points"""
        y_pred = np.array([self.find_winner(x) for x in X])
        colors = sns.color_palette("hsv", len(set([tuple(c) for c in y_pred])))

        for idx, (x, y) in enumerate(y_pred):
            ax.scatter(x + np.random.uniform(-0.2, 0.2),
                       y + np.random.uniform(-0.2, 0.2),
                       color=colors[idx % len(colors)],
                       marker="o", s=80, alpha=0.6)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                ax.scatter(i, j, color="black", marker="s", s=150)

        ax.set_title(f"Epoch {epoch}")
        ax.set_xticks(range(self.grid_size))
        ax.set_yticks(range(self.grid_size))
        ax.grid(True)


def load_and_preprocess_data():
    """Load and return the dataset without normalization"""
    df = pd.read_csv("../data/Mall_Customers.csv")
    X = df.iloc[:, 2:].values  # Select numerical features
    return X


if __name__ == "__main__":
    X = load_and_preprocess_data()
    som = SelfOrganizingMap(grid_size=5, n_features=X.shape[1], learning_rate=0.5, epochs=500)
    som.fit(X)
