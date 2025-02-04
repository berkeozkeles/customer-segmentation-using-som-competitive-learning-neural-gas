import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


class SimpleCompetitiveLearning:
    def __init__(self, n_clusters, n_features, learning_rate=0.1, epochs=100):
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(n_clusters, n_features)  #starting random weights

    def fit(self, X):
        """Simple Competitive Learning Training."""
        for epoch in range(self.epochs):
            for x in X:
                # choosing winning neuron
                winner_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))

                # update winning neuron (Competitive Learning)
                self.weights[winner_idx] += self.learning_rate * (x - self.weights[winner_idx])

            # decreasing learning rate
            self.learning_rate *= 0.99

            if epoch % 10 == 0:
                print(f"Epoch {epoch} - Winner Weights:\n", self.weights)

    def predict(self, X):
        """Assigns each data point to the nearest neuron."""
        clusters = [np.argmin(np.linalg.norm(self.weights - x, axis=1)) for x in X]
        return np.array(clusters)


def load_and_preprocess_data():
    """loading dataset, function that scales features."""
    df = pd.read_csv("../data/Mall_Customers.csv")

    # use only numerical values
    X = df.iloc[:, 2:].values

    # z-score normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def plot_clusters(X, y_pred, model):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_pred, palette="viridis", alpha=0.8)
    plt.scatter(model.weights[:, 0], model.weights[:, 1], color="red", marker="X", s=200, label="Cluster Centers")
    plt.title("Simple Competitive Learning - Customer Segmentation")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # load the data
    X_scaled = load_and_preprocess_data()

    # create scl model and train it
    scl = SimpleCompetitiveLearning(n_clusters=5, n_features=X_scaled.shape[1], learning_rate=0.1, epochs=100)
    scl.fit(X_scaled)

    # make a guess
    y_pred = scl.predict(X_scaled)

    # visualise the results
    plot_clusters(X_scaled, y_pred, scl)
