import numpy as np
from scipy.sparse import csr_matrix

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def get_neighbors(self, x):
        distances = []
        for i in range(self.X_train.shape[0]):
            dist = self.euclidean_distance(x, self.X_train[i])
            distances.append((dist, self.y_train[i]))
        distances.sort(key=lambda x: x[0])
        neighbors = []
        for i in range(self.n_neighbors):
            neighbors.append(distances[i][1])
        return neighbors

    def predict(self, X_test):
        if isinstance(X_test, csr_matrix):
            X_test = X_test.toarray()
        y_pred = []
        for x in X_test:
            neighbors = self.get_neighbors(x)
            counts = np.bincount(neighbors)
            y_pred.append(np.argmax(counts))
        return y_pred