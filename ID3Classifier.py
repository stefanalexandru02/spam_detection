import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.label = label
        self.left = None
        self.right = None

class ID3Classifier:
    def __init__(self):
        self.root = None

    def fit(self, X_train, y_train):
        self.root = self.build_tree(X_train, y_train)

    def build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return Node(label=y[0])

        best_feature, best_threshold = self.find_best_split(X, y)
        node = Node(feature=best_feature, threshold=best_threshold)

        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold

        left_X, left_y = X[left_indices], y[left_indices]
        right_X, right_y = X[right_indices], y[right_indices]

        node.left = self.build_tree(left_X, left_y)
        node.right = self.build_tree(right_X, right_y)

        return node

    def find_best_split(self, X, y):
        best_gain = 0
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self.calculate_information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def calculate_information_gain(self, X, y, feature, threshold):
        parent_entropy = self.calculate_entropy(y)

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        left_y = y[left_indices]
        right_y = y[right_indices]

        left_entropy = self.calculate_entropy(left_y)
        right_entropy = self.calculate_entropy(right_y)

        left_weight = len(left_y) / len(y)
        right_weight = len(right_y) / len(y)

        child_entropy = (left_weight * left_entropy) + (right_weight * right_entropy)

        information_gain = parent_entropy - child_entropy
        return information_gain

    def calculate_entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def predict(self, X_test):
        predictions = []
        for sample in X_test:
            node = self.root
            while node.label is None:
                if sample[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.label)
        return np.array(predictions)