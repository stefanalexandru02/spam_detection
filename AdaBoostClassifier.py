import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = {}

    def fit(self, X, y, sample_weights):
        self.tree = self._build_tree(X, y, sample_weights, depth=0)

    def predict(self, X):
        X_transformed = self.vectorizer.transform(X)
        predictions = np.array([self._traverse_tree(x, self.tree) for x in X_transformed])
        return predictions

    def _build_tree(self, X, y, sample_weights, depth):
        n_samples, n_features = X.shape
        classes = np.unique(y)
        best_split = {}
        best_gini = float('inf')

        for feature_idx in range(n_features):
            feature_values = np.unique(X[:, feature_idx])

            for value in feature_values:
                left_indices = np.where(X[:, feature_idx] <= value)[0]
                right_indices = np.where(X[:, feature_idx] > value)[0]

                left_labels = y[left_indices]
                right_labels = y[right_indices]

                left_weights = sample_weights[left_indices]
                right_weights = sample_weights[right_indices]

                gini = self._gini_index(left_labels, right_labels, left_weights, right_weights)

                if gini < best_gini:
                    best_split = {
                        'feature_idx': feature_idx,
                        'value': value,
                        'left_indices': left_indices,
                        'right_indices': right_indices
                    }
                    best_gini = gini

        if best_gini == 0 or (self.max_depth is not None and depth >= self.max_depth):
            leaf_value = self._most_common_class(y)
            return {'leaf': True, 'value': leaf_value}

        left_tree = self._build_tree(X[best_split['left_indices']], y[best_split['left_indices']],
                                     sample_weights[best_split['left_indices']], depth + 1)
        right_tree = self._build_tree(X[best_split['right_indices']], y[best_split['right_indices']],
                                      sample_weights[best_split['right_indices']], depth + 1)

        return {
            'leaf': False,
            'feature_idx': best_split['feature_idx'],
            'value': best_split['value'],
            'left': left_tree,
            'right': right_tree
        }

    def _traverse_tree(self, x, tree):
        if tree['leaf']:
            return tree['value']

        if x[tree['feature_idx']] <= tree['value']:
            return self._traverse_tree(x, tree['left'])
        else:
            return self._traverse_tree(x, tree['right'])

    def _gini_index(self, left_labels, right_labels, left_weights, right_weights):
        left_gini = self._calculate_gini(left_labels, left_weights)
        right_gini = self._calculate_gini(right_labels, right_weights)
        total_samples = len(left_labels) + len(right_labels)
        return (len(left_labels) / total_samples) * left_gini + (len(right_labels) / total_samples) * right_gini

    def _calculate_gini(self, labels, weights):
        classes = np.unique(labels)
        gini = 1.0

        for class_val in classes:
            p = np.sum(weights[labels == class_val]) / np.sum(weights)
            gini -= p ** 2

        return gini

    def _most_common_class(self, labels):
        unique, counts = np.unique(labels, return_counts=True)
        return unique[np.argmax(counts)]

class AdaBoostClassifier:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimator_weights = []
        self.vectorizer = TfidfVectorizer()

    def fit(self, X, y):
        X_transformed = self.vectorizer.fit_transform(X)
        n_samples = X_transformed.shape[0]
        sample_weights = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_estimators):
            estimator = DecisionTreeClassifier(max_depth=1)
            estimator.vectorizer = self.vectorizer
            estimator.fit(X_transformed, y, sample_weights)
            predictions = estimator.predict(X_transformed)

            error = np.sum(sample_weights * (predictions != y))
            estimator_weight = 0.5 * np.log((1 - error) / error)

            sample_weights *= np.exp(-estimator_weight * y * predictions)
            sample_weights /= np.sum(sample_weights)

            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)

    def predict(self, X):
        X_transformed = self.vectorizer.transform(X)
        predictions = np.zeros(X_transformed.shape[0])

        for estimator, estimator_weight in zip(self.estimators, self.estimator_weights):
            predictions += estimator_weight * estimator.predict(X_transformed)

        return np.sign(predictions)