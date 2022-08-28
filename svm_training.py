from sklearn.datasets import make_classification
from sklearn.svm import LinearSVM

x, y = make_classification(n_samples = 100, n_classes = 4, n_features = 10, n_redundant = 3, random_state = 30)