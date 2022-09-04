from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


x, y = make_classification(n_samples = 100,  n_features = 8, n_redundant = 3, random_state = 30)

train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=42)

model = LinearSVC()

model.fit(train_x, train_y)

print(model.score(test_x, test_y))