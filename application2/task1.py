"""
Task 1 : LinearSVC class
Author : Patrick Jojola
"""
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class LinearSVC:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y, C = 0.01): # C - regularization parameter
        _ , features = X.shape

        self.w = np.zeros(features)
        self.b = 0

        y = np.where(y <= 0, -1, 1) # Labels need to be +1 and -1 for linear classifiers...

        for _ in range(self.n_iter):
            for i, xi in enumerate(X):
                margin = y[i] * (np.dot(self.w, xi) + self.b)
                
                if margin >= 1:
                    self.w -= self.eta * C * self.w
                
                else:
                    self.w -= self.eta * (C * self.w - xi * y[i])
                    self.b -= self.eta * -(y[i])

    def net_input(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.where(self.net_input(X) <= 0, -1, 1)