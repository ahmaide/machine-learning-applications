import numpy as np
import pandas as pd

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, sse=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.sse = sse
        
        self.value = value

class RegressionTree():
    def __init__(self, data, max_height=None, leaf_size=None, limit="height"):
        self.root = None
        self.max_height = max_height if limit == "height" else None
        self.leaf_size = leaf_size if limit == "leaf size" else None
        self.limit = limit
        self.height = 0
        
        self.root = self.create_tree(data)

    def create_tree(self, data, depth=0):
        X, Y = data[:, :-1], data[:, -1]
        samples, features = np.shape(X)
        best = {}

        if (depth > self.height):
            self.height = depth

        if ((self.leaf_size is not None and samples <= self.leaf_size) or (self.max_height is not None and depth >= self.max_height) or (samples <= 1)):
            l_value = np.mean(Y)
            return Node(value=l_value)

        best = self.best_split(data, features)

        if best["sse"] < np.inf:
            left = self.create_tree(best["left"], depth + 1)
            right = self.create_tree(best["right"], depth + 1)
            return Node(best["feature_index"], best["threshold"], left, right, best["sse"])

        l_value = np.mean(Y)
        return Node(value=l_value)

    def best_split(self, data, features):
        best = {}
        min_sse = np.inf

        for f in range(features):
            f_values = data[:, f]
            thresholds = np.unique(f_values)
            
            for t in thresholds:
                left, right = self.split(data, f, t)
                
                if len(left) > 0 and len(right) > 0:
                    left_y, right_y = left[:, -1], right[:, -1]
                    sse = self.calculate_sse(left_y, right_y)
                    
                    if sse < min_sse:
                        best["feature_index"] = f
                        best["threshold"] = t
                        best["left"] = left
                        best["right"] = right
                        best["sse"] = sse
                        min_sse = sse
        
        return best
    
    def split(self, data, feature_index, threshold):
        left = np.array([r for r in data if r[feature_index] <= threshold])
        right = np.array([r for r in data if r[feature_index] > threshold])
        
        return left, right
    
    def calculate_sse(self, left, right):
        left_sse = np.sum((left - np.mean(left))**2)
        right_sse = np.sum((right - np.mean(right))**2)
        total_sse = left_sse + right_sse

        return total_sse
    
    def traverse(self, x, tree):
        if tree.value is not None:
            return tree.value
        
        f_val = x[tree.feature_index]
        
        if f_val <= tree.threshold:
            return self.traverse(x, tree.left)

        else:
            return self.traverse(x, tree.right)

    def predict(self, X):
        predicts = [self.traverse(x, self.root) for x in X]
        return predicts
    
    def decision_path(self, x, node=None):
        if node is None:
            node = self.root

        if node.value is not None:
            print(f"Prediction: {node.value}")
            return "END"
        
        if x[node.feature_index] <= node.threshold:
            print(f"X(feature = {node.feature_index}) <= {node.threshold}")
            return self.decision_path(x, node.left)

        else:
            print(f"X(feature = {node.feature_index}) > {node.threshold}")
            return self.decision_path(x, node.right)
