import numpy as np
import pandas as pd

# adds new method predict_step() and modifies original regression tree to handle multidimensional and discrete time dynamical system

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, sse=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.sse = sse
        self.value = value

class MultiDimRegressionTree():
    def __init__(self, data, max_height=None, leaf_size=None, limit="height"):
        self.root = None
        self.max_height = max_height if limit == "height" else None
        self.leaf_size = leaf_size if limit == "leaf size" else None
        self.limit = limit
        self.height = 0
        self.root = self.create_multi_dim_tree(data)

    def create_multi_dim_tree(self, data, depth=0):
        X, Y = data[:, :-2], data[:, -2:]  # input (xk, vk), output (xk+1, vk+1)
        samples, features = np.shape(X)
        
        if depth > self.height:
            self.height = depth

        if (self.leaf_size is not None and samples <= self.leaf_size) or (self.max_height is not None and depth >= self.max_height) or (samples <= 1):
            avg_value = np.mean(Y, axis=0)
            return Node(value=avg_value)

        best = self.best_multi_dim_split(data, features)
        
        if best["sse"] < np.inf:
            left = self.create_multi_dim_tree(best["left"], depth + 1)
            right = self.create_multi_dim_tree(best["right"], depth + 1)
            return Node(best["feature_index"], best["threshold"], left, right, best["sse"])

        avg_value = np.mean(Y, axis=0)
        return Node(value=avg_value)

    def best_multi_dim_split(self, data, features):
        best = {}
        min_sse = np.inf

        for f in range(features):
            f_values = data[:, f]
            thresholds = np.unique(f_values)
            
            for t in thresholds:
                left, right = self.split(data, f, t)
                
                if len(left) > 0 and len(right) > 0:
                    left_y, right_y = left[:, -2:], right[:, -2:]  # Both output variables (xk+1, vk+1)
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
        # Calculate SSE for both outputs (x(k+1) and v(k+1))
        left_sse = np.sum((left - np.mean(left, axis=0))**2)
        right_sse = np.sum((right - np.mean(right, axis=0))**2)
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
        return np.array(predicts)
    
    # new method
    def predict_step(self, X, steps=1):
        predicts = []
        for x in X:
            state = x
            for _ in range(steps):
                next_state = self.traverse(state, self.root)
                state = next_state
            predicts.append(state)
        return np.array(predicts)

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


####################################################################################
# part 1 vehicle example
# generate data for the vehicle example problem
np.random.seed(42)
x = np.linspace(0, 10, 50)
v = np.array([10] * 50) # velocity, 10 m/s
x_new = x + 0.1 * v  # vehicle example equation for v_{k+1}
v_new = v  # always 10 m/s

data = np.column_stack((x, v, x_new, v_new))
np.random.shuffle(data)

train = data[:40] # x_k, v_k for training
print("train data: ", train[:5])
test = data[40:] # x_k, v_k for testing
print("test data: ", test[:5])

tree = MultiDimRegressionTree(train)

initial_state = test[:, :-2] 
predictions = tree.predict_step(initial_state, steps=1)  # 1 step or next state
print("Prediction: ", predictions)

####################################################################################
# part 2 example -----> no model and instead using sets of paired data only - two dimensional
# generate paired data
np.random.seed(42)
x1 = np.linspace(0, 10, 50) # x_k
v1 = np.random.uniform(0, 10, 50)  # v_k
x1_new = np.linspace(0, 10, 50) # x_k
v1_new = np.random.uniform(0, 10, 50) # v_{k+1}

data1 = np.column_stack((x1, v1, x1_new, v1_new))
np.random.shuffle(data1)

train1 = data1[:40] # x_k, v_k for training
print("train data: ", train1[:5])
test1 = data1[40:] # x_k, v_k for testing
print("test data: ", test1[:5])

tree1 = MultiDimRegressionTree(train1)

initial_state1 = test1[:, :-2]
predictions1 = tree1.predict_step(initial_state1, steps=1)  # 1 step or next state
print("Prediction: ", predictions1)

