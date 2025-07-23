"""
Task 2 : make_classification function

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def make_classification(d, n, u, r_seed):

    np.random.seed(r_seed)
    vector = np.random.randn(d)

    X = np.random.uniform(-u, u, size=(n, d))
    y = np.where(np.dot(X, vector) < 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=r_seed)
    
    return X_train, X_test, y_train, y_test, X, y

# Demo
d = 2
n = 1000
u = 10
seed = 42
X_train, X_test, y_train, y_test, X, y = make_classification(d, n, u, seed)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X[y == -1, 0],
             X[y == -1, 1], 
             c='tomato', marker='s',
             label='Class -1')
plt.scatter(X[y == 1, 0],
             X[y == 1, 1],
             c='royalblue', marker='o',
             label='Class 1')

plt.xlim([-5, 5])
plt.ylim([-5, 5])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title("Task 2: make_classification")
plt.legend(loc='best')
plt.tight_layout()
plt.show()