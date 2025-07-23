import task1
import task2
from sklearn.metrics import accuracy_score
import time
import numpy as np

class LinearSVC_Modified:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y, C=0.01):
        samples, features = X.shape

        self.w = [0] * features
        self.b = 0

        y = [-1 if yi <= 0 else 1 for yi in y]

        for _ in range(self.n_iter):
            for i in range(samples):
                margin = y[i] * (sum(self.w[j] * X[i][j] for j in range(features)) + self.b)

                if margin >= 1:
                    self.w = [self.w[j] - self.eta * (2 * C * self.w[j]) for j in range(features)]
                else:
                    self.w = [self.w[j] - self.eta * (2 * C * self.w[j] - X[i][j] * y[i]) for j in range(features)]
                    self.b -= self.eta * -y[i]

    def net_input(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.where(self.net_input(X) <= 0, -1, 1)

d_list = [10, 50, 100, 500, 1000]

n_list = [500, 1000, 5000, 10000, 100000]

training_times = {}
testing_times = {}
imp_times = {}
accuracies = {}

for d in d_list:
    for n in n_list:
        x_train, x_test, y_train, y_test, x, y = task2.make_classification(d, n, 1000, 42)
        Svm = task1.LinearSVC(0.01, 50, 1)
        start_time = time.time()
        Svm.fit(x_train, y_train)
        time_between = time.time()
        predictions = Svm.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        end_time = time.time()
        training_times[(d, n)] = round(time_between - start_time, 3)
        testing_times[(d, n)] = round(end_time - time_between, 3)
        imp_times[(d, n)] = round(end_time - start_time, 3)
        accuracies[(d, n)] = round(accuracy, 2)
        print("(d, n) = (" + str(d) + ", " + str(n) + "): { Training Time: " + str(training_times[(d, n)]) + ", Testing Time: "
              + str(testing_times[(d, n)]) + ", Full Time: " + str(imp_times[(d, n)]) + ", Accuracy: " +
              str(round(accuracy, 2)) + "}")
print("======================================================================================================")

for d in d_list:
    for n in n_list:
        x_train, x_test, y_train, y_test, x, y = task2.make_classification(d, n, 1000, 42)
        Svm = LinearSVC_Modified(0.01, 50, 1)
        start_time = time.time()
        Svm.fit(x_train, y_train)
        time_between = time.time()
        predictions = Svm.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        end_time = time.time()
        training_times[(d, n)] = round(time_between - start_time, 3)
        testing_times[(d, n)] = round(end_time - time_between, 3)
        imp_times[(d, n)] = round(end_time - start_time, 3)
        accuracies[(d, n)] = round(accuracy, 2)
        print("(d, n) = (" + str(d) + ", " + str(n) + "): { Training Time: " + str(training_times[(d, n)]) + ", Testing Time: "
              + str(testing_times[(d, n)]) + ", Full Time: " + str(imp_times[(d, n)]) + ", Accuracy: " +
              str(round(accuracy, 2)) + "}")