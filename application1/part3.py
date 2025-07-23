import sys
from urllib.error import HTTPError
import numpy as np
import pandas as pd

import matplotlib
from matplotlib.colors import ListedColormap

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, '..')

d = {
    'numpy': '1.21.2',
    'matplotlib': '3.4.3',
    'pandas': '1.3.2'
}



class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size= x.shape[1]+1)
        self.errors_list = []

        for _ in range(self.n_iter):
            errors_count = 0
            for x_variable, target in zip(x, y):
                prediction = self.predict(x_variable)
                update = self.eta * (target - prediction)
                # appending 1 for bias
                self.weights += update * np.append(x_variable, 1)
                errors_count += int(update != 0.0)
            self.errors_list.append(errors_count)
        return self

    def net_input(self, x):
        if x.ndim == 1:
            x = np.append(x, 1)
        else:
            x = np.hstack((x, np.ones((x.shape[0], 1))))
        return np.dot(x, self.weights)

    def predict(self, x):
        """Return class label after unit step"""
        return np.where(self.net_input(x) >= 0.0, 1, 0)


def split_data(x, y, train_size=0.8):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size,
                                                        random_state=55, stratify=y)
    return x_train, x_test, y_train, y_test


def plot_decision_regions(x, y, classifier, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0],
                    y=x[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')

def delete_label(x, y, class_index):
    start_index = 50*class_index
    end_index = 50*(class_index+1)
    x = np.concatenate((x[0:start_index], x[end_index:150]))
    y = np.concatenate((y[0:start_index], y[end_index:150]))
    return x, y


def filter_data(x, y, mode, segment_size=50):
    keep_indices = []
    segment_slices = {
        0: [slice(0, segment_size),
            np.random.choice(np.arange(segment_size, 2 * segment_size), segment_size // 2, replace=False),
            np.random.choice(np.arange(2 * segment_size, 3 * segment_size), segment_size // 2, replace=False)],
        1: [np.random.choice(np.arange(0, segment_size), segment_size // 2, replace=False),
            slice(segment_size, 2 * segment_size),
            np.random.choice(np.arange(2 * segment_size, 3 * segment_size), segment_size // 2, replace=False)],
        2: [np.random.choice(np.arange(0, segment_size), segment_size // 2, replace=False),
            np.random.choice(np.arange(segment_size, 2 * segment_size), segment_size // 2, replace=False),
            slice(2 * segment_size, 3 * segment_size)]
    }
    for idx in segment_slices[mode]:
        if isinstance(idx, slice):
            keep_indices.extend(range(idx.start, idx.stop))
        else:
            keep_indices.extend(idx)

    x_filtered = x[keep_indices]
    y_filtered = y[keep_indices]
    return x_filtered, y_filtered


v1 = np.array([1, 2, 3])
v2 = 0.5 * v1
np.arccos(v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


try:
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    print('From URL:', s)
    df = pd.read_csv(s,
                     header=None,
                     encoding='utf-8')

except HTTPError:
    s = 'iris.data'
    print('From local Iris path:', s)
    df = pd.read_csv(s,
                     header=None,
                     encoding='utf-8')

df.tail()

x = df.iloc[:150, 0:4].values
y = df.iloc[:150, 4].values



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

axes[0].scatter(x[:50, 0], x[:50, 2], color='red', marker='o', label='Setosa')
axes[0].scatter(x[50:100, 0], x[50:100, 2], color='blue', marker='s', label='Versicolor')
axes[0].scatter(x[100:150, 0], x[100:150, 2], color='green', marker='^', label='Virginica')
axes[0].set_xlabel('Sepal Length [cm]')
axes[0].set_ylabel('Petal Length [cm]')
axes[0].legend(loc='upper left')
axes[0].set_title('Sepal vs Petal Length')

axes[1].scatter(x[:50, 1], x[:50, 3], color='red', marker='o', label='Setosa')
axes[1].scatter(x[50:100, 1], x[50:100, 3], color='blue', marker='s', label='Versicolor')
axes[1].scatter(x[100:150, 1], x[100:150, 3], color='green', marker='^', label='Virginica')
axes[1].set_xlabel('Sepal Width [cm]')
axes[1].set_ylabel('Petal Width [cm]')
axes[1].legend(loc='upper left')
axes[1].set_title('Sepal vs Petal Width')

plt.tight_layout()
plt.show()

eta_global = 0.01
n_iter_global = 15

perceptron_demos = []
max_accuracy = 0
best_labeling=-1
best_labels = []


unique_labels = np.unique(y)

for index, unique_label in enumerate(unique_labels):
    x_filtered, y_filtered = filter_data(x, y, index)

    x_train, x_test, y_train, y_test = split_data(x_filtered, y_filtered)
    y_filtered = np.where(y_filtered == unique_label, 0, 1)
    y_train = np.where(y_train == unique_label, 0, 1)
    y_test = np.where(y_test == unique_label, 0, 1)


    perceptron_demo = Perceptron(eta=eta_global, n_iter=n_iter_global)
    perceptron_demos.append(perceptron_demo)
    perceptron_demo.fit(x_train, y_train)

    predictions = perceptron_demo.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy for {unique_label} is: {accuracy}')
    if accuracy > max_accuracy:
        best_labeling=index
        max_accuracy = accuracy
        best_labels = y_filtered
    plt.plot(range(1, len(perceptron_demo.errors_list) + 1), perceptron_demo.errors_list, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of updates')
    plt.title(f'Classification for {unique_label} VS all')
    plt.show()
    """Only used for 2-feature regions"""
    # plot_decision_regions(x_filtered, y_filtered, classifier=perceptron_demo)
    # plt.xlabel('Sepal width [cm]')
    # plt.ylabel('Petal width [cm]')
    # plt.legend(loc='upper left')
    # plt.title(f'Decision region for {unique_label} VS all')
    # plt.show()

""" Part 2 of the classification"""

x_new, y_new = delete_label(x, y, best_labeling)
x_train, x_test, y_train, y_test = split_data(x_new, y_new)
unique_labels = np.unique(y_new)

training_labels = np.where(y_train == unique_labels[0], 0, 1)
testing_labels = np.where(y_test == unique_labels[0], 0, 1)
y_labels = np.where(y_new == unique_labels[0], 0, 1)
perceptron2 = Perceptron(eta=eta_global, n_iter=n_iter_global)
perceptron2.fit(x_train, training_labels)
predictions = perceptron2.predict(x_test)
accuracy = accuracy_score(testing_labels, predictions)
print("=========================================================")
print(f'Accuracy for  {unique_labels[0]} VS {unique_labels[1]} is: {accuracy}')
plt.plot(range(1, len(perceptron2.errors_list) + 1), perceptron2.errors_list, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.title(f'Classification for {unique_labels[0]} VS {unique_labels[1]}')
plt.show()

"""Only used for 2-feature regions"""
# plot_decision_regions(x_new, y_new, classifier=perceptron2)
# plt.xlabel('Sepal width [cm]')
# plt.ylabel('Petal with [cm]')
# plt.legend(loc='upper left')
# plt.show()

def plot_decision_regions_for3(x, y, class1, class2, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'green', 'gray')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    x_plot, y_plot = np.meshgrid(np.arange(x_min, x_max, resolution),
                           np.arange(y_min, y_max, resolution))

    graph1 = class1.predict(np.array([x_plot.ravel(), y_plot.ravel()]).T)
    graph2 = class2.predict(np.array([x_plot.ravel(), y_plot.ravel()]).T)
    graph1 = graph1.reshape(x_plot.shape)
    graph2 = graph2.reshape(y_plot.shape)
    combined_decisions = graph1 * 2 + graph2
    plt.contourf(x_plot, y_plot, combined_decisions, alpha=0.3, cmap=cmap)
    plt.xlim(x_plot.min(), x_plot.max())
    plt.ylim(y_plot.min(), y_plot.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=f'Class {cl}',
                    edgecolor='black')
    plt.xlabel('Sepal width [cm]')
    plt.ylabel('Petal width [cm]')
    plt.legend(loc='upper left')
    plt.title('All 3 Decision Regions')
    plt.show()

"""Only used for 2-feature regions"""
#plot_decision_regions_for3(x, y, perceptron_demos[best_labeling], perceptron2)

