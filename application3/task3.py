import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from task2 import train_images_reshape, test_images_reshape, f_train_images_reshape, f_test_images_reshape
from task1 import f_test_labels, f_train_labels, test_labels, train_labels

def grid_search(pipe_svc, X, y):
    kernels = ['LINEAR', 'RBF', 'POLY']

    param_range = [0.1, 1.0, 10.0, 100.0]
    gamma_range = [0.001, 0.01, 0.1]
    degree_range = [2, 3]

    param_matrix = {
        'LINEAR': {
            'svc__C': param_range,
            'svc__kernel': ['linear']
        },

        'RBF': {
            'svc__C': param_range,
            'svc__gamma': gamma_range,
            'svc__kernel': ['rbf']
        },

        'POLY': {
            'svc__C': param_range,
            'svc__gamma': gamma_range,
            'svc__degree': degree_range,
            'svc__kernel': ['poly']
        }
    }

    for k in kernels:
        print(k)
        param_grid = param_matrix[k]

        pipe = Pipeline(pipe_svc + [('svc', SVC(random_state=1))])
        gs = GridSearchCV(estimator=pipe,
                        param_grid=param_grid,
                        scoring='accuracy',
                        cv=4,
                        refit=True,
                        n_jobs=-1)

        gs.fit(X, y)

        print("score: ", gs.best_score_)
        print("parameters: ", gs.best_params_)
        print("\n")

x_test_pool = [test_images_reshape, f_test_images_reshape]
x_train_pool = [train_images_reshape, f_train_images_reshape]
y_test_pool = [test_labels, f_test_labels]
y_train_pool = [train_labels, f_train_labels]

dims = [50, 100, 200]
datasets = ["MNIST", "Fashion-MNIST"]

for j in range(2):
    print("----- ", datasets[j], " ------")
    X_train = x_train_pool[j]
    y_train = y_train_pool[j]
    X_test = x_test_pool[j]
    y_test = y_test_pool[j]

    X_train, y_train = resample(X_train, y_train, n_samples = 5000, random_state=42, stratify=y_train)

    for i in dims:
        print(f"****** PCA SVC *****")
        print("DIM: ", i)

        pipe = [
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=i))]

        grid_search(pipe, X_train, y_train)
