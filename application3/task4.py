import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from task1 import train_labels, f_train_labels, test_labels, f_test_labels
from task2 import train_images_reshape, test_images_reshape, f_train_images_reshape, f_test_images_reshape

dimsPCA = [50, 100, 200]
dimsLDA = [2, 5, 9]

datasets = {
    'MNIST': {
        'train_data': train_images_reshape,
        'train_labels': train_labels,
        'test_data': test_images_reshape,
        'test_labels': test_labels
    },
    'Fashion-MNIST': {
        'train_data': f_train_images_reshape,
        'train_labels': f_train_labels,
        'test_data': f_test_images_reshape,
        'test_labels': f_test_labels
    }
}

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

for dataset in datasets:
    x_train = datasets[dataset]['train_data']
    x_test = datasets[dataset]['test_data']
    y_train = datasets[dataset]['train_labels']
    y_test = datasets[dataset]['test_labels']
    x_train, y_train = resample(x_train, y_train, n_samples=5000, random_state=42, stratify=y_train)
    for kernel in kernels:

        for dim in dimsPCA:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('reduce_dim', PCA(n_components=dim)),
                ('svc', SVC(random_state=1))
            ])
            grid_search = GridSearchCV(pipe, param_matrix[kernel], cv=3, scoring='accuracy', n_jobs=-1, refit=True)
            grid_search.fit(x_train, y_train)
            y_pred = grid_search.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            matrix = confusion_matrix(y_test, y_pred)
            print(f"Dataset: {dataset}, Reduction: {dim}, Kernel: {kernel}, Accuracy: {accuracy}")
            ConfusionMatrixDisplay(matrix).plot()
            plt.title(f'Dataset: {dataset}, Reduction: {dim}, Kernel: {kernel}')
            plt.show()

        for dim in dimsLDA:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('reduce_dim', LDA(n_components=10)),
                ('svc', SVC(random_state=1))
            ])
            grid_search = GridSearchCV(pipe, param_matrix[kernel], cv=3, scoring='accuracy', n_jobs=-1, refit=True)
            grid_search.fit(x_train, y_train)
            y_pred = grid_search.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            matrix = confusion_matrix(y_test, y_pred)
            print(f"Dataset: {dataset}, Reduction: {dim}, Kernel: {kernel}, Accuracy: {accuracy}")
            ConfusionMatrixDisplay(matrix).plot()
            plt.title(f'Dataset: {dataset}, Reduction: {dim}, Kernel: {kernel}')
            plt.show()

