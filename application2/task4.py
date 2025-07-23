import task2
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import time
import pandas as pd

results = []

d_list = [10, 50, 100, 500, 1000]
n_list = [500, 1000, 5000, 10000, 100000]

for d in d_list:
    for n in n_list:
   
        x_train, x_test, y_train, y_test, x, y = task2.make_classification(d, n, 1000, 42)

        svm_dual = LinearSVC(dual=True, max_iter=1000)
        start_time = time.time()
        svm_dual.fit(x_train, y_train)
        training_time_dual = time.time() - start_time
        predictions_dual = svm_dual.predict(x_test)
        accuracy_dual = accuracy_score(y_test, predictions_dual)
        testing_time_dual = time.time() - (start_time + training_time_dual)

        svm_primal = LinearSVC(dual=False, max_iter=1000)
        start_time = time.time()
        svm_primal.fit(x_train, y_train)
        training_time_primal = time.time() - start_time
        predictions_primal = svm_primal.predict(x_test)
        accuracy_primal = accuracy_score(y_test, predictions_primal)
        testing_time_primal = time.time() - (start_time + training_time_primal)

        results.append({
            'd': d, 'n': n,
            'primal train time': round(training_time_primal, 3),
            'primal test time': round(testing_time_primal, 3),
            'primal accuracy': round(accuracy_primal, 2),
            'dual train time': round(training_time_dual, 3),
            'dual test time': round(testing_time_dual, 3),
            'dual accuracy': round(accuracy_dual, 2)
        })

df_results = pd.DataFrame(results)
print(df_results)
