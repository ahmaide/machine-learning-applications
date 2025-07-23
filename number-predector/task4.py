import numpy as np
from task3 import MultiDimRegressionTree
import time

def next_state(x1, x2):
    new_x1 = 0.9*x1 - 0.2*x2
    new_x2 = 0.2*x1 + 0.9*x2
    return round(new_x1, 4), round(new_x2, 4)

def generate_states():
    states_list = [[0.5, 1.5]]
    for i in range(50):
        new_data1, new_data2 = next_state(states_list[i][0], states_list[i][1])
        if abs(new_data1) <= 5 and abs(new_data2) <= 5:
            states_list.append([new_data1, new_data2])
    return states_list

def data_split(data, train_percentage=0.8):
    train_length = int(train_percentage * len(data))
    training_states = data[:train_length]
    testing_states = data[train_length:]
    x_test = testing_states[:, :2]
    y_test = testing_states[:, 2:]
    return training_states, x_test, y_test

def rounding(round_value, two_states):
    first = round(two_states[0], round_value)
    second = round(two_states[1], round_value)
    return [first, second]



states_list = generate_states()
print("states list: ")
print(states_list)
data = []
for i in range(49):
    data.append([states_list[i][0], states_list[i][1], states_list[i+1][0], states_list[i+1][1]])

data = np.array(data)
np.random.seed(52)
np.random.shuffle(data)
training, x_test, y_test = data_split(data)
regression_tree1 = MultiDimRegressionTree(training, 8, 1, "height")

predicitons = regression_tree1.predict(x_test)
predicitons = [rounding(4, prediction) for prediction in predicitons]
predicitons = np.array(predicitons)

MSE = sum([pow(predection - test_sample, 2) for predection, test_sample in zip(predicitons, y_test)])

print("Prediction")
print(predicitons)
print("Actual Data")
print(y_test)
print("Mean Sqaure Error" + str(MSE))


print("Part 2 ##########################################################################################")

def func(x):
    z=0
    for _ in range(20):
        if x > 1:
            x=0
        else:
            x = x+0.2
            z=z+x
    return [x, z]

x_and_z = [[2, 0]]

for i in range (50):
    next_one = func(x_and_z[i][0])
    if abs(next_one[0])<=3 and next_one[1]>=0 and next_one[1]<=15:
        x_and_z.append(next_one)

print("X and Z values:")
print(x_and_z)

data2 = []

for i in range(50):
    data2.append([x_and_z[i][0], x_and_z[i][1], x_and_z[i+1][0], x_and_z[i+1][1]])

data2 = np.array(data)
np.random.seed(20)
np.random.shuffle(data2)

training2, x_test2, y_test2 = data_split(data2)

regression_tree2 = MultiDimRegressionTree(training2, 8, 1, "height")

predicitons2 = regression_tree2.predict(x_test2)
predicitons2 = [rounding(4, prediction2) for prediction2 in predicitons2]
predicitons2 = np.array(predicitons2)

MSE_2 = sum([pow(predection2 - test_sample2, 2) for predection2, test_sample2 in zip(predicitons2, y_test2)])

print("Prediction for X and Z")
print(predicitons2)
print("Actual X and Z")
print(y_test2)
print("Mean Sqaure Error" + str(MSE_2))


print("Part 3 for  ################# for the first section model ##################################################################")

tree_heights = [3, 4, 5, 6, 8, 16, 32]

for tree_height in tree_heights:
    for num_of_samples in range(1, 5):
        start = time.time()
        regression_tree3 = MultiDimRegressionTree(training, tree_height, num_of_samples, "height")
        data = np.array(data)
        np.random.seed(52)
        np.random.shuffle(data)
        training, x_test, y_test = data_split(data)
        regression_tree1 = MultiDimRegressionTree(training, 8, 1, "height")

        predicitons = regression_tree1.predict(x_test)
        predicitons = [rounding(4, prediction) for prediction in predicitons]
        predicitons = np.array(predicitons)

        MSE_3 = sum([pow(predection - test_sample, 2) for predection, test_sample in zip(predicitons, y_test)])
        end = time.time()

        print("(Tree Height: " + str(tree_height) + ", # of Samples: " + str(num_of_samples) + ") \t MSE: " + str(MSE_3) + "\t" + ", Time: " + str(end-start))


print("Part 3 for  ################# for the second section model ##################################################################")

tree_heights = [3, 4, 5, 6, 8, 16, 32]

for tree_height in tree_heights:
    for num_of_samples in range(1, 5):
        start = time.time()
        regression_tree4 = MultiDimRegressionTree(training2, tree_height, num_of_samples, "height")
        data = np.array(data2)
        np.random.seed(52)
        np.random.shuffle(data2)
        training2, x_test2, y_test2 = data_split(data2)
        regression_tree4 = MultiDimRegressionTree(training2, 8, 1, "height")

        predicitons2 = regression_tree4.predict(x_test2)
        prediciton2 = [rounding(4, prediction2) for prediction2 in predicitons2]
        predicitons2 = np.array(predicitons2)

        MSE_4 = sum([pow(predection2 - test_sample2, 2) for predection2, test_sample2 in zip(predicitons2, y_test2)])
        end = time.time()

        print("(Tree Height: " + str(tree_height) + ", # of Samples: " + str(num_of_samples) + ") \t MSE: " + str(MSE_4) + "\t" + ", Time: " + str(end-start))