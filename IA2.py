import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# constants
train_data = "/home/sam/Documents/source/Python/AI534/IA2/IA2-train.csv"
val = "/home/sam/Documents/source/Python/AI534/IA2/IA2-dev.csv"
THRESHOLD = 20
LEARNING_RATE = 0.1
LAMBDA_VALUE = 1


# Loads a data file from a provided file location.
def load_data(path):
    # Your code here:
    return pd.read_csv(path)


# Implements dataset preprocessing. For this assignment, you just need to implement normalization 
# of the three numerical features.

def preprocess_data(loaded_data, normalize):

    normalize_data = loaded_data[['Age', 'Vintage', 'Annual_Premium']]
    loaded_data = loaded_data.drop(['Age', 'Vintage', 'Annual_Premium'], axis=1)

    if normalize:
        normalize_data = (normalize_data - normalize_data.min()) / (normalize_data.max() - normalize_data.min())

    loaded_data['Age'] = normalize_data['Age']
    loaded_data['Vintage'] = normalize_data['Vintage']
    loaded_data['Annual_Premium'] = normalize_data['Annual_Premium']

    target_classes = loaded_data['Response']
    loaded_data = loaded_data.drop(['Response'], axis=1)
    return loaded_data, target_classes


def regression(x, y, lambda_value):
    m = np.shape(x)[0]  # total number of samples
    n = np.shape(x)[1]  # total number of features

    weights = np.zeros(n, )

    costs = []

    while True:
        y_estimated = x.dot(weights)
        error = y_estimated - y
        ridge_reg_term = (lambda_value / m) * np.sum(np.square(weights))
        cost = (0.5 * m) * np.sum(error ** 2) + ridge_reg_term
        gradient = (1 / m) * (x.T.dot(error) + (lambda_value * weights))
        weights = weights - LEARNING_RATE * gradient
        # print(f"iter: {len(costs)}\tcost:{cost}")
        costs.append(cost)
        if len(costs) > 2 and (
                (costs[len(costs)-2]-costs[len(costs)-1] < THRESHOLD) or costs[len(costs)-1] < 0):
            return weights, costs

def sig(x):
    return 1/(1 + np.exp(-x))

def regression2(x, y, lambda_value):
    m = np.shape(x)[0]  # total number of samples
    n = np.shape(x)[1]  # total number of features

    weights = np.zeros(n, )

    costs = []

    while True:
        y_estimated = x.dot(weights)
        error = y_estimated - y
        ridge_reg_term = (lambda_value / m) * np.sum(np.square(weights))
        cost = (0.5 * m) * np.sum(error ** 2) + ridge_reg_term
        gradient = (1 / m) * (x.T.dot(error) + (lambda_value * weights))
        for index, weight_value in enumerate(weights):
            weight_tmp = 0
            if (abs(weight_value) - (lambda_value *LEARNING_RATE)) < 0:
                weight_tmp = 0
            else:
                weight_tmp = abs(weight_value)-(lambda_value*LEARNING_RATE)
            weight_tmp*sig(weight_value)
            weights[index] = weight_tmp
        # print(f"iter: {len(costs)}\tcost:{cost}")
        costs.append(cost)
        if len(costs) > 2 and (
                (costs[len(costs)-2]-costs[len(costs)-1] < THRESHOLD) or costs[len(costs)-1] < 0):
            return weights, costs


def check_accuracy(weights, data, classes):
    pred = [0] * len(data)
    dif = 0
    estimated = data.dot(weights)
    for estimated_index, est_value in enumerate(estimated):
        pred[estimated_index] = round(est_value)
    for index, value in enumerate(pred):
        dif = dif + abs(classes[index] - value)

    dif = (dif/len(data))*100
    print(dif)
    print(len(data))
    return dif


def lr_l2_train(train_data, classes, val_data, val_classes, lambda_value = LAMBDA_VALUE):
    weights, cost_history = regression(train_data, classes.to_numpy(), lambda_value)
    print(f"Calculation after {len(cost_history)} iterations completed")

    train_acc = check_accuracy(weights, train_data, classes)
    val_acc = check_accuracy(weights, val_data, val_classes)

    print(train_acc)
    print(val_acc)

    return weights, train_acc, val_acc


# Trains a logistic regression model with L1 regularization on the provided train_data, using the supplied lambda
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def lr_l1_train(train_data, classes, val_data, val_classes, lambda_value = LAMBDA_VALUE):
    # Your code here:
    weights, cost_history = regression2(train_data, classes.to_numpy(), lambda_value)
    print(f"Calculation after {len(cost_history)} iterations completed")

    train_acc = check_accuracy(weights, train_data, classes)
    val_acc = check_accuracy(weights, val_data, val_classes)

    print(train_acc)
    print(val_acc)

    return weights, train_acc, val_acc


# Generates and saves plots of the accuracy curves. Note that you can interpret accs as a matrix
# containing the accuracies of runs with different lambda values and then put multiple loss curves in a single plot.
def plot_accuracy(lambda_values, train, val):
    # make

    # plot
    fig, ax = plt.subplots()

    ax.plot(lambda_values, train, linewidth=2.0)
    ax.plot(lambda_values, val, linewidth=2.0)

    plt.xscale('log')

    plt.show()
    return


def print_weights(weights):
    weights = weights.sort_values(ascending=False)
    print(weights.head(n=5))


def calculate2(values, powered=False):
    data = load_data(train_data)
    data, classes = preprocess_data(data, True)
    val_data = load_data(val)
    val_data, val_classes = preprocess_data(val_data, True)

    train_acc_collected = []
    val_acc_collected = []
    weights_collected = []
    raw_lambda_values = values
    lambda_values = []
    for value in raw_lambda_values:
        if not powered:
            lambda_values.append(10 ** value)
        else:
            lambda_values.append(value)

    for value in lambda_values:
        LAMBDA_VALUE = value
        print(f'Training with {LAMBDA_VALUE} as Lambda')
        weights, train_acc, val_acc = lr_l1_train(data, classes, val_data, val_classes, value)
        weights_collected.append(weights)
        train_acc_collected.append(train_acc)
        val_acc_collected.append(val_acc)

        print_weights(weights)

    plot_accuracy(lambda_values, train_acc_collected, val_acc_collected)
    return weights_collected


def calculate(values, powered=False):
    data = load_data(train_data)
    data, classes = preprocess_data(data, True)
    val_data = load_data(val)
    val_data, val_classes = preprocess_data(val_data, True)

    train_acc_collected = []
    val_acc_collected = []
    weights_collected = []
    raw_lambda_values = values
    lambda_values = []
    for value in raw_lambda_values:
        if not powered:
            lambda_values.append(10 ** value)
        else:
            lambda_values.append(value)

    for value in lambda_values:
        LAMBDA_VALUE = value
        print(f'Training with {LAMBDA_VALUE} as Lambda')
        weights, train_acc, val_acc = lr_l2_train(data, classes, val_data, val_classes, value)
        weights_collected.append(weights)
        train_acc_collected.append(train_acc)
        val_acc_collected.append(val_acc)

        print_weights(weights)

    plot_accuracy(lambda_values, train_acc_collected, val_acc_collected)
    return weights_collected



def Task1a():
    calculate([-4, -3, -2, -1, 0, 0.5, 1, 1.5, 2, 3], False)

def Task1b():
    calculate([10, 100, 1000], True)

def Task1c():
    weights = calculate([0.00000001, 0.0001, 0.001, 0.01, 0.1, 0, 0.5, 0.25, 0.45, 1, 10, 12, 15, 20], True)
    for index_outer, value in enumerate(weights):
        counter = 0
        for index_inner, weight_value in enumerate(value):
            if(abs(weight_value)<(10**-6)):
                counter = counter + 1
        print(counter)
        # print(value)

def Task3():




# Task1a()
#Task1b()
Task1c()


# Part 2  Training and experimenting with IA2-train-noisy data.
# Your code here:


# Part 3  Implement logistic regression with L1 regularization and experiment with different lambdas
# Your code here:
