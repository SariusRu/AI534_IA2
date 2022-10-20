import numpy as np
import pandas as pd

# constants
train_data = "/home/sam/Documents/source/Python/AI534/IA2/IA2-train.csv"
val = "IA1_dev.csv"
THRESHOLD = 50
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


def regression(x, y):
    m = np.shape(x)[0]  # total number of samples
    n = np.shape(x)[1]  # total number of features

    x = np.concatenate((np.ones((m, 1)), x), axis=1)
    weights = np.random.randn(n + 1, )

    costs = []

    while True:
        y_estimated = x.dot(weights)
        error = y_estimated - y
        ridge_reg_term = (LAMBDA_VALUE / 2 * m) * np.sum(np.square(weights))
        cost = (0.5 * m) * np.sum(error ** 2) + ridge_reg_term
        gradient = (1 / m) * (x.T.dot(error) + (LAMBDA_VALUE * weights))
        weights = weights - LEARNING_RATE * gradient
        print(f"cost:{cost}")
        costs.append(cost)
        if len(costs) > 2 and (costs[len(costs)-2]-costs[len(costs)-1] < THRESHOLD):
            return weights, costs


def lr_l2_train(train_data, classes, val_data, val_classes):
    weights, cost_history = regression(train_data, classes.to_numpy())
    print(f"Calculation after {len(cost_history)} iterations completed")
    print(weights)

    train_acc = None
    val_acc = None

    return weights, train_acc, val_acc


# Trains a logistic regression model with L1 regularization on the provided train_data, using the supplied lambda
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L1_train(train_data, val_data, lambda_value):
    # Your code here:

    return weights, train_acc, val_acc


# Generates and saves plots of the accuracy curves. Note that you can interpret accs as a matrix
# containing the accuracies of runs with different lambda values and then put multiple loss curves in a single plot.
def plot_losses(accs):
    # Your code here:

    return


# Invoke the above functions to implement the required functionality for each part of the assignment.
# Part 0  : Data preprocessing.
# Your code here:


data = load_data(train_data)
data, classes = preprocess_data(data, True)
val_data = load_data(val)
val_data, val_classes = preprocess_data(val_data, True)
lr_l2_train(data, classes, val_data, val_classes)

# Part 1 . Implement logistic regression with L2 regularization and experiment with different lambdas
# Your code here:


# Part 2  Training and experimenting with IA2-train-noisy data.
# Your code here:


# Part 3  Implement logistic regression with L1 regularization and experiment with different lambdas
# Your code here:
