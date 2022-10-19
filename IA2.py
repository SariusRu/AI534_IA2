import numpy as np
import pandas as pd

# constants
train_data = "/home/sam/Documents/source/Python/AI534/IA2/IA2-train.csv"
val = "IA1_dev.csv"
convergence_threshold = 0.0005
learning_rate = 0.0001


# Loads a data file from a provided file location.
def load_data(path):
    # Your code here:
    return pd.read_csv(path)


# Implements dataset preprocessing. For this assignment, you just need to implement normalization 
# of the three numerical features.

def preprocess_data(data, normalize):

    normalize_data = data[['Age', 'Vintage', 'Annual_Premium']]
    data = data.drop(['Age', 'Vintage', 'Annual_Premium'], axis=1)

    if normalize:
        normalize_data = (normalize_data - normalize_data.min()) / (normalize_data.max() - normalize_data.min())

    data['Age'] = normalize_data['Age']
    data['Vintage'] = normalize_data['Vintage']
    data['Annual_Premium'] = normalize_data['Annual_Premium']


    classes = data['Response']
    data = data.drop(['Response'], axis=1)
    return data, classes


def ridge_regression(X, y, alpha=0.01, lambda_value=1, epochs=30):
    """
    :param X: feature matrix
    :param y: target vector
    :param alpha: learning rate (default:0.01)
    :param lambda_value: lambda (default:1)
    :param epochs: maximum number of iterations of the
           linear regression algorithm for a single run (default=30)
    :return: weights, list of the cost function changing overtime
    """

    m = np.shape(X)[0]  # total number of samples
    n = np.shape(X)[1]  # total number of features

    X = np.concatenate((np.ones((m, 1)), X), axis=1)
    W = np.random.randn(n + 1, )

    # stores the updates on the cost function (loss function)
    cost_history_list = []

    # iterate until the maximum number of epochs
    for current_iteration in range(epochs):  # begin the process
        y_estimated = X.dot(W)
        error = y_estimated - y
        ridge_reg_term = (lambda_value / 2 * m) * np.sum(np.square(W))
        cost = (1 / 2 * m) * np.sum(error ** 2) + ridge_reg_term
        gradient = (1 / m) * (X.T.dot(error) + (lambda_value * W))
        W = W - alpha * gradient
        print(f"cost:{cost} \t iteration: {current_iteration}")
        cost_history_list.append(cost)

    return W, cost_history_list



# Trains a logistic regression model with L2 regularization on the provided train_data, using the supplied lambd
# weights should store the per-feature weights of the learned logisitic regression model. train_acc and val_acc 
# should store the training and validation accuracy respectively. 
def LR_L2_train(train_data, classes):
    weights, cost_history = ridge_regression(train_data, classes.to_numpy())
    print(weights)
    print(cost_history)

    train_acc = None
    val_acc = None

    return weights, train_acc, val_acc


# Trains a logistic regression model with L1 regularization on the provided train_data, using the supplied lambd
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
LR_L2_train(data, classes)

# Part 1 . Implement logistic regression with L2 regularization and experiment with different lambdas
# Your code here:


# Part 2  Training and experimenting with IA2-train-noisy data.
# Your code here:


# Part 3  Implement logistic regression with L1 regularization and experiment with different lambdas
# Your code here:
