from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


def generate_f_true(x: np.ndarray):
    return np.sin(4 * np.pi * x)


def generate_training_and_validation_data():
    x_train = np.linspace(0., 1., 10)  # training set
    x_valid = np.linspace(0., 1., 100)  # validation set

    y_train = generate_f_true(x_train) + 0.3 * np.random.randn(10)
    y_valid = generate_f_true(x_valid) + 0.3 * np.random.randn(100)

    return x_train, y_train, x_valid, y_valid


def train_model(X_mat: np.ndarray, t_train: np.ndarray) -> np.ndarray:
    """
    Train a model using the analytical solution for least squares linear regression

    :param X_mat: the X matrix that corresponds to the degree of the polynomial
    :param t_train: the training set target row vector
    """
    coefficients = np.linalg.inv(X_mat.T @ X_mat) @ (X_mat.T @ t_train)  # w = (X_transpose * X)^-1 * X_transpose * t

    return coefficients


def train_model_with_reg(X_mat: np.ndarray, B_mat: np.ndarray, t_train: np.ndarray) -> np.ndarray:
    """
    Train a model using the analytical solution for least squares linear regression

    :param X_mat: the X matrix that corresponds to the degree of the polynomial
    :param t_train: the training set target row vector
    """
    # w = (X_transpose * X + (N/2 .* B))^-1 * X_transpose * t
    coefficients = np.linalg.inv(X_mat.T @ X_mat + B_mat) @ (X_mat.T @ t_train)

    return coefficients


def predict_model(coeffs, x_values):
    """
    Calculate the predicted target values for the given model coefficients and given feature values

    :param coeffs: the coefficients of the trained model to use
    :param x_values: the feature matrix to use
    :return: the model predicted target values for a given feature matrix
    """
    # calculate the product of the coefficients and training features element-wise and then sum them row-wise
    return (coeffs * x_values).sum(axis=1, keepdims=True)


def calculate_errors(coeffs: np.ndarray, x_train: np.ndarray, t_train: np.ndarray, x_valid: np.ndarray,
                     t_valid: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the training and validation error for a given model's coefficients

    :param coeffs: the coefficients of the trained model to evaluate
    :param x_train: the training set feature matrix
    :param t_train: the training set target column vector
    :param x_valid: the validation set feature matrix
    :param t_valid: the validation set target column vector
    """
    # get the model predictions for the training feature matrix before subtracting them by the training targets
    element_delta = predict_model(coeffs, x_train) - t_train

    # square each element in the element delta column vector, sum them together, and divide by the number of elements
    training_error = np.sum(np.square(element_delta)) / t_train.shape[0]

    # do the same as above for the validation set
    element_delta = predict_model(coeffs, x_valid) - t_valid
    validation_error = np.sum(np.square(element_delta)) / t_valid.shape[0]

    return training_error, validation_error


def calculate_errors_with_reg(Lambda, coeffs: np.ndarray, x_train: np.ndarray, t_train: np.ndarray, x_valid: np.ndarray,
                              t_valid: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the training and validation error for a given regularised model's coefficients

    :param coeffs: the coefficients of the trained model to evaluate
    :param x_train: the training set feature matrix
    :param t_train: the training set target column vector
    :param x_valid: the validation set feature matrix
    :param t_valid: the validation set target column vector
    """
    # get the original error
    training_error, validation_error = calculate_errors(coeffs, x_train, t_train, x_valid, t_valid)

    training_error += Lambda * np.sum(np.square(coeffs))
    validation_error += Lambda * np.sum(np.square(coeffs))

    return training_error, validation_error


def plot_data(idx, t_pred, x_train, t_train, x_valid, t_valid, reg=False):
    # create a new figure per model
    fig = plt.figure(idx)
    fig.suptitle(f"{'Lambda=e^' if reg else 'Degree '}{idx}")

    # plot the training and validation data
    plt.plot(x_train, t_train, 'o', color="blue", label="Training data", mfc="none")
    plt.plot(x_valid, t_valid, 'o', color="red", label="Validation data", mfc="none")

    # plot the model function and the true function
    plt.plot(x_valid, generate_f_true(x_valid), color="black", label="f_true")
    plt.plot(x_valid, t_pred, color="lightgreen", label="f_pred")

    # show the figure
    plt.legend(loc="best")
    plt.show(block=False)


def plot_error_curves(x_train, t_train, t_valid):
    # create a new figure per model
    fig = plt.figure()
    fig.suptitle(f"Errors vs M")

    # plot the training and validation data
    plt.plot(x_train, t_train, 'o', color="blue", label="Training data", mfc="none")
    plt.plot(x_train, t_valid, 'o', color="red", label="Validation data", mfc="none")

    # show the figure
    plt.legend(loc="best")
    plt.show(block=False)


def main():
    np.random.seed(637)  # student number is 400190637

    x_train, t_train, x_valid, t_valid = generate_training_and_validation_data()

    degrees = np.arange(0, 10)
    exponents = []
    results = {"degree": list(degrees), "training error": [], "validation error": []}

    # Run training for models from degree 0 to 9 inclusive
    for degree in degrees:
        exponents.append(degree)

        # create the X matrix by converting the input x row vector to a column vector, and then raising the
        # elements by each power in the exponents vector with broadcasting
        x_train_mat: np.ndarray = x_train[:, np.newaxis] ** exponents
        x_valid_mat: np.ndarray = x_valid[:, np.newaxis] ** exponents

        # convert target row vectors into column vectors
        t_train_col: np.ndarray = t_train[:, np.newaxis]
        t_valid_col: np.ndarray = t_valid[:, np.newaxis]

        # train model
        coeffs = train_model(x_train_mat, t_train)

        # calculate training and validation error
        train_error, valid_error = calculate_errors(coeffs, x_train_mat, t_train_col, x_valid_mat, t_valid_col)
        print(f"Degree {degree}: training error = {train_error:22}, validation error = {valid_error}")
        results["training error"].append(train_error)
        results["validation error"].append(valid_error)

        plot_data(degree, predict_model(coeffs, x_valid_mat), x_train, t_train, x_valid, t_valid)

    print(f"The lowest training error was at degree {results['degree'][np.argmin(results['training error'])]} and the "
          f"lowest validation error was at degree {results['degree'][np.argmin(results['validation error'])]}")

    # Train the degree 9 model with regularization
    # delete the column of ones at the start of each matrix, so the resultant matrix is 10x9
    x_train_std = np.delete(x_train_mat, 0, 1)
    x_valid_std = np.delete(x_valid_mat, 0, 1)

    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train_std)
    x_valid_std = sc.transform(x_valid_std)

    # add the column of ones back, at the start of each matrix, so the resultant matrix is 10x10
    x_train_std = np.c_[np.ones(t_train.shape[0]), x_train_std]
    x_valid_std = np.c_[np.ones(t_valid.shape[0]), x_valid_std]

    constants = np.arange(-40.0, -1.0)
    reg_results = {"constant": list(constants), "training error": [], "validation error": [], "coeffs": []}
    for constant in constants:
        Lambda = np.e ** constant
        b_mat = np.zeros((10, 10))
        np.fill_diagonal(b_mat, Lambda * t_train.shape[0])  # 2 * Lambda * N / 2

        coeffs = train_model_with_reg(x_train_std, b_mat, t_train)
        reg_results["coeffs"].append(coeffs)

        train_error, valid_error = calculate_errors_with_reg(Lambda, coeffs, x_train_std, t_train_col, x_valid_std,
                                                             t_valid_col)
        print(f"Constant {constant}: training error = {train_error:22}, validation error = {valid_error}")
        reg_results["training error"].append(train_error)
        reg_results["validation error"].append(valid_error)

    idx_min_valid = np.argmin(reg_results['validation error'])
    lambda1 = reg_results['constant'][idx_min_valid]
    lambda2 = reg_results['constant'][-5]

    print(f"The lowest regularised validation error was at constant {lambda1}")

    plot_data(lambda1, predict_model(reg_results["coeffs"][idx_min_valid], x_valid_std), x_train, t_train, x_valid,
              t_valid, reg=True)
    plot_data(lambda2, predict_model(reg_results["coeffs"][-5], x_valid_std), x_train, t_train, x_valid, t_valid,
              reg=True)

    # Add regularised model results, for lambda 1 and 2
    results["degree"].append(9)
    results["training error"].append(reg_results['training error'][idx_min_valid])
    results["validation error"].append(reg_results['validation error'][idx_min_valid])

    results["degree"].append(9)
    results["training error"].append(reg_results['training error'][-5])
    results["validation error"].append(reg_results['validation error'][-5])

    plot_error_curves(results["degree"], results["training error"], results["validation error"])

    avg_valid_set_error = np.sum(np.abs(generate_f_true(x_valid) - t_valid)) / t_valid.shape[0]
    print(f"The average error between the validation set and true function is {avg_valid_set_error}")

    plt.show()


if __name__ == '__main__':
    main()
