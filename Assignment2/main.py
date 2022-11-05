import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures


def get_dataset_splits(dataset, rnd_seed):
    """
    Get training and testing splits from a given dataset

    :param dataset:
    :param rnd_seed:
    :return:
    """
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = dataset.target

    return train_test_split(X, y, test_size=0.2, random_state=rnd_seed)


def train_model(X_mat: np.ndarray, t_train: np.ndarray) -> np.ndarray:
    """
    Train a model using the analytical solution for least squares linear regression

    :param X_mat: the X matrix that corresponds to the degree of the polynomial
    :param t_train: the training set target row vector
    """
    coefficients = np.linalg.inv(X_mat.T @ X_mat) @ (X_mat.T @ t_train)  # w = (X_transpose * X)^-1 * X_transpose * t

    return coefficients


def predict_model(coeffs, x_values):
    """
    Calculate the predicted target values for the given model coefficients and given feature values

    :param coeffs: the coefficients of the trained model to use
    :param x_values: the feature matrix to use
    :return: the model predicted target values for a given feature matrix
    """
    # calculate the product of the coefficients and training features element-wise and then sum them row-wise
    return (coeffs * x_values).sum(axis=1, keepdims=False)


def calc_validation_err(coeffs: np.ndarray, x_valid: np.ndarray, t_valid: np.ndarray) -> float:
    """
    Calculate the cross validation error (MSE) for a given model's coefficients

    :param coeffs: the coefficients of the trained model to evaluate
    :param x_valid: the validation set feature matrix
    :param t_valid: the validation set target column vector
    """
    # get the model predictions for the fold validation feature matrix before subtracting them by the validation targets
    element_delta = predict_model(coeffs, x_valid) - t_valid

    # sum each element in the element delta column vector together, and divide by the number of elements
    validation_error = np.square(element_delta).mean()

    return validation_error


def perform_cross_validation(kf, x_features, y_train):
    cross_valid_score = 0.0

    # Run through all num K_FOLDS cross-validation
    for train, test in kf.split(x_features):
        x_train_mat, x_test_mat = x_features[train], x_features[test]
        y_train_mat, y_test_mat = y_train[train], y_train[test]

        coeffs = train_model(x_train_mat, y_train_mat)

        cross_valid_score += calc_validation_err(coeffs, x_test_mat, y_test_mat)

    # Average final cross-validation error
    return cross_valid_score / kf.n_splits


def find_best_feature(kf, remaining_features, selected_features, X_train, y_train):
    # Initialize cross-validation scores array to 0
    cv_scores = dict.fromkeys(remaining_features.keys(), 0.0)

    for feature_name in remaining_features.keys():
        # Generate k-folds from training set with specific features (selected + current)
        x_features = np.c_[np.ones(X_train.shape[0]), X_train[selected_features], X_train[feature_name]]

        cv_scores[feature_name] = perform_cross_validation(kf, x_features, y_train)

    # Find best feature, as the one with the lowest cross-validation error
    best_feature = min(cv_scores, key=cv_scores.get)

    return best_feature, cv_scores


def create_poly_regression_model(degree, x_train, y_train):
    """Creates a polynomial regression model for the given degree"""
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)

    # transforms the existing features to higher degree features
    x_train_poly = poly_features.fit_transform(x_train)

    # fit the transformed features to Linear Regression
    poly_model = LinearRegression(fit_intercept=False)
    poly_model.fit(x_train_poly, y_train)

    return poly_features, poly_model


def create_log_regression_model(x_train, y_train):
    """Creates a logarithmic regression model"""
    transformer = FunctionTransformer(np.log, validate=True)

    # transforms the existing features to logarithmic features
    x_train_log = transformer.fit_transform(x_train)

    # fit the transformed features to Linear Regression
    log_model = LinearRegression(fit_intercept=False)
    log_model.fit(x_train_log, y_train)

    return transformer, log_model


def get_poly_model(deg, x_train, y_train, x_test):
    poly_features, poly_model = create_poly_regression_model(deg, x_train, y_train)
    x_test_poly_mat = poly_features.fit_transform(x_test)
    coeffs = poly_model.coef_

    return coeffs, x_test_poly_mat


def get_log_model(x_train, y_train, x_test):
    # Add a small value to fix log(0) errors
    transformer, log_model = create_log_regression_model(x_train + 1e-8, y_train)
    x_test_log_mat = transformer.fit_transform(x_test + 1e-8)
    coeffs = log_model.coef_

    return coeffs, x_test_log_mat


def perform_basis_cross_validation(kf, x_features, y_train, model_fn):
    cross_valid_score = 0.0

    # Run through all num K_FOLDS cross-validation
    for train, test in kf.split(x_features):
        x_train_mat, x_test_mat = x_features[train], x_features[test]
        y_train_mat, y_test_mat = y_train[train], y_train[test]

        coeffs, x_test_fit_mat = model_fn(x_train_mat, y_train_mat, x_test_mat)

        cross_valid_score += calc_validation_err(coeffs, x_test_fit_mat, y_test_mat)

    # Average final cross-validation error
    return cross_valid_score / kf.n_splits


def process_base_model_subset(x_train, y_train, x_test, y_test):
    coeffs = train_model(x_train, y_train)

    valid_error = calc_validation_err(coeffs, x_test, y_test)

    return valid_error


def process_poly_model_subset(deg, kf, x_train, y_train, x_test, y_test):
    coeffs, x_test_poly_mat = get_poly_model(deg, x_train, y_train, x_test)

    # Calculate test error
    valid_error = calc_validation_err(coeffs, x_test_poly_mat, y_test)

    # Calculate cross validation error
    cv_error = perform_basis_cross_validation(kf, x_train, y_train, lambda x, y, z: get_poly_model(deg, x, y, z))

    return valid_error, cv_error


def process_log_model_subset(kf, x_train, y_train, x_test, y_test):
    coeffs, x_test_log_mat = get_log_model(x_train, y_train, x_test)

    # Calculate test error
    valid_error = calc_validation_err(coeffs, x_test_log_mat, y_test)

    # Calculate cross-validation error
    cv_error = perform_basis_cross_validation(kf, x_train, y_train, get_log_model)

    return valid_error, cv_error


def plot_error_curves(k, base_cv, base_test, basis_cv, basis_test):
    # create a new figure per model
    fig = plt.figure()
    fig.suptitle(f"Errors vs k")

    # plot the training and validation data
    plt.plot(k, base_cv, 'o-', color="blue", label="Base CV Error", mfc="none")
    plt.plot(k, base_test, 'o-', color="red", label="Base Test Error", mfc="none")
    plt.plot(k, basis_cv, 'o-', color="lightblue", label="Basis CV Error", mfc="none")
    plt.plot(k, basis_test, 'o-', color="red", label="Basis Test Error", mfc="none")

    # show the figure
    plt.legend(loc="best")
    plt.savefig("errors_vs_k.png")


def main():
    K_FOLDS = 5  # use 5 folds
    RANDOM_SEED = 637
    np.random.seed(RANDOM_SEED)  # student number is 400190637
    np.set_printoptions(linewidth=10000)

    # Load dataset, ignore FutureWarning about Boston dataset
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        dataset = load_boston()

    X_train, X_test, y_train, y_test = get_dataset_splits(dataset, RANDOM_SEED)

    # Keep track of features
    remaining_features = dict(zip(dataset.feature_names, range(len(dataset.feature_names))))
    selected_features = []

    # Initialize K-Fold splitter
    kf = KFold(n_splits=K_FOLDS)

    results = {"k": [], "base_cv_err": [], "base_test_err": [], "basis_cv_err": [], "basis_test_err": []}

    for k in range(1, len(dataset.feature_names) + 1):
        # Find the best feature to add to the selected set
        best_feature, cv_scores = find_best_feature(kf, remaining_features, selected_features, X_train, y_train)

        print(f"For k={k}, the cross-validation errors were: {cv_scores}")
        print(f"The best feature was: {best_feature}, x-val err: {cv_scores[best_feature]}")

        # Remove best feature and append to selected set
        selected_features.append(best_feature)
        del remaining_features[best_feature]

        # Train model across only selected subset training dataset
        x_train_mat = np.c_[np.ones(X_train.shape[0]), X_train[selected_features]]
        x_test_mat = np.c_[np.ones(X_test.shape[0]), X_test[selected_features]]

        base_test_error = process_base_model_subset(x_train_mat, y_train, x_test_mat, y_test)
        print(f"The selected subset test error was: {base_test_error}")

        # Do basis expansion
        # Model 1: Polynomial model of degree 2
        poly_test_error, poly_cv_error = process_poly_model_subset(2, kf, x_train_mat, y_train, x_test_mat, y_test)
        print(f"The selected subset polynomial model test error was: {poly_test_error}")
        print(f"The selected subset polynomial model x-val error was: {poly_cv_error}")

        # Model 2: Logarithmic model
        log_test_error, log_cv_error = process_log_model_subset(kf, x_train_mat, y_train, x_test_mat, y_test)
        print(f"The selected subset logarithmic model test error was: {log_test_error}")
        print(f"The selected subset logarithmic model x-val error was: {log_cv_error}")
        print()

        results["k"].append(k)
        results["base_cv_err"].append(cv_scores[best_feature])
        results["base_test_err"].append(base_test_error)
        if poly_cv_error < log_cv_error:
            results["basis_cv_err"].append(poly_cv_error)
            results["basis_test_err"].append(poly_test_error)
        else:
            results["basis_cv_err"].append(log_cv_error)
            results["basis_test_err"].append(log_test_error)

    plot_error_curves(results["k"], results["base_cv_err"], results["base_test_err"], results["basis_cv_err"],
                      results["basis_test_err"])
    plt.show()


if __name__ == '__main__':
    main()
