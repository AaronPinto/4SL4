import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures


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


def calculate_validation_error(coeffs: np.ndarray, x_valid: np.ndarray, t_valid: np.ndarray) -> float:
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


def find_best_feature(kf, remaining_features, selected_features, X_train, y_train):
    # Initialize cross-validation scores array to 0
    cv_scores = dict.fromkeys(remaining_features.keys(), 0.0)

    for feature_name in remaining_features.keys():
        # Generate k-folds from training set with specific features (selected + current)
        x_features = np.c_[np.ones(X_train.shape[0]), X_train[selected_features], X_train[feature_name]]

        cross_valid_score = 0.0

        # Run through all num K_FOLDS cross-validation
        for train, test in kf.split(x_features):
            x_train_mat, x_test_mat = x_features[train], x_features[test]
            y_train_mat, y_test_mat = y_train[train], y_train[test]

            coeffs = train_model(x_train_mat, y_train_mat)

            cross_valid_score += calculate_validation_error(coeffs, x_test_mat, y_test_mat)

        # Average final cross-validation error
        cv_scores[feature_name] = cross_valid_score / kf.n_splits

    # Find best feature, as the one with the lowest cross-validation error
    best_feature = min(cv_scores, key=cv_scores.get)

    return best_feature, cv_scores


def create_polynomial_regression_model(degree, x_train, y_train):
    """Creates a polynomial regression model for the given degree"""
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)

    # transforms the existing features to higher degree features.
    x_train_poly = poly_features.fit_transform(x_train)

    # fit the transformed features to Linear Regression
    poly_model = LinearRegression(fit_intercept=False)
    poly_model.fit(x_train_poly, y_train)

    return poly_features, poly_model


def main():
    K_FOLDS = 5  # use 5 folds
    RANDOM_SEED = 637
    np.random.seed(RANDOM_SEED)  # student number is 400190637

    # Load dataset
    dataset = load_boston()
    X_train, X_test, y_train, y_test = get_dataset_splits(dataset, RANDOM_SEED)

    # Keep track of features
    remaining_features = dict(zip(dataset.feature_names, range(len(dataset.feature_names))))
    selected_features = []

    # Initialize K-Fold splitter
    kf = KFold(n_splits=K_FOLDS)

    for k in range(1, len(dataset.feature_names) + 1):
        # Find the best feature to add to the selected set
        best_feature, cv_scores = find_best_feature(kf, remaining_features, selected_features, X_train, y_train)

        print(f"For k={k}, the best feature was: {best_feature}")
        print(f"The cross-validation errors were: {cv_scores}")

        # Remove best feature and append to selected set
        selected_features.append(best_feature)
        del remaining_features[best_feature]

        # Train model across only selected subset training dataset
        x_train_mat = np.c_[np.ones(X_train.shape[0]), X_train[selected_features]]
        x_test_mat = np.c_[np.ones(X_test.shape[0]), X_test[selected_features]]

        coeffs = train_model(x_train_mat, y_train)

        valid_error = calculate_validation_error(coeffs, x_test_mat, y_test)
        print(f"The selected subset test error was: {valid_error}")

        # Do basis expansion
        # Model 1: Polynomial model of degree 2
        poly_features, poly_model = create_polynomial_regression_model(2, x_train_mat, y_train)
        x_test_mat = poly_features.fit_transform(x_test_mat)
        coeffs = poly_model.coef_

        valid_error = calculate_validation_error(coeffs, x_test_mat, y_test)
        print(f"The selected subset polynomial model test error was: {valid_error}")
        print()


if __name__ == '__main__':
    main()
