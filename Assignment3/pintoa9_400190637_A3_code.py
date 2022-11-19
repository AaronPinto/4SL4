import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, f1_score, \
    mean_absolute_error, \
    precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


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


def calc_misclassification_rate(predictions, y_test_mat):
    return np.mean(predictions != y_test_mat)


def logistic_loss(y, pred_proba, eps=1e-15):
    # Log loss is undefined for p=0 or p=1, so clip probabilities
    pred_proba = np.fmax(eps, np.fmin(1.0 - eps, pred_proba))

    return -np.mean(y * np.log(pred_proba) - (1.0 - y) * np.log(1.0 - pred_proba))


def train_logistic_regression(x_train, y_train, tolerance, lr):
    """
    Perform batch gradient descent to train the logistic regression model

    :param x_train: the training input
    :param y_train: the true values
    :param tolerance: the threshold at which to stop training
    :param lr: the learning rate
    :return: the weights and bias for the model
    """
    # initialize weights and bias to zero
    weights = np.zeros(x_train.shape[1])

    while True:
        # calculate predictions
        preds = predict_log_reg_proba(x_train, weights)

        # update the params
        grad = (lr / x_train.shape[0]) * x_train.T @ (preds - y_train)

        weights -= grad

        # calculate the loss
        loss = logistic_loss(y_train, predict_log_reg_proba(x_train, weights))

        if loss <= tolerance:
            break

    return weights


def predict_log_reg_proba(x_test, weights):
    # calculate prediction probabilities
    return expit(x_test @ weights)


def predict_log_reg(x_test, weights, threshold=0.5):
    preds = predict_log_reg_proba(x_test, weights)

    # split classification at threshold
    pred_class = [1 if i > threshold else 0 for i in preds]

    return np.array(pred_class)


def calc_pr_curve(y_test, probas_pred):
    precisions, recalls, thresholds = [], [], []

    sorted_probs = sorted(probas_pred)

    # calculate the precision and recall score for each threshold
    for thresh in sorted_probs:
        precisions.append(precision_score(y_test, probas_pred > thresh, zero_division=1))
        recalls.append(recall_score(y_test, probas_pred > thresh))

    return np.array(precisions), np.array(recalls), np.array(probas_pred)


def plot_pr_curve(precision, recall, title, filename):
    disp = PrecisionRecallDisplay(precision, recall)
    disp.plot()
    disp.ax_.set_title(title)
    disp.figure_.savefig(filename)


def predict_knn(k, x_train_mat, x_test_mat, y_train_mat):
    predictions = []

    # For each row in the test set, calculate the distance between it and each row in the training set,
    # and pick the K rows with the smallest distances between them and the test row
    for x in x_test_mat:
        distances = np.linalg.norm(x_train_mat - x, axis=1)
        neighbours = np.argsort(distances, kind='stable')[0:k]
        classes = list(y_train_mat[neighbours])
        predictions.append(max(set(classes), key=classes.count))

    return predictions


def perform_cross_validation(k, kf, x_train, y_train):
    cross_valid_score = 0.0

    # Run through all num K_FOLDS cross-validation
    for train, test in kf.split(x_train):
        x_train_mat, x_test_mat = x_train[train], x_train[test]
        y_train_mat, y_test_mat = y_train[train], y_train[test]

        predictions = predict_knn(k, x_train_mat, x_test_mat, y_train_mat)

        cross_valid_score += calc_misclassification_rate(predictions, y_test_mat)

    # Average final cross-validation error
    return cross_valid_score / kf.n_splits


def main():
    K_FOLDS = 5  # use 5 folds
    RANDOM_SEED = 637
    np.random.seed(RANDOM_SEED)  # student number is 400190637
    np.set_printoptions(linewidth=10000)
    dataset = load_breast_cancer()

    x_train, x_test, y_train, y_test = get_dataset_splits(dataset, RANDOM_SEED)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # Initialize K-Fold splitter
    kf = KFold(n_splits=K_FOLDS)

    """logistic regression"""
    # My implementation
    x_train_mat = np.c_[np.ones(x_train.shape[0]), x_train]
    x_test_mat = np.c_[np.ones(x_test.shape[0]), x_test]
    log_reg_weights = train_logistic_regression(x_train_mat, y_train, 1e-4, 0.1)

    predictions = predict_log_reg(x_test_mat, log_reg_weights)

    best_misclass_rate = calc_misclassification_rate(predictions, y_test)
    f1_s = f1_score(y_test, predictions)
    print(f"The test error for my implementation is: {best_misclass_rate}, the f1-score is: {f1_s}")

    probas_pred = predict_log_reg_proba(x_test_mat, log_reg_weights)
    precision, recall, thresholds = calc_pr_curve(y_test, probas_pred)
    plot_pr_curve(precision, recall, "2-class Precision-Recall curve my impl", "pr_curve_my_impl.png")

    # Scikit learn implementation
    log_reg_clf = LogisticRegression(random_state=RANDOM_SEED)
    log_reg_clf.fit(x_train, y_train)

    predictions = log_reg_clf.predict(x_test)

    best_misclass_rate = mean_absolute_error(y_test, predictions)
    f1_s = f1_score(y_test, predictions)
    print(f"The test error for the sklearn implementation is: {best_misclass_rate}, the f1-score is: {f1_s}")

    probas_pred = log_reg_clf.predict_proba(x_test)[:, 0]
    precision, recall, thresholds = precision_recall_curve(y_test, probas_pred, pos_label=0)
    plot_pr_curve(precision, recall, "2-class Precision-Recall curve sklearn impl", "pr_curve_sklearn.png")
    plt.show()

    print("\n")

    """k-nearest neighbour classifier"""
    cv_scores = []
    for k in range(1, 6):  # number of neighbours
        # My implementation
        cv_score = perform_cross_validation(k, kf, x_train, y_train)
        cv_scores.append(cv_score)
        print(f"For k={k}, the cross-validation error was: {cv_score}")

        # Scikit learn implementation
        knn_clf = KNeighborsClassifier(n_neighbors=k)
        skl_cv_score = -cross_val_score(knn_clf, x_train, y_train, cv=kf, scoring="neg_mean_absolute_error").mean()
        print(f"The sklearn cv error was: {skl_cv_score}")

    print()
    best_k = np.argmin(cv_scores) + 1
    print(f"The best model was at k={best_k}")

    # My implementation
    predictions = predict_knn(best_k, x_train, x_test, y_train)

    best_misclass_rate = calc_misclassification_rate(predictions, y_test)
    f1_s = f1_score(y_test, predictions)
    print(f"The test error for my implementation is: {best_misclass_rate}, the f1-score is: {f1_s}")

    # Scikit learn implementation
    knn_clf = KNeighborsClassifier(n_neighbors=best_k)
    knn_clf.fit(x_train, y_train)

    predictions = knn_clf.predict(x_test)

    best_misclass_rate = mean_absolute_error(y_test, predictions)
    f1_s = f1_score(y_test, predictions)
    print(f"The test error for the sklearn implementation is: {best_misclass_rate}, the f1-score is: {f1_s}")


if __name__ == '__main__':
    main()
