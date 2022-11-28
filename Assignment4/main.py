import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


def get_dataset_splits(dataset, rnd_seed):
    """
    Get training and testing splits from a given dataset

    :param dataset:
    :param rnd_seed:
    :return:
    """
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    return train_test_split(X, y, test_size=1.0 / 3.0, random_state=rnd_seed)


def calc_test_error(y_test, y_pred):
    return 1.0 - accuracy_score(y_test, y_pred)


def train_decision_tree(seed, k_folds, leaf_range, x_train, y_train, x_test, y_test):
    # Decision tree classifier
    params = {"max_leaf_nodes": leaf_range}
    grid = GridSearchCV(DecisionTreeClassifier(random_state=seed), params, cv=k_folds, n_jobs=-1)
    grid.fit(x_train, y_train)
    model = grid.best_estimator_

    y_pred = model.fit(x_train, y_train).predict(x_test)
    test_error = calc_test_error(y_test, y_pred)

    return grid.best_params_["max_leaf_nodes"], test_error, 1.0 - grid.cv_results_["mean_test_score"]


def train_bagging(seed, n_estimators, x_train, y_train, x_test, y_test):
    # Bagging classifier, defaults to using a DecisionTreeClassifier as the base
    clf = BaggingClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    test_error = calc_test_error(y_test, y_pred)

    return test_error


def train_random_forest(seed, n_estimators, x_train, y_train, x_test, y_test):
    # Random Forest classifier
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    test_error = calc_test_error(y_test, y_pred)

    return test_error


def train_adaboost(seed, n_estimators, base, x_train, y_train, x_test, y_test):
    # Adaboost classifier
    clf = AdaBoostClassifier(base_estimator=base, n_estimators=n_estimators, random_state=seed)
    clf.fit(x_train, y_train)

    test_errors = []

    for y_pred in tqdm(clf.staged_predict(x_test)):
        test_errors.append(calc_test_error(y_test, y_pred))

    return test_errors


def plot_test_errors(num_predictors, dtc_errors, bag_errors, rfc_errors, ada1_errors, ada2_errors, ada3_errors):
    # create a new figure
    plt.figure(figsize=(13, 13))
    plt.title("Test errors", fontsize=16)

    # axis labels
    plt.xlabel("Number of predictors")
    plt.ylabel("Test Error")

    colors = sns.color_palette("colorblind")

    # plot test errors
    plt.plot(num_predictors, dtc_errors, 'o-', color=colors[0], label="Decision Tree", mfc="none")
    plt.plot(num_predictors, bag_errors, 'o-', color=colors[1], label="Bagging", mfc="none")
    plt.plot(num_predictors, rfc_errors, 'o-', color=colors[2], label="Random Forest", mfc="none")
    plt.plot(num_predictors, ada1_errors, 'o-', color=colors[3], label="Adaboost 1", mfc="none")
    plt.plot(num_predictors, ada2_errors, 'o-', color=colors[4], label="Adaboost 2", mfc="none")
    plt.plot(num_predictors, ada3_errors, 'o-', color=colors[5], label="Adaboost 3", mfc="none")

    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("test_errors.png")


def plot_dtc_cross_valid_errors(num_leaves, cv_errors):
    # create a new figure
    plt.figure(figsize=(13, 13))
    plt.title(f"Decision Tree Classifier Cross-Validation Errors vs Max Number of Leaves", fontsize=16)

    # axis labels
    plt.xlabel("Max Number of Leaves")
    plt.ylabel("Cross-Validation Error")

    # plot the data
    plt.plot(num_leaves, cv_errors, 'k-', mfc="none")

    plt.savefig("errors_vs_leaves.png")


def main():
    K_FOLDS = 5  # use 5 folds
    RND_SEED = 637
    np.random.seed(RND_SEED)  # student number is 400190637
    np.set_printoptions(linewidth=1000)
    dataset = pd.read_csv("spambase/spambase.data", header=None)

    x_train, x_test, y_train, y_test = get_dataset_splits(dataset, RND_SEED)

    # Decision tree classifier
    print("Training decision tree classifier...")
    leafs = np.arange(2, 401)
    num_leaf, dt_test_error, scores = train_decision_tree(RND_SEED, K_FOLDS, leafs, x_train, y_train, x_test, y_test)
    print("Done!")
    print(f"The best test error of {dt_test_error}, was at max_leaf_nodes: {num_leaf}")

    END_NUM_CLASSIFIERS = 2500
    NUM_CLASSIFIERS = range(50, END_NUM_CLASSIFIERS + 1, 50)

    bag_test_errors = []
    rnd_forest_test_errors = []

    # Ensemble methods
    print("Training 50 bagging and random forest classifiers. This might take a while...")
    for n_preds in tqdm(NUM_CLASSIFIERS):
        bag_test_errors.append(train_bagging(RND_SEED, n_preds, x_train, y_train, x_test, y_test))
        rnd_forest_test_errors.append(train_random_forest(RND_SEED, n_preds, x_train, y_train, x_test, y_test))

    print("Done!")
    print("Training 50 Adaboost classifiers (x3). This might take a while...")
    base1 = DecisionTreeClassifier(max_depth=1, random_state=RND_SEED)
    base2 = DecisionTreeClassifier(max_leaf_nodes=10, random_state=RND_SEED)
    base3 = DecisionTreeClassifier(random_state=RND_SEED)

    # Train Adaboost classifiers, start at the 50th error, and take every 50th error from there to align with others
    ada1_test_errors = train_adaboost(RND_SEED, END_NUM_CLASSIFIERS, base1, x_train, y_train, x_test, y_test)[49::50]
    ada2_test_errors = train_adaboost(RND_SEED, END_NUM_CLASSIFIERS, base2, x_train, y_train, x_test, y_test)[49::50]
    ada3_test_errors = train_adaboost(RND_SEED, END_NUM_CLASSIFIERS, base3, x_train, y_train, x_test, y_test)[49::50]
    print("Done!")

    plot_test_errors(NUM_CLASSIFIERS, [dt_test_error] * len(ada1_test_errors), bag_test_errors, rnd_forest_test_errors,
                     ada1_test_errors, ada2_test_errors, ada3_test_errors)
    plot_dtc_cross_valid_errors(leafs, scores)
    plt.show()


if __name__ == '__main__':
    main()
