from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd


def fit_clf(clf, param_grid, X_train, y_train):
    """
    Fits a classifier using RandomizedSearchCV with 5-fold cross-validation.

    Parameters:
                clf: The classifier object to be fitted.
                param_grid: The parameter grid for RandomizedSearchCV.
                X_train: The training data features.
                y_train: The training data labels.

    Returns:
                best_score: The score of the best model.
                best_est: The parameters of the best model.
    """
    # Initialize RandomizedSearchCV with 5-fold cross-validation
    random_search = RandomizedSearchCV(
        clf, param_grid, cv=5, n_jobs=-1, n_iter=10, verbose=3)

    # Fit RandomizedSearchCV to the training data
    random_search.fit(X_train, y_train)

    # Get the score of the best model and the parameters of the best model
    best_score = random_search.best_score_
    best_est = random_search.best_estimator_

    return best_score, best_est


def train_and_evaluate_classifiers(classifiers, X_train, y_train):
    """
    Trains and evaluates multiple classifiers using the given training data.

    Parameters:
                classifiers (dict): A dictionary containing the classifiers as keys and their corresponding parameters as values.
                X_train (array-like): The training data features.
                y_train (array-like): The training data labels.

    Returns:
        clf_df (DataFrame): A pandas DataFrame containing the best scores and estimators for each classifier.
    """
    clf_names = []
    clf_scores = []
    clf_best_ests = []
    clf_dict = {}

    for classifier, params in classifiers.items():
        best_score, best_est = fit_clf(classifier, params, X_train, y_train)
        clf_names.append(classifier.__class__.__name__)
        clf_scores.append(best_score)
        clf_best_ests.append(best_est)

    clf_dict['best_score'] = clf_scores
    clf_dict['best_est'] = clf_best_ests
    clf_df = pd.DataFrame(clf_dict, index=clf_names)

    return clf_df
