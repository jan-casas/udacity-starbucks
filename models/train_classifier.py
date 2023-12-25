from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import pandas as pd


def fit_clf(model, param_grid, X_train, y_train):
    """
    Fits a classifier model using grid search with cross-validation.

    Parameters:
        model (object): The classifier model to be trained.
        param_grid (dict): The parameter grid for grid search.
        X_train (array-like): The training data features.
        y_train (array-like): The training data labels.

    Returns:
            tuple: A tuple containing the best f1 score and the best estimator model.
    """
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        scoring='f1', cv=5, verbose=0)
    print(f"Training {model.__class__.__name__} :")
    grid.fit(X_train, y_train)

    print(f"{model.__class__.__name__}\nBest f1_score : {round(grid.best_score_, 4)}\n{'*'*40}")

    return grid.best_score_, grid.best_estimator_


def train_and_evaluate_classifiers(classifiers, X_train, y_train):
    """
    Trains and evaluates multiple classifiers using the given training data.

    Parameters:
        classifiers (dict): A dictionary containing the classifiers as keys and their corresponding parameters as values.
        X_train (array-like): The training data features.
        y_train (array-like): The training data labels.

    Returns:
    clf_df (DataFrame): A pandas DataFrame containing the best F1 scores and corresponding best estimators for each classifier.
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

    clf_dict['best_f1_score'] = clf_scores
    clf_dict['best_est'] = clf_best_ests
    clf_df = pd.DataFrame(clf_dict, index=clf_names)

    return clf_df
