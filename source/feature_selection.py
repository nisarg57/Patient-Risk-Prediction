import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def select_features_kbest(X_processed, y, k=10):
    if y is None:
        raise ValueError("Target variable 'y' cannot be None for feature selection.")

    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X_processed, y)

    return X_new, selector

def select_features_from_model(X_processed, y, estimator=None, threshold='median'):
    if y is None:
        raise ValueError("Target variable 'y' cannot be None for feature selection.")

    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the estimator to get feature importances
    estimator.fit(X_processed, y)

    # Select features based on importance
    selector = SelectFromModel(estimator, threshold=threshold, prefit=True)
    X_new = selector.transform(X_processed)

    return X_new, selector

