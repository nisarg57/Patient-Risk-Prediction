import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def select_features_kbest(X_processed, y, k=10):
    """
    Selects the top K best features using SelectKBest with f_classif.
    X_processed should be a numpy array or pandas DataFrame of features.
    """
    if y is None:
        raise ValueError("Target variable 'y' cannot be None for feature selection.")

    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X_processed, y)
    
    # You might need to map back to original feature names for interpretability.
    # This is more complex if X_processed came from a ColumnTransformer with OneHotEncoder.
    # For now, we return the transformed X.

    return X_new, selector

def select_features_from_model(X_processed, y, estimator=None, threshold='median'):
    """
    Selects features based on importance from a fitted estimator (e.g., RandomForestClassifier).
    """
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

if __name__ == '__main__':
    # Example Usage (replace with your actual preprocessed data)
    from sklearn.datasets import make_classification
    X_dummy, y_dummy = make_classification(n_samples=100, n_features=20, n_informative=10, n_redundant=5, random_state=42)

    print("Original X shape:", X_dummy.shape)

    # Using SelectKBest
    X_selected_kbest, selector_kbest = select_features_kbest(X_dummy, y_dummy, k=5)
    print("X shape after SelectKBest:", X_selected_kbest.shape)
    # print("Selected feature indices (KBest):", selector_kbest.get_support(indices=True))

    # Using SelectFromModel with RandomForest
    X_selected_rf, selector_rf = select_features_from_model(X_dummy, y_dummy)
    print("X shape after SelectFromModel (RandomForest):", X_selected_rf.shape)
    # print("Selected feature indices (RF):", selector_rf.get_support(indices=True))
