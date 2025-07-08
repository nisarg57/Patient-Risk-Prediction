from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

def train_model(X_train, y_train, model_name='LogisticRegression', random_state=42):
    """
    Trains a specified machine learning model.
    """
    if model_name == 'LogisticRegression':
        model = LogisticRegression(random_state=random_state, solver='liblinear')
    elif model_name == 'DecisionTree':
        model = DecisionTreeClassifier(random_state=random_state)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=random_state)
    elif model_name == 'SVC':
        model = SVC(probability=True, random_state=random_state) # probability=True for ROC AUC
    else:
        raise ValueError(f"Model '{model_name}' not supported.")

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model and returns common classification metrics.
    """
    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0)
    }

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
    except AttributeError:
        # Some models (e.g., SVC without probability=True) don't have predict_proba
        metrics['roc_auc'] = None

    return metrics

def save_model(model, filepath):
    """Saves the trained model to a file."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Loads a trained model from a file."""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model