from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib # For saving/loading models

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

if __name__ == '__main__':
    # Example Usage (replace with your actual preprocessed and split data)
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=0, random_state=42)
    X_train_ex, X_test_ex, y_train_ex, y_test_ex = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Logistic Regression model...")
    lr_model = train_model(X_train_ex, y_train_ex, model_name='LogisticRegression')
    lr_metrics = evaluate_model(lr_model, X_test_ex, y_test_ex)
    print("Logistic Regression Metrics:", lr_metrics)

    print("\nTraining RandomForest model...")
    rf_model = train_model(X_train_ex, y_train_ex, model_name='RandomForest')
    rf_metrics = evaluate_model(rf_model, X_test_ex, y_test_ex)
    print("RandomForest Metrics:", rf_metrics)

    # Example of saving and loading
    save_model(rf_model, 'random_forest_model.pkl')
    loaded_rf_model = load_model('random_forest_model.pkl')
    # Can then use loaded_rf_model for predictions
