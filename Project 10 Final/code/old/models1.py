from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_logistic_regression(random_state=42):
    """Initializes a Logistic Regression model."""
    return LogisticRegression(max_iter=1000, random_state=random_state)

def get_random_forest(random_state=42):
    """Initializes a Random Forest model."""
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=random_state)

def get_mlp(random_state=42):
    """Initializes a simple Multi-Layer Perceptron (Neural Network)."""
    # 1 hidden layer with 50 neurons is enough for a baseline text classifier
    return MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, early_stopping=True, random_state=random_state)