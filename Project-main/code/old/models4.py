from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_logistic_regression(random_state=42):
    """
    Tweak: We added C=5.0. This decreases the regularization, allowing the model 
    to fit closer to the training data and pay attention to more nuanced words.
    We also increased max_iter so it doesn't time out while doing the extra math.
    """
    return LogisticRegression(C=5.0, max_iter=2000, random_state=random_state)

def get_random_forest(random_state=42):
    """
    Tweak: We increased n_estimators from 100 to 300. This builds 300 distinct 
    decision trees instead of 100, creating a much stronger 'wisdom of the crowd'.
    """
    return RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=random_state)

def get_mlp(random_state=42):
    """
    Tweak: We added a second hidden layer! It now passes the data through 100 neurons, 
    and then another 50 neurons before guessing. This allows the neural network to 
    learn highly complex relationships between word pairs.
    """
    return MLPClassifier(
        hidden_layer_sizes=(100, 50), 
        max_iter=30, 
        early_stopping=True, 
        random_state=random_state
    )