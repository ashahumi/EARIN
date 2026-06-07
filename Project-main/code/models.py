from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_logistic_regression(random_state=42):
    """
    Tweak: L1 penalty to automatically drop useless words, 
    using 'saga' solver which handles L1.
    """
    return LogisticRegression(
        l1_ratio=1,    
        solver='saga', 
        C=1.0, 
        max_iter=2000, 
        random_state=random_state
    )

def get_random_forest(random_state=42):
    """
    Tweak: Constrained Forest to fight overfitting (max_depth and min_samples_split).
    """
    return RandomForestClassifier(
        n_estimators=300, 
        max_depth=50,          
        min_samples_split=10,  
        n_jobs=-1, 
        random_state=random_state
    )

def get_mlp(random_state=42):
    """
    Tweak: Used 'tanh' activation and higher 'alpha' to penalize complex weights 
    and reduce overfitting.
    """
    return MLPClassifier(
        hidden_layer_sizes=(100, 50), 
        activation='tanh', 
        alpha=0.05,        
        max_iter=30, 
        early_stopping=True, 
        random_state=random_state
    )