from sklearn.neural_network import MLPClassifier

def get_optimized_mlp(random_state=42):
    """
    Returns an MLPClassifier instantiated with the globally optimal
    hyperparameters discovered during Grid Search.
    """
    return MLPClassifier(
        hidden_layer_sizes=(100,), # Single layer proved optimal for sparse matrices
        activation='relu',         # Modern standard for smooth gradient flow
        alpha=0.1,                 # High L2 regularization penalty to prevent overfitting
        max_iter=40,               # Maximizes convergence speed
        early_stopping=True,       # Prevents waste of compute cycles
        random_state=random_state
    )