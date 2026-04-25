"""
Linear Regression Model Module
This module contains the function to initialize and train a standard 
Linear Regression model for the diabetes progression dataset.
"""

from sklearn.linear_model import LinearRegression

def train_linear_regression(X_train, y_train):
    """
    Trains a standard Linear Regression model on the provided training data.
    Because it uses Ordinary Least Squares, it does not require hyperparameter tuning.

    Parameters:
    X_train (DataFrame): The training features.
    y_train (Series/Array): The training target values.

    Returns:
    model: The fully trained Linear Regression model.
    """
    print("Training standard Linear Regression model...")
    
    # Initialize the model
    model = LinearRegression()
    
    # Train the model on the data
    model.fit(X_train, y_train)
    
    print("Linear Regression training complete.")
    
    return model