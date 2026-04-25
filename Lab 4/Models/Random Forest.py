"""
Random Forest Model Module
This module contains the function to initialize, tune, and train a 
Random Forest Regressor using Grid Search Cross-Validation.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest model using GridSearchCV to find the optimal hyperparameters.

    Parameters:
    X_train (DataFrame): The training features.
    y_train (Series/Array): The training target values.

    Returns:
    best_model: The fully trained Random Forest model with the best hyperparameters.
    """
    print("Starting Grid Search for Random Forest...")
    
    # Initialize the base model and cross-validation strategy
    rf_model = RandomForestRegressor(random_state=42)
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    
    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Set up the Grid Search
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1 
    )
    
    # Execute the Grid Search (this handles the training)
    grid_search.fit(X_train, y_train)
    
    # Extract the winning model and parameters
    best_model = grid_search.best_estimator_
    print(f"Random Forest training complete.")
    print(f"Best Parameters Found: {grid_search.best_params_}")
    
    return best_model