import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # ---------------------------------------------------------
    # 1. DATA PREPARATION & SPLIT
    # ---------------------------------------------------------
    # Load the diabetes dataset
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    y = diabetes.target

    # Split the data: 80% train, 20% test (as defined in the report)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42 # 42 = Answer to the Ultimate Question of Life, the Universe, and Everything." :D
    )
    
    # ---------------------------------------------------------
    # 2. DATASET EXPORT TO CSV
    # ---------------------------------------------------------
    # Create the 'Dataset' directory if it doesn't already exist
    os.makedirs('Dataset', exist_ok=True)

    # Combine training features and target, then save to CSV
    train_df = X_train.copy()
    train_df['target_disease_progression'] = y_train
    train_df.to_csv('Dataset/Train.csv', index=False)

    # Combine testing features and target, then save to CSV
    test_df = X_test.copy()
    test_df['target_disease_progression'] = y_test
    test_df.to_csv('Dataset/Test.csv', index=False)

    print("CSV files 'Train.csv' and 'Test.csv' successfully saved in the 'Dataset' folder.\n")

    # Setup 4-fold cross-validation
    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    # ---------------------------------------------------------
    # 3. MODEL DEFINITION & TRAINING: Linear Regression
    # ---------------------------------------------------------
    print("--- Linear Regression ---")
    lr_model = LinearRegression()
    
    # Evaluate using 4-fold CV on training data
    lr_cv_results = cross_validate(
        lr_model, X_train, y_train, cv=kf,
        scoring=('neg_root_mean_squared_error', 'r2')
    )
    
    # Calculate average CV metrics
    lr_cv_rmse = -np.mean(lr_cv_results['test_neg_root_mean_squared_error'])
    lr_cv_r2 = np.mean(lr_cv_results['test_r2'])
    print(f"CV RMSE: {lr_cv_rmse:.2f}")
    print(f"CV R2: {lr_cv_r2:.4f}")

    # Train final LR model on the full training set
    lr_model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 4. MODEL DEFINITION & TRAINING: Random Forest
    # ---------------------------------------------------------
    print("\n--- Random Forest Regressor ---")
    rf_model = RandomForestRegressor(random_state=42)
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Perform Grid Search with 4-fold CV
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=kf,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1 
    )
    
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_rf_cv_rmse = -grid_search.best_score_
    
    print(f"Best Parameters: {best_params}")
    print(f"Best CV RMSE: {best_rf_cv_rmse:.2f}")

    # ---------------------------------------------------------
    # 5. FINAL EVALUATION ON TEST SET
    # ---------------------------------------------------------
    print("\n--- Test Set Evaluation ---")
    
    # Linear Regression Test Metrics
    lr_predictions = lr_model.predict(X_test)
    lr_test_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
    lr_test_r2 = r2_score(y_test, lr_predictions)
    
    print(f"Linear Regression - Test RMSE: {lr_test_rmse:.2f}, Test R2: {lr_test_r2:.4f}")

    # Random Forest Test Metrics
    rf_predictions = best_rf.predict(X_test)
    rf_test_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
    rf_test_r2 = r2_score(y_test, rf_predictions)
    
    print(f"Random Forest - Test RMSE: {rf_test_rmse:.2f}, Test R2: {rf_test_r2:.4f}")

if __name__ == "__main__":
    main()