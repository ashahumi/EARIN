import os
import sys
import numpy as np
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

from data_loader import load_and_split_data

class DualLogger:
    """
    Acts as a splitter: writes to both the original terminal and the text file simultaneously.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def get_next_output_filename():
    """Finds the next available output_x.txt filename, prefixed for BERT."""
    i = 1
    while os.path.exists(f"output_bert_{i}.txt"):
        i += 1
    return f"output_bert_{i}.txt"

def main():
    # 1. Setup Auto-Incrementing File Logging
    out_filename = get_next_output_filename()
    sys.stdout = DualLogger(out_filename)

    print("="*60)
    print("HYPERPARAMETER OPTIMIZATION (Grid Search)")
    print("Vector Method: Pre-computed BERT Embeddings (Dense)")
    print(f"[SYSTEM] Live output is also being saved to: {out_filename}")
    print("="*60 + "\n")

    # 2. Load the Raw Data to get the Targets (y)
    print("Loading dataset labels...")
    dataset_path = Path(__file__).resolve().parent.parent / "dataset" / "my_50k_reviews.csv"
    _, _, y_train, y_test = load_and_split_data(
        dataset_path, text_col='review/text', target_col='review/score'
    )

    # 3. Load the Pre-computed BERT Vectors (X)
    # NOTE: Ensure these file paths exactly match where your .npy files are saved!
    print("Loading pre-computed BERT embeddings from disk...")
    bert_train_path = "bert_train.npy"
    bert_test_path = "bert_test.npy"
    
    if not os.path.exists(bert_train_path) or not os.path.exists(bert_test_path):
        print(f"\n[ERROR] Could not find {bert_train_path} or {bert_test_path}.")
        print("Please update the paths in the code to point to your saved BERT .npy files.")
        sys.exit(1)

    X_train = np.load(bert_train_path)
    X_test = np.load(bert_test_path)
    
    print(f"BERT Training Data Shape: {X_train.shape}")

    # 4. Define the Model (No pipeline needed since data is pre-vectorized)
    mlp = MLPClassifier(early_stopping=True, max_iter=50, random_state=42)

    # 5. Define the Parameter Grid for Dense BERT Vectors
    # We provide deeper network options because BERT vectors are dense and complex
    param_grid = {
        'hidden_layer_sizes': [(100,), (256, 128), (128, 64)], 
        'alpha': [0.01, 0.05, 0.1],                
        'activation': ['tanh', 'relu']             
    }

    print("\nInitializing GridSearchCV...")
    print(f"Grid configurations to test: {len(param_grid['hidden_layer_sizes']) * len(param_grid['alpha']) * len(param_grid['activation'])}")
    print("Cross-validation folds: 3")
    print("--------------------------------------------------\n")
    
    # Using n_jobs=1 to ensure terminal output is completely visible
    grid_search = GridSearchCV(
        mlp, 
        param_grid, 
        cv=3, 
        n_jobs=-1, 
        verbose=3 
    )
    
    grid_search.fit(X_train, y_train)

    # 6. Output the Best Results
    print("\n" + "="*60)
    print("GRID SEARCH COMPLETED (BERT)!")
    print("="*60)
    print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
    print("\nBEST PARAMETERS FOUND:")
    for param, value in grid_search.best_params_.items():
        print(f" - {param}: {value}")
        
    print("\n" + "="*60)
    print("FINAL EVALUATION ON UNSEEN TEST DATA (BERT)")
    print("="*60)
    
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    print(f"\nFinal Test Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    main()