import os
import sys
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Import our custom final modules
from data_loader import load_and_split_data
from preprocessing_final import clean_text, get_best_vectorizer
from model_final import get_optimized_mlp

class DualLogger:
    """Writes output to both the terminal screen and the log text file simultaneously."""
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
    """Finds the next available output_final_x.txt filename."""
    i = 1
    while os.path.exists(f"output_final_{i}.txt"):
        i += 1
    return f"output_final_{i}.txt"

def main():
    # 1. Setup Auto-Incrementing Logger
    out_filename = get_next_output_filename()
    sys.stdout = DualLogger(out_filename)

    print("="*60)
    print("PROGRESSIVE LEARNING CURVE EXPERIMENT")
    print("Evaluating Model Robustness Across Varied Training Set Sizes")
    print(f"[SYSTEM] Full results are being saved to: {out_filename}")
    print("="*60 + "\n")

    # 2. Load Raw Data
    print("Loading master dataset...")
    script_dir = Path(__file__).resolve().parent
    dataset_path = script_dir.parent / "dataset" / "my_50k_reviews.csv"
    X_train_raw, X_test_raw, y_train, y_test = load_and_split_data(
        dataset_path, text_col='review/text', target_col='review/score'
    )
    
    print("Applying text cleaning step...")
    X_train_clean = X_train_raw.apply(clean_text)
    X_test_clean = X_test_raw.apply(clean_text)
    print(f"Master Training Pool Size: {len(X_train_clean)}")
    print(f"Fixed Unseen Test Evaluation Set Size: {len(X_test_clean)}\n")

    # 3. Define the Progressive Data Sizes (From 40k down to 5k stepping down by 5k)
    sample_sizes = [40000, 35000, 30000, 25000, 20000, 15000, 10000, 5000]
    
    # Dictionary to collect results for a summary table at the end
    history_results = {}

    print("Starting sequential training loops...")
    print("------------------------------------------------------------")

    for size in sample_sizes:
        print(f"\n[RUNNING] Training with {size:,} samples...")
        
        # Safe positional slicing for both Pandas Series and Numpy Arrays
        X_train_slice = X_train_clean.iloc[:size] if hasattr(X_train_clean, 'iloc') else X_train_clean[:size]
        y_train_slice = y_train.iloc[:size] if hasattr(y_train, 'iloc') else y_train[:size]

        # Assemble the optimized pipeline
        pipeline = Pipeline([
            ('tfidf', get_best_vectorizer()),
            ('mlp', get_optimized_mlp())
        ])

        # Train model on the data subset
        pipeline.fit(X_train_slice, y_train_slice)

        # Evaluate performance on the FIXED 10,000 test reviews
        predictions = pipeline.predict(X_test_clean)
        acc = accuracy_score(y_test, predictions)
        
        # Store for the final display table
        history_results[size] = acc
        
        print(f"[COMPLETED] Size {size:,} achieved Test Accuracy: {acc:.4f}")
        print(f"--- Short Classification Report for {size:,} samples ---")
        print(classification_report(y_test, predictions, digits=4))
        print("-" * 60)

    # 4. Generate the Grand Summary Table
    print("\n" + "="*60)
    print("LEARNING CURVE ANALYSIS SUMMARY")
    print("="*60)
    print(f"{'Training Samples':<20} | {'Fixed Test Accuracy':<20}")
    print("-" * 45)
    for size in sample_sizes:
        print(f"{size:<20,} | {history_results[size]:.4%}")
    print("="*60)
    print("[SYSTEM] Execution complete. Log permanently saved.")

if __name__ == "__main__":
    main()