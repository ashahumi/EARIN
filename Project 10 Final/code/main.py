from pathlib import Path

from sklearn.metrics import classification_report, accuracy_score
from data_loader import load_and_split_data
from preprocessing import vectorize_text
from models import get_logistic_regression, get_random_forest, get_mlp

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Trains the model and prints the evaluation metrics."""
    print(f"\n--- Training {model_name} ---")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    acc = accuracy_score(y_test, predictions)
    print(f"{model_name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, predictions))
    return acc

def main():
    # 1. Load and Split Data
    # NOTE: If your CSV has different column names, change them here!
    dataset_path = Path(__file__).resolve().parent.parent / "dataset" / "my_50k_reviews.csv"
    X_train, X_test, y_train, y_test = load_and_split_data(
        dataset_path,
        text_col='review/text', 
        target_col='review/score'
    )
    
    # 2. Preprocess Data (Default: with stopwords removed)
    X_train_vec, X_test_vec, _ = vectorize_text(X_train, X_test, remove_stopwords=True)
    
    # 3. Initialize Models
    models = {
        "Logistic Regression": get_logistic_regression(),
        "Random Forest": get_random_forest(),
        "MLP (Neural Network)": get_mlp()
    }
    
    # 4. Train and Evaluate All Models
    print("\n" + "="*40)
    print("MAIN EXPERIMENTS (Comparing 3 Algorithms)")
    print("="*40)
    
    from sklearn.base import clone
    best_acc = 0
    best_model_name = ""
    best_model = None

    for name, model in models.items():
        # Evaluate the model and get its accuracy
        acc = evaluate_model(model, X_train_vec, y_train, X_test_vec, y_test, name)
        
        # Check if this is the new best model
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            # Clone creates a fresh, untrained copy of the winning architecture
            best_model = clone(model) 
            
    # 5. Ablation Study (Testing impact of Stop-words)
    print("\n" + "="*40)
    print(f"ABLATION STUDY: Does removing stop-words help? (Using best model: {best_model_name})")
    print("="*40)
    
    # Create a new vectorized dataset WITHOUT removing stopwords
    X_train_vec_ablation, X_test_vec_ablation, _ = vectorize_text(X_train, X_test, remove_stopwords=False)
    
    # Test it using the dynamically selected best model
    print(f"\n--- Training {best_model_name} (NO Stop-word removal) ---")
    evaluate_model(best_model, X_train_vec_ablation, y_train, X_test_vec_ablation, y_test, f"{best_model_name} (No Stopwords)")

if __name__ == "__main__":
    main()