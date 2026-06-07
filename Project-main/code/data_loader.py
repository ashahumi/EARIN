from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(filepath, text_col='review/text', target_col='review/score', test_size=0.2, random_state=42):
    """
    Loads the dataset, handles missing values, and splits it into training and testing sets.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Keep only the columns we need
    df = df[[text_col, target_col]]
    
    # Drop rows with missing text or scores
    df = df.dropna()
    
    X = df[text_col]
    y = df[target_col]
    
    # Stratified split ensures the ratio of 1-5 star reviews is the same in train and test sets
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Quick test to make sure it runs
    dataset_path = Path(__file__).resolve().parent.parent / "dataset" / "my_50k_reviews.csv"
    X_train, X_test, y_train, y_test = load_and_split_data(dataset_path)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")