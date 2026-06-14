import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    """
    Basic text cleaning: lowercasing and removing non-alphanumeric characters.
    """
    text = str(text).lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def vectorize_text(X_train, X_test, remove_stopwords=True):
    """
    Cleans and vectorizes the text using TF-IDF. 
    The 'remove_stopwords' flag is used for the ablation study.
    """
    print(f"Cleaning and Vectorizing text (Stopwords removed: {remove_stopwords})...")
    
    # Clean the raw text
    X_train_clean = X_train.apply(clean_text)
    X_test_clean = X_test.apply(clean_text)
    
    # Set up the TF-IDF Vectorizer
    stop_words_setting = 'english' if remove_stopwords else None
    
    # Smarter Vectorizer Tweak
    vectorizer = TfidfVectorizer(
        max_features=5000, 
        min_df=5,       
        max_df=0.85,    
        ngram_range=(1, 3),
        stop_words=stop_words_setting
    )
    
    # Fit on training data, transform both train and test data
    X_train_vec = vectorizer.fit_transform(X_train_clean)
    X_test_vec = vectorizer.transform(X_test_clean)
    
    return X_train_vec, X_test_vec, vectorizer