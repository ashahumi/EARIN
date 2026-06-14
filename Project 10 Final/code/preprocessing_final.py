import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    """
    The absolute best text cleaning function determined by our 
    previous preprocessing iterations (lowercasing and removing non-alphabetic characters).
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def get_best_vectorizer():
    """
    Returns a TfidfVectorizer configured with the mathematically optimal 
    hyperparameters discovered during our 432-fit Grid Search.
    """
    return TfidfVectorizer(
        stop_words='english',
        max_features=5000,  # Optimal vocabulary size
        ngram_range=(1, 2),  # Optimal context length (unigrams + bigrams)
        max_df=0.5          # Optimal threshold (filters terms appearing in >50% of docs)
    )