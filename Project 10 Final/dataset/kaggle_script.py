import pandas as pd

print("1. Loading the dataset (this might take a minute)...")
# We only load the columns we actually need to save RAM
file_path = '/kaggle/input/datasets/mohamedbakhet/amazon-books-reviews/Books_rating.csv'
df = pd.read_csv(file_path, usecols=['review/text', 'review/score'])

print("2. Dropping empty reviews...")
df = df.dropna()

print("3. Grabbing exactly 10,000 reviews for each star rating...")
# This groups the data by the score, then randomly picks 10,000 from each group
df_balanced = df.groupby('review/score').sample(n=10000, random_state=42)

print("4. Shuffling the dataset...")
# We shuffle it so the 1-star and 5-star reviews are completely mixed up
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("5. Saving to file...")
df_balanced.to_csv('balanced_50k_reviews.csv', index=False)

print("Done! File 'balanced_50k_reviews.csv' is ready for download.")