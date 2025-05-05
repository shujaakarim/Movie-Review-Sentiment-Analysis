import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the training data
df = pd.read_csv('train_clean.csv')

# Features and labels
X = df['review']
y = df['sentiment']  # should be 0 (Negative) or 1 (Positive)

# Convert text to numbers using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Train the model
model = LogisticRegression()
model.fit(X_vec, y)

# Create a folder to save model files
import os
os.makedirs('model', exist_ok=True)

# Save the model
with open('model/sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the vectorizer
with open('model/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
