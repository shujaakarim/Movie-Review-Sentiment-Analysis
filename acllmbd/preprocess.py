import os
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def load_data(folder_path):
    reviews = []
    sentiments = []

    for label in ['pos', 'neg']:
        path = os.path.join(folder_path, label)
        for file_name in os.listdir(path):
            with open(os.path.join(path, file_name), encoding='utf-8') as file:
                review = file.read().lower()
                review = review.translate(str.maketrans('', '', string.punctuation))
                tokens = word_tokenize(review)
                filtered_tokens = [word for word in tokens if word not in stop_words]
                reviews.append(' '.join(filtered_tokens))
                sentiments.append(1 if label == 'pos' else 0)

    return pd.DataFrame({'review': reviews, 'sentiment': sentiments})

# Load training and test data
train_data = load_data("aclImdb/train")
test_data = load_data("aclImdb/test")

# Save as CSV
train_data.to_csv("train_clean.csv", index=False)
test_data.to_csv("test_clean.csv", index=False)

print("✅ Preprocessing completed and saved as CSV!")
