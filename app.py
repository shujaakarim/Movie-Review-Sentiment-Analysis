from flask import Flask, render_template, request, jsonify
import pickle
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Define the paths for loading the model and vectorizer from the 'models' folder
model_path = os.path.join('models', 'sentiment_model.pkl')
vectorizer_path = os.path.join('models', 'tfidf_vectorizer.pkl')

# Load the sentiment model and vectorizer
model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))


# Function to preprocess the review text
def preprocess_text(text):
    # Remove non-alphabetic characters (keeping only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespaces
    text = ' '.join(text.split())
    return text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data['review']

    cleaned_review = preprocess_text(review)

    # Reject input that becomes too short or empty after cleaning
    if not cleaned_review or len(cleaned_review.split()) < 3:
        return jsonify({'sentiment': 'Please enter a meaningful review (at least 3 real words).'})

    review_vectorized = vectorizer.transform([cleaned_review])
    prediction = model.predict(review_vectorized)

    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return jsonify({'sentiment': sentiment})


if __name__ == '__main__':
    app.run(debug=True)
