from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

app = Flask(__name__)

# Load your dataset from the 'datasets' folder
train_data = pd.read_csv('datasets/train.tsv', sep='\t', header=0)
train_data.columns = train_data.columns.str.strip()  # Strip any whitespace from column names

# Print the columns (for debugging)
print("Columns in train_data:", train_data.columns.tolist())

# Preprocess data
X_train = train_data['text']  # Column containing news content
y_train = train_data['label']  # Column indicating fake or real news

# Here you can continue with the validation and testing datasets if needed
valid_data = pd.read_csv('datasets/valid.tsv', sep='\t', header=0)
valid_data.columns = valid_data.columns.str.strip()  # Ensure to strip whitespace for valid dataset

X_valid = valid_data['text']  # Column containing news content for validation
y_valid = valid_data['label']  # Column indicating fake or real news for validation

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)
X_valid_vectorized = vectorizer.transform(X_valid)

# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Save the model and vectorizer
joblib.dump(model, 'model/fake_news_model.pkl')
joblib.dump(vectorizer, 'model/vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    article = request.form['article']
    # Load the model and vectorizer
    model = joblib.load('model/fake_news_model.pkl')
    vectorizer = joblib.load('model/vectorizer.pkl')

    # Vectorize the input article
    article_vectorized = vectorizer.transform([article])
    prediction = model.predict(article_vectorized)[0]

    if prediction == 'false':
        result = 'Fake News'
    else:
        result = 'Real News'

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

