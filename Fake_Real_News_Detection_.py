import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Try loading from Colab Drive, else use local
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_PATH = '/content/drive/MyDrive/fake_news.csv'
    print("Loading data from Google Drive:", DATA_PATH)
except ModuleNotFoundError:
    DATA_PATH = 'fake_news.csv'
    print("Looking for fake_news.csv in working directory.")

# Load dataset
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Could not find {DATA_PATH}. Please ensure 'fake_news.csv' is present.")

print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Basic EDA
print("\nLabel Distribution:")
print(df['label'].value_counts())

df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
plt.hist(df[df['label'] == 'FAKE']['text_length'], bins=30, alpha=0.5, label='Fake')
plt.hist(df[df['label'] == 'REAL']['text_length'], bins=30, alpha=0.5, label='Real')
plt.legend()
plt.title('Text Length Distribution')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Preprocessing Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)               # remove punctuation and numbers
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Train-Test Split
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model: Passive Aggressive Classifier
model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(np.arange(2), ['FAKE', 'REAL'])
plt.yticks(np.arange(2), ['FAKE', 'REAL'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# Save model and vectorizer
with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("Model and vectorizer saved.")

# Prediction Utility
def predict_news(text: str) -> str:
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction

# Example usage
if __name__ == '__main__':
    example_text = df['text'].iloc[0]
    prediction = predict_news(example_text)
    print("Sample prediction:", prediction)
