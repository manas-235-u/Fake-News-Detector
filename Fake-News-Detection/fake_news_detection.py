import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load Datasets

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], axis=0)

# Shuffle dataset
data = data.sample(frac=1, random_state=42)

# Select useful columns
data = data[["title", "text", "label"]]

# Combine title and text
data["content"] = data["title"] + " " + data["text"]


# Text Cleaning Function

def clean_text(text):
    
    text = text.lower()
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    return text


# Apply cleaning
data["content"] = data["content"].apply(clean_text)


# Input and Output


X = data["content"]
y = data["label"]


# Convert Text to Numbers

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X = vectorizer.fit_transform(X)


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train Model


model = MultinomialNB()

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)


# Model Evaluation


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# Manual News Prediction


def predict_news(news):

    news = clean_text(news)

    vector = vectorizer.transform([news])

    prediction = model.predict(vector)

    if prediction[0] == 1:
     print("\nPrediction: Real News")
    else:
     print("\nPrediction: Fake News")


# User input
user_news = input("\nEnter a news headline or paragraph:\n")

predict_news(user_news)