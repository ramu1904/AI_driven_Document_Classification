import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import joblib

# Load the dataset
data = pd.read_csv("documents.csv")

# Load SpaCy's English language model
nlp = spacy.load('en_core_web_sm')

# Preprocess the text
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

data['Processed_Text'] = data['Document_Text'].apply(preprocess_text)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['Processed_Text'])

# Save the vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, data['Category'], test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "model.pkl")

# Test the model
y_pred = model.predict(X_test)
print("Classification Report:\n", metrics.classification_report(y_test, y_pred))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))
