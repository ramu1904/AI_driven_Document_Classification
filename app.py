from flask import Flask, request, jsonify
import joblib
import spacy

# Load the trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
nlp = spacy.load('en_core_web_sm')

# Initialize Flask app
app = Flask(__name__)

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

@app.route('/classify', methods=['POST'])
def classify_document():
    data = request.get_json()
    document_text = data['document_text']
    
    # Preprocess the text
    processed_text = preprocess_text(document_text)
    
    # Transform text to vector
    vectorized_text = vectorizer.transform([processed_text])
    
    # Predict category
    prediction = model.predict(vectorized_text)
    
    return jsonify({'category': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
