from flask import Flask, request, jsonify
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app = Flask(__name__)

# Load intents
with open('intents.json') as f:
    raw_data = json.load(f)
    intents = raw_data.get("intents", [])

questions = []
answers = []
for intent in intents:
    questions.extend(intent["patterns"])
    answers.extend([intent["responses"][0]] * len(intent["patterns"]))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

def get_best_answer(query):
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, X).flatten()
    idx = np.argmax(scores)
    if scores[idx] > 0.1:
        return answers[idx]
    else:
        return "I'm not sure about that. Please fill the contact form below."

@app.route("/chat")
def chat():
    user_query = request.args.get("query")
    response = get_best_answer(user_query)
    return jsonify({"response": response})

@app.route("/lead", methods=["POST"])
def lead():
    data = request.json
    # Save to CSV or DB
    with open("leads.csv", "a") as f:
        f.write(f"{data['name']},{data['email']},{data['message']}\n")
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)
