from io import BytesIO
import fitz
import spacy
import nltk
from nltk.corpus import stopwords
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import joblib
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import base64

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")

df = pd.read_csv("processed\custom_resume_dataset_cleaned.csv")  # dataset

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["resume_text"].apply(preprocess_text)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["job_role"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

career_opportunities = {
    "Data Scientist": ["AI Researcher", "Data Analyst", "ML Engineer"],
    "Machine Learning Engineer": ["AI Developer", "Data Scientist", "Robotics Engineer"],
    "Business Analyst": ["Product Manager", "Consultant", "Financial Analyst"],
    "Software Developer": ["Backend Developer", "Frontend Developer", "Fullstack Developer"]
}

def extract_text_from_file(file_stream, filename):
    text = ""
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        for page in doc:
            text += page.get_text()
    else:
        text = file_stream.read().decode("utf-8", errors="ignore")
    return text

def predict_job_role(text):
    X_vec = vectorizer.transform([text])
    pred_label = model.predict(X_vec)[0]
    pred_prob = model.predict_proba(X_vec).max()
    job_role = label_encoder.inverse_transform([pred_label])[0]
    return job_role, pred_prob

def generate_feedback(text):
    feedback = []
    word_count = len(text.split())
    if word_count < 200:
        feedback.append("Resume is too short, consider adding more details about your experience.")
    elif word_count > 2000:
        feedback.append("Resume is too long, consider summarizing key points.")
    else:
        feedback.append("Resume length looks good.")
    if "experience" not in text.lower():
        feedback.append("Consider adding a section about work experience.")
    if "skills" not in text.lower():
        feedback.append("Consider adding a skills section to highlight technical abilities.")
    return feedback

def create_base64_plot(title, value):
    plt.figure(figsize=(4,3))
    plt.bar([title], [value], color="skyblue")
    plt.title(title)
    plt.ylabel("Value")
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze_resume():
    if "resume" not in request.files:
        return jsonify({"error":"No file uploaded"}), 400
    file = request.files["resume"]
    raw_text = extract_text_from_file(file, file.filename)
    clean_text = preprocess_text(raw_text)
    job_role, score = predict_job_role(clean_text)
    feedback = generate_feedback(raw_text)
    opportunities = career_opportunities.get(job_role, [])
    word_count_plot = create_base64_plot("Resume Word Count", len(raw_text.split()))
    confidence_plot = create_base64_plot(f"{job_role} Confidence (%)", score*100)
    return jsonify({
        "predicted_role": job_role,
        "suitability_score": round(score*100,2),
        "feedback": feedback,
        "career_opportunities": opportunities,
        "charts": {
            "word_count_plot": word_count_plot,
            "confidence_plot": confidence_plot
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
