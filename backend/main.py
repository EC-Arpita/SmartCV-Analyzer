import os
import json
import re
import fitz  # PyMuPDF
import spacy
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- Enable CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier
import joblib
from nltk.corpus import stopwords
import nltk

# -----------------------
# Setup and Config
# -----------------------
nltk.download('stopwords', quiet=True)
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
CORS(app)  # <-- Allow cross-origin requests from your frontend

app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------
# Helper Functions
# -----------------------
def extract_text(file_path):
    """Extract text from PDF or DOCX."""
    text = ""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text("text")
    return text.strip()

def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return ' '.join([word for word in text.split() if word not in stop_words])

def extract_skills(text):
    """Rule-based skill extraction."""
    common_skills = [
        'python', 'java', 'c++', 'machine learning', 'deep learning',
        'data analysis', 'nlp', 'tensorflow', 'pytorch', 'sql', 'flask',
        'django', 'xgboost', 'excel', 'powerbi', 'tableau',
        'communication', 'leadership', 'problem solving', 'data visualization'
    ]
    found = [s for s in common_skills if s in text]
    missing = [s for s in common_skills if s not in text]
    return found, missing

# -----------------------
# Model Management
# -----------------------
def load_or_train_model():
    """Loads model if exists, else trains it."""
    model_path = os.path.join(MODEL_DIR, "xgb_model.joblib")
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.joblib")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")

    if all(os.path.exists(p) for p in [model_path, vectorizer_path, encoder_path]):
        print("✅ Using existing trained model...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        label_encoder = joblib.load(encoder_path)
        return model, vectorizer, label_encoder

    # Train new model
    print("🧠 Training new XGBoost model...")
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "processed", "custom_resume_dataset_cleaned.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"❌ Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df.dropna(subset=["resume_text", "job_category"], inplace=True)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["resume_text"].astype(str))
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["job_category"])

    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", verbosity=0)
    model.fit(X, y)

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(label_encoder, encoder_path)
    print("✅ Model trained and saved successfully!")

    return model, vectorizer, label_encoder

# Load or train model
model, vectorizer, label_encoder = load_or_train_model()

# -----------------------
# API Routes
# -----------------------
@app.route("/upload", methods=["POST"])
def upload_resume():
    """Endpoint to handle resume upload and analysis."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(save_path)

    resume_text = extract_text(save_path)
    if not resume_text:
        return jsonify({"error": "Could not extract text"}), 500

    clean_resume = clean_text(resume_text)

    X_vec = vectorizer.transform([clean_resume])
    y_pred = model.predict(X_vec)[0]
    predicted_job = label_encoder.inverse_transform([y_pred])[0]

    proba = model.predict_proba(X_vec)[0]
    suitability_score = round(float(np.max(proba)) * 100, 2)

    found_skills, missing_skills = extract_skills(clean_resume)

    if suitability_score > 80:
        feedback = "Excellent fit! Your resume strongly aligns with this role."
    elif suitability_score > 60:
        feedback = "Good fit. Consider improving your skill coverage."
    else:
        feedback = "Needs improvement. Add more relevant skills and experience."

    result = {
        "predicted_job": predicted_job,
        "suitability_score": suitability_score,
        "feedback": feedback,
        "match_skills": found_skills,
        "missing_skills": missing_skills
    }

    return jsonify(result)

@app.route("/results", methods=["GET"])
def get_results():
    """Fetch all past resume analyses."""
    analysis_file = os.path.join(app.config['UPLOAD_FOLDER'], "analyses.json")
    if os.path.exists(analysis_file):
        with open(analysis_file, "r") as f:
            return jsonify(json.load(f))
    else:
        return jsonify([])
    
@app.route("/")
def home():
    return "✅ SmartCV Analyzer API is running! Use /upload to POST resumes."


# -----------------------
# Run Flask App
# -----------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

