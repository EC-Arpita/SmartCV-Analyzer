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
import matplotlib.pyplot as plt  # Added for generating graph data
import seaborn as sns  # Added for generating graph data
import base64  # Added for sending graph images to frontend
from io import BytesIO  # Added for in-memory image handling

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
    """Rule-based skill extraction. Updated for more general skills."""
    technical_skills = [
        'python', 'java', 'c++', 'machine learning', 'deep learning',
        'data analysis', 'nlp', 'tensorflow', 'pytorch', 'sql', 'flask',
        'django', 'xgboost', 'excel', 'powerbi', 'tableau', 'html', 'css', 'javascript'
    ]
    soft_skills = [
        'communication', 'leadership', 'problem solving', 'teamwork', 'adaptability',
        'critical thinking', 'time management', 'data visualization'
    ]
    
    all_skills = technical_skills + soft_skills

    found_tech = [s for s in technical_skills if s in text]
    found_soft = [s for s in soft_skills if s in text]
    found = found_tech + found_soft

    missing_tech = [s for s in technical_skills if s not in text]
    missing_soft = [s for s in soft_skills if s not in text]
    missing = missing_tech + missing_soft
    
    return found, missing, found_soft  # Return soft skills for personalized feedback


def generate_graphs(predicted_jobs, found_skills, missing_skills):
    """Generates base64 encoded images for the career analytics dashboard."""
    graph_data = {}

    # Colors derived from style.css for better visual consistency
    MATCHED_COLOR = '#4ade80'  # Green for matched skills
    MISSING_COLOR = '#f87171'  # Red for missing skills
    PIE_CHART_PALETTE = 'magma'  # Palette that complements the app's deep purple/blue theme

    # 1. Skills Match vs. Missing Skills
    plt.figure(figsize=(7, 5))
    sns.set_style("whitegrid")
    labels = ['Matched Skills', 'Missing Skills']
    counts = [len(found_skills), len(missing_skills)]
    sns.barplot(x=labels, y=counts, palette=[MATCHED_COLOR, MISSING_COLOR])
    plt.title('Skill Coverage Analysis')
    plt.ylabel('Count')
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph_data['skills_bar_chart'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    # 2. Top Predicted Jobs 
    plt.figure(figsize=(7, 5))
    job_labels = [j['job'] for j in predicted_jobs]
    job_scores = [j['score'] for j in predicted_jobs]
    plt.pie(job_scores, labels=job_labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette(PIE_CHART_PALETTE, len(job_labels)))
    plt.title('Suitability Score Distribution (Top 3 Roles)')
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    graph_data['jobs_pie_chart'] = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return graph_data

# -----------------------
# Model Management
# -----------------------
def load_or_train_model():
    """Loads model if exists, else trains it."""
    model_path = os.path.join(MODEL_DIR, "xgb_model.joblib")
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.joblib")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.joblib")

    if all(os.path.exists(p) for p in [model_path, vectorizer_path, encoder_path]):
        print(" Using existing trained model...")
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        label_encoder = joblib.load(encoder_path)
        return model, vectorizer, label_encoder

    # Train new model
    print("Training new XGBoost model...")
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed", "custom_resume_dataset_cleaned.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f" Dataset not found: {dataset_path}")

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
    print(" Model trained and saved successfully!")

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
    
    # --- Predict Top 3 Jobs ---
    proba = model.predict_proba(X_vec)[0]
    
    # Get top 3 predicted job indices
    top_3_indices = np.argsort(proba)[::-1][:3]
    predicted_jobs_raw = label_encoder.inverse_transform(top_3_indices)
    predicted_job_scores = proba[top_3_indices]
    
    predicted_jobs = []
    for job, score in zip(predicted_jobs_raw, predicted_job_scores):
        predicted_jobs.append({
            "job": job,
            "score": round(float(score) * 100, 2)
        })
    
    predicted_job = predicted_jobs[0]["job"]
    suitability_score = predicted_jobs[0]["score"]  # Use the score of the top job
    
    # --- Skill Extraction and Feedback Logic ---
    found_skills, missing_skills, found_soft_skills = extract_skills(clean_resume)

    #  Personalized Feedback (3 Points)
    personalized_feedback = []
    if 'python' in found_skills or 'java' in found_skills:
        personalized_feedback.append("Excellent work! Your core programming skills are well highlighted, a crucial aspect for technical roles.")
    else:
        personalized_feedback.append(f"Consider adding or detailing experience with core technical skills like Python or Java for competitive roles.")
    
    if 'data visualization' in found_skills and ('powerbi' in found_skills or 'tableau' in found_skills):
        personalized_feedback.append("Your data visualization and tool knowledge is a strong asset; make sure to quantify your achievements using these tools.")
    elif len(found_soft_skills) < 3:
        personalized_feedback.append("Your soft skills (e.g., communication, leadership) are important. Try to integrate them more explicitly into your experience section.")
    else:
        personalized_feedback.append("You have a good mix of technical and soft skills. Focus on tailoring them to the specific job description.")

    if len(missing_skills) > 5:
        personalized_feedback.append(f"You are currently missing {len(missing_skills)} key skills. Focus on training in areas like {', '.join(missing_skills[:2])} to improve your score.")
    else:
        personalized_feedback.append("Minimal missing skills detected. A slight boost in domain-specific keywords will further enhance your profile.")

    # 2. Default/General Feedback (5 Points)
    default_feedback = [
        "Ensure your contact information (email, phone) is accurate and easily scannable.",
        "Use strong action verbs at the beginning of bullet points in your experience section.",
        "Quantify your achievements! Use numbers, percentages, and dollar amounts to show impact.",
        "Review your resume for consistent formatting and proper use of headings.",
        "Keep your resume concise and target it specifically to the job role you are applying for."
    ]

    #  Overall Feedback (The original single feedback logic, adapted)
    if suitability_score > 80:
        overall_feedback = "Excellent fit! Your resume strongly aligns with this role."
    elif suitability_score > 60:
        overall_feedback = "Good fit. Consider improving your skill coverage."
    else:
        overall_feedback = "Needs improvement. Add more relevant skills and experience."

    # --- Career Analytics Dashboard Data ---
    graph_images = generate_graphs(predicted_jobs, found_skills, missing_skills)

    result = {
        "predicted_jobs": predicted_jobs,
        "suitability_score": suitability_score,
        "overall_feedback": overall_feedback,
        "personalized_feedback": personalized_feedback,
        "default_feedback": default_feedback,
        "match_skills": found_skills,
        "missing_skills": missing_skills,
        "career_analytics_dashboard_data": graph_images
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
