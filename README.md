# SmartCV Analyzer

**SmartCV Analyzer** is a web app that analyzes resumes and provides actionable career insights. Upload a resume to get:

- **Predicted Job Role** – Suggests the best-fit job category.  
- **Suitability Score** – Measures alignment with job description.  
- **Actionable Feedback** – Highlights missing skills, achievements, and improvements.  
- **Career Analytics Dashboard** – Visualizes word count, skill distribution, and role matches.  

---

## Features

- ML-based **Job Prediction** using TF-IDF + XGBoost / scikit-learn.  
- **Resume Suitability Scoring** and **Detailed Feedback**.  
- **Career Analytics** for single resumes or aggregated data.  
- Fully supports **custom datasets**.  
- Easily integrates with **HTML/CSS frontend**.  

---

## Tech Stack

- Python: scikit-learn, pandas, numpy, NLTK, SpaCy, PyMuPDF, joblib  
- Frontend: HTML & CSS  

---

## Installation

```bash
git clone <https://github.com/EC-Arpita/SmartCV-Analyzer>
cd SmartCV-Analyzer
python -m venv myenv
# Windows
env\Scripts\activate
# macOS/Linux
source myenv/bin/activate
pip install -r requirements.txt
