import pandas as pd
import random
import os

# Expanded tech-related job roles with common skills
job_roles = {
    "Python Developer": ["Python", "Flask", "Django", "SQL", "APIs", "OOP"],
    "Software Engineer": ["Java", "C++", "Algorithms", "System Design", "Databases", "Git"],
    "AI Engineer": ["Python", "TensorFlow", "PyTorch", "Deep Learning", "NLP", "Computer Vision"],
    "ML Engineer": ["Python", "Scikit-learn", "PyTorch", "Model Optimization", "MLOps", "Feature Engineering"],
    "Prompt Engineer": ["NLP", "ChatGPT", "Prompt Tuning", "Python", "LangChain", "Vector Databases"],
    "Full Stack Developer": ["JavaScript", "React", "Node.js", "Express", "MongoDB", "SQL", "Docker"],
    "Frontend Developer": ["HTML", "CSS", "JavaScript", "React", "Redux", "UI/UX"],
    "Backend Developer": ["Python", "Django", "REST API", "SQL", "Docker", "Authentication"],
    "Mobile Developer": ["Flutter", "Kotlin", "Swift", "Android Studio", "Firebase"],
    "Cloud Engineer": ["AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform"],
    "DevOps Engineer": ["CI/CD", "Docker", "Kubernetes", "Linux", "Terraform", "Jenkins"],
    "Data Scientist": ["Python", "Pandas", "Statistics", "Machine Learning", "SQL", "Scikit-learn"],
    "Data Analyst": ["SQL", "Excel", "Tableau", "Data Cleaning", "Power BI", "Visualization"],
    "Cybersecurity Analyst": ["Network Security", "Firewalls", "Ethical Hacking", "Linux", "Python"],
    "Java Developer": ["Java", "Spring", "Hibernate", "Maven", "REST API", "SQL", "Microservices"],
    "C++ Developer": ["C++", "STL", "Multithreading", "Linux", "Debugging", "System Programming"],
    "Ruby Developer": ["Ruby", "Rails", "PostgreSQL", "JavaScript", "RSpec", "Git"],
    "Go Developer": ["Go", "Microservices", "Docker", "Kubernetes", "REST API", "Linux"],
}

# General feedback suggestions
general_feedback = [
    "Add project links to showcase work.",
    "Include certifications related to your role.",
    "Expand on tools you've used professionally.",
    "Add quantifiable achievements to stand out.",
    "Mention teamwork or leadership experience.",
    "Include internships or volunteer work if available.",
    "List recent tech stacks and tools."
]

rows = []
resume_id = 1

# Control total dataset size (~1000 resumes)
total_samples = 1000
samples_per_role = total_samples // len(job_roles)

for role, all_skills in job_roles.items():
    for _ in range(samples_per_role):
        included = random.sample(all_skills, k=random.randint(3, len(all_skills)))
        missing = list(set(all_skills) - set(included))

        # Suitability scoring (weighted by skills present/missing)
        score = 55 + len(included) * 8 - len(missing) * 4
        score = min(max(score, 0), 100)

        # More realistic resume text
        resume = (
            f"As a {role}, I have hands-on experience with {', '.join(included)}. "
            f"I have contributed to projects involving {random.choice(included)} "
            f"and collaborated in agile teams to deliver high-quality solutions. "
            f"My goal is to apply {role.lower()} skills effectively in challenging environments."
        )

        job_desc = f"This role requires expertise in {', '.join(all_skills)}."

        # Feedback based on missing skills & score
        feedback = []
        if missing:
            feedback.append(f"Consider adding experience with: {', '.join(missing[:2])}.")
        if score < 60:
            feedback.append("Resume needs stronger alignment with the role requirements.")
        if not feedback:
            feedback.append("Resume is strong and matches well with job description.")
        feedback += random.sample(general_feedback, k=2)

        rows.append({
            "resume_id": f"R{resume_id:04d}",
            "resume_text": resume,
            "job_category": role,
            "job_description": job_desc,
            "suitability_score": score,
            "feedback": " ".join(feedback),
            "match_skills": ", ".join(included),
            "missing_skills": ", ".join(missing)
        })

        resume_id += 1

# Save dataset
df = pd.DataFrame(rows)
os.makedirs("processed", exist_ok=True)
df.to_csv("processed/custom_resume_dataset.csv", index=False)

print(f"✅ Custom dataset with {len(df)} resumes saved to processed/custom_resume_dataset.csv")
