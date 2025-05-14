import os
import spacy
import pdfplumber
import re
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from collections import Counter
import random
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
nlp = spacy.load("en_core_web_sm")

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
use_gemini = False
model = None
if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        use_gemini = True
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")

SKILL_KEYWORDS = ["python", "java", "sql", "excel", "communication", "teamwork", "leadership", "machine learning", "artificial intelligence",
                  "tensorflow", "react", "node\.js", "aws", "cloud computing", "linux", "html", "css", "javascript", "data analysis",
                  "project management", "marketing", "sales", "finance", "accounting", "c++", "c#", ".net", "docker", "kubernetes",
                  "agile", "scrum", "git", "testing", "automation", "ui", "ux", "design", "writing", "presentation", "research"]

EDUCATION_KEYWORDS = ["B\.Tech", "M\.Tech", "Bachelor", "Master", "PhD", "B\.Sc", "M\.Sc", "MBA", "BCA", "MCA", "high school", "associate"]
EXPERIENCE_KEYWORDS = ['experience', 'worked', 'employment', 'professional summary', 'responsibilities']

def extract_text_from_pdf(file_path):
    text = ''
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + '\n'
    except Exception:
        return None
    return text

def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else 'Not found'

def extract_phone(text):
    match = re.search(r'\+?\d[\d\s-]{8,}\d', text)
    return match.group(0) if match else 'Not found'

def extract_education(text):
    education = [line for line in text.split('\n') if any(re.search(r'\b' + edu + r'\b', line, re.IGNORECASE) for edu in EDUCATION_KEYWORDS)]
    return education if education else ["Not found"]

def extract_experience(text):
    experience_lines = [line for line in text.split('\n') if any(word in line.lower() for word in EXPERIENCE_KEYWORDS)]
    years = re.findall(r'(\d+)\+?\s+years?', text.lower())
    exp_years = max([int(y) for y in years], default=0)
    return experience_lines, exp_years

def extract_skills_nlp(text):
    doc = nlp(text.lower())
    skills = set()
    for token in doc:
        if token.lemma_ in SKILL_KEYWORDS or token.text in SKILL_KEYWORDS:
            skills.add(token.text.capitalize())
        if token.dep_ in ["compound"] and token.head.lemma_ in SKILL_KEYWORDS:
            skills.add(f"{token.text.capitalize()} {token.head.text.capitalize()}")

    extracted_skills = set()
    for skill in skills:
        if skill.lower() in [kw.lower() for kw in SKILL_KEYWORDS]:
            extracted_skills.add(skill)
        else:
            for kw in SKILL_KEYWORDS:
                if kw.lower() in skill.lower():
                    extracted_skills.add(kw.capitalize())
                    break
    return sorted(list(extracted_skills))

def generate_summary(name, education, exp_years, skills):
    edu_summary = education[0] if education else "No formal education listed."
    top_n_skills = ", ".join(skills[:min(5, len(skills))]) if skills else "None listed."
    experience_str = f"approximately {exp_years} years" if exp_years > 0 else "some"
    return f"""{name} is a professional with {experience_str} of experience.
They hold an academic background in {edu_summary}, and their key skills include: {top_n_skills}."""

def generate_suggestions(skills, exp_years, text, role=None):
    suggestions = []
    common_soft_skills = ["Communication", "Teamwork", "Problem-solving", "Critical Thinking", "Adaptability"]
    text_lower = text.lower()

    if role == 'candidate':
        if exp_years < 2:
            suggestions.append("Consider gaining more hands-on experience through internships or entry-level roles.")
        missing_soft_skills = [skill for skill in common_soft_skills if skill not in skills]
        if missing_soft_skills:
            suggestions.append(f"Highlight or develop skills in: {', '.join(missing_soft_skills[:2])}.")
        relevant_tech = [kw.capitalize() for kw in ["python", "java", "sql", "machine learning", "cloud"] if kw.capitalize() not in skills]
        if relevant_tech and len(skills) < 5:
            suggestions.append(f"Consider showcasing skills like: {', '.join(relevant_tech[:2])}.")
        if not any(keyword in text_lower for keyword in EXPERIENCE_KEYWORDS):
            suggestions.append("Ensure you have a clear 'Experience' or 'Work History' section.")
        if not any(edu.lower() in text_lower for edu in EDUCATION_KEYWORDS):
            suggestions.append("Make sure your 'Education' details are clearly mentioned.")
    elif role == 'hr':
        if "quantified achievements" not in text_lower:
            suggestions.append("Look for opportunities to quantify achievements with numbers and data.")
        if not re.search(r'\b(action verbs)\b', text_lower):
            suggestions.append("Check for the use of strong action verbs to describe responsibilities.")
        if text_lower.count(" ") < 300: # Very rough heuristic for detail
            suggestions.append("The resume might lack sufficient detail in describing roles and responsibilities.")

    if not suggestions:
        suggestions.append("The resume looks generally well-structured.")
    return suggestions

def recommend_jobs(skills, text):
    text_lower = text.lower()
    recommended_roles = set()

    if any(skill in skills for skill in ["Python", "Machine Learning", "Tensorflow", "Data Analysis", "AI"]):
        recommended_roles.add("Machine Learning Engineer")
        recommended_roles.add("Data Scientist")
        recommended_roles.add("AI Engineer")
        recommended_roles.add("Software Engineer (ML/AI)")

    if any(skill in skills for skill in ["Java", "C++", "C#", "Software Development"]):
        recommended_roles.add("Software Engineer")
        recommended_roles.add("Backend Developer")

    if any(skill in skills for skill in ["React", "Node.js", "Javascript", "HTML", "CSS", "Frontend"]):
        recommended_roles.add("Frontend Developer")
        recommended_roles.add("Web Developer")
        recommended_roles.add("Full Stack Developer")

    if any(skill in skills for skill in ["SQL", "Database", "Data Modeling", "Data Analysis"]):
        recommended_roles.add("Data Analyst")
        recommended_roles.add("Database Administrator")

    if "project management" in text_lower or any(skill in skills for skill in ["Project Management", "Leadership", "Agile", "Scrum"]):
        recommended_roles.add("Project Manager")

    if "business analysis" in text_lower or any(skill in skills for skill in ["Business Analyst", "Analysis", "Excel"]):
        recommended_roles.add("Business Analyst")
        recommended_roles.add("Consultant")

    if not recommended_roles:
        recommended_roles.add("Various Technical or Professional Roles")

    return list(recommended_roles)

def calculate_ats_score(text, skills, role=None):
    score = 0
    text_lower = text.lower()

    for skill in skills:
        if skill.lower() in text_lower:
            score += 5

    score += len(skills) * 2

    if re.search(r'\b(experience|work history)\b', text_lower):
        score += 10
    if any(edu.lower() in text_lower for edu in EDUCATION_KEYWORDS):
        score += 8
    if re.search(r'\b(skills|technical proficiencies)\b', text_lower):
        score += 7
    if re.search(r'\b(summary|objective)\b', text_lower):
        score += 5

    max_possible_score = len(SKILL_KEYWORDS) * 5 + 100
    if max_possible_score > 0:
        ats_score = min(100, int((score / max_possible_score) * 100))
    else:
        ats_score = 50

    return max(0, ats_score)

def generate_summary_gemini(name, education, exp_years, skills, text):
    prompt = f"""You are an AI resume analyst. Provide a summary of the following resume. Do not use bold formatting.

        Candidate Name: {name}
        Education: {', '.join(education)}
        Years of Experience: {exp_years}
        Key Skills: {', '.join(skills)}
        Resume Text: {text[:800]}

        Focus on summarizing the candidate's qualifications, experience, and key strengths. Do not Exceed 100 words.
        """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating Gemini summary: {e}")
        return generate_summary(name, education, exp_years, skills)

def generate_suggestions_gemini(skills, exp_years, text, role=None):
    prompt = f"""You are an AI resume improvement assistant. Based on the following information from a resume, provide 2-3 concise suggestions for improvement in a numbered list format. Do not use bold formatting.

        Extracted Skills: {', '.join(skills)}
        Years of Experience: {exp_years}
        Overall Resume Text: {text[:800]}

        The user is a {'candidate' if role == 'candidate' else 'recruiter reviewing a candidate'}. Tailor your suggestions accordingly.
        """
    try:
        response = model.generate_content(prompt)
        suggestions = [part.text.lstrip('1234567890. ').strip() for part in response.parts if hasattr(part, 'text')]
        return suggestions
    except Exception as e:
        print(f"Error generating Gemini suggestions: {e}")
        return generate_suggestions(skills, exp_years, text, role)

def recommend_jobs_gemini(skills, text):
    prompt = f"""You are a job recommendation AI. Based on the following skills, suggest 3-4 relevant job titles. Do not use bold formatting.

        Skills: {', '.join(skills)}
        Resume Text: {text[:500]}

        Provide only the job titles, separated by commas.
        """
    try:
        response = model.generate_content(prompt)
        return [j.strip() for j in response.text.split(',')]
    except Exception as e:
        print(f"Error generating Gemini job roles: {e}")
        return recommend_jobs(skills, text)

def generate_ats_score_gemini(text, skills, role=None):
    prompt = f"""You are an AI ATS (Applicant Tracking System) scoring tool. Assess the following resume text and skills for ATS compatibility. Provide a score out of 100 and a brief explanation (no bolding).

        Resume Text: {text[:1000]}
        Extracted Skills: {', '.join(skills)}

        Provide the score and explanation in this format: "Score: [score]/100. [Explanation]"
        """
    try:
        response = model.generate_content(prompt)
        score_match = re.search(r"Score: (\d+)/100\. (.*)", response.text)
        if score_match:
            return int(score_match.group(1)), score_match.group(2)
        else:
            return calculate_ats_score(text, skills, role), "ATS Score could not be fully determined by AI."
    except Exception as e:
        print(f"Error generating Gemini ATS score: {e}")
        return calculate_ats_score(text, skills, role), "ATS Score could not be fully determined by AI."
    
def detect_red_flags(text):
    flags = []
    text_lower = text.lower()
    if re.search(r'\b(fired|terminated)\b', text_lower):
        flags.append("Mention of termination.")
    if text_lower.count("experience") < 1 and "professional summary" not in text_lower:
        flags.append("Limited work experience description.")
    if re.search(r'(gap of \d+ years|significant employment gap)', text_lower):
        flags.append("Possible significant employment gap.")
    return flags if flags else ["None detected"]

def generate_hr_insights_gemini(text):
    prompt = f"""You are an AI HR assistant reviewing a resume. Provide 2-3 key insights or potential red flags for an HR professional based on the following resume text. Do not use bold formatting.

    Resume Text: {text[:1000]}
    """
    try:
        response = model.generate_content(prompt)
        return [part.text.strip() for part in response.parts if hasattr(part, 'text')]
    except Exception as e:
        print(f"Error generating Gemini HR insights: {e}")
        return detect_red_flags(text)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    role = request.form.get('role')

    if 'resume' not in request.files:
        print("Error: 'resume' key not found in request.files")
        return "Error: No resume file uploaded.", 400

    file = request.files['resume']
    print(f"request.files: {request.files}")

    if file.filename == '':
        return "Error: No file selected.", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    text = extract_text_from_pdf(filepath)
    if text is None:
        return render_template("error.html", message="The uploaded PDF file could not be read."), 400

    email = extract_email(text)
    phone = extract_phone(text)
    education = extract_education(text)
    experience_lines, exp_years = extract_experience(text)
    skills = extract_skills_nlp(text)
    name_match = re.search(r'(?:Name[:\s]*)?([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)', text)
    name = name_match.group(1) if name_match else "Candidate"

    if use_gemini:
        try:
            summary = generate_summary_gemini(name, education, exp_years, skills, text)
            suggestions = generate_suggestions_gemini(skills, exp_years, text, role)
            job_roles = recommend_jobs_gemini(skills, text)
            ats_score, ats_explanation = generate_ats_score_gemini(text, skills, role)
            hr_insights = generate_hr_insights_gemini(text) if role == 'hr' else detect_red_flags(text)
        except Exception as e:
            print(f"Error during Gemini analysis: {e}")
            summary = generate_summary(name, education, exp_years, skills)
            suggestions = generate_suggestions(skills, exp_years, text, role)
            job_roles = recommend_jobs(skills, text)
            ats_score = calculate_ats_score(text, skills, role)
            ats_explanation = "ATS Score calculated using rule-based method due to AI issue."
            hr_insights = detect_red_flags(text)
    else:
        summary = generate_summary(name, education, exp_years, skills)
        suggestions = generate_suggestions(skills, exp_years, text, role)
        job_roles = recommend_jobs(skills, text)
        ats_score = calculate_ats_score(text, skills, role)
        ats_explanation = "ATS Score calculated using rule-based method."
        hr_insights = detect_red_flags(text)

    red_flags = hr_insights if role == 'hr' else detect_red_flags(text)

    return render_template("result.html", role=role, name=name, email=email, phone=phone,
                           education=education, experience=experience_lines,
                           exp_years=exp_years, skills=skills, summary=summary,
                           suggestions=suggestions, job_roles=job_roles,
                           red_flags=red_flags, ats_score=ats_score, ats_explanation=ats_explanation)

if __name__ == "__main__":
    app.run(debug=True)
