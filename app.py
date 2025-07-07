import streamlit as st
import pickle
import docx
import PyPDF2
import re
import spacy
import pandas as pd
import matplotlib.pyplot as plt

# Load models and NLP
nlp = spacy.load("en_core_web_sm")
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# Load Coursera dataset
df = pd.read_csv("coursera_course_dataset_v3.csv")

# ------------- Utility Functions -----------------
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

def clean_resume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def extract_name(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # Try to catch name from first few lines (avoid tech words)
    for line in lines[:5]:
        match = re.match(r"([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)+)", line)
        if match:
            name = match.group(0)
            bad_words = ['Engineer', 'Developer', 'AI', 'ML', 'Data', 'Python', 'Resume']
            if not any(bad in name for bad in bad_words):
                return name
    # SpaCy fallback
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'PERSON' and len(ent.text.split()) <= 4:
            return ent.text.strip()
    return lines[0] if lines else "Not found"

def extract_email(text):
    text_clean = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text_clean = text_clean.replace("(at)", "@").replace("[at]", "@").replace("{at}", "@")
    text_clean = re.sub(r"\s+at\s+", "@", text_clean, flags=re.I)
    text_clean = text_clean.replace("(dot)", ".").replace("[dot]", ".").replace("{dot}", ".")
    text_clean = re.sub(r"\s+dot\s+", ".", text_clean, flags=re.I)
    text_clean = re.sub(r"[>|:|=|•|\-|_]", " ", text_clean)
    text_clean = re.sub(r"\s+", " ", text_clean)
    regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    matches = re.findall(regex, text_clean)
    if not matches:
        regex2 = r"[a-zA-Z0-9_.+-]+\s*@?\s*[a-zA-Z0-9-]+\s*(?:\.|dot)\s*[a-zA-Z0-9-.]+"
        fallback = re.findall(regex2, text_clean)
        fallback_cleaned = [
            re.sub(r"\s*(at|@)\s*", "@", e, flags=re.I).replace(" dot ", ".").replace(" com", ".com").replace(" gmail", "@gmail")
            for e in fallback
        ]
        return fallback_cleaned[0].strip() if fallback_cleaned else "Not found"
    return matches[0].strip() if matches else "Not found"

def extract_phone(text):
    match = re.search(r"\+?\d[\d\s\-]{8,}\d", text)
    return match.group(0) if match else "Not found"

# --- Dynamic Skills Extraction using Coursera Dataset ---
def build_skills_set_from_coursera(df):
    skills_set = set()
    for skills_str in df['Skills'].dropna():
        for skill in skills_str.split(','):
            skills_set.add(skill.strip().lower())
    return skills_set

def extract_all_skills(text, all_skills_set):
    text_lower = text.lower()
    found_skills = set()
    for skill in all_skills_set:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    # Optionally: Skills section catch (even if not in master set)
    possible_skills = []
    skills_section = re.search(r'(skills|technical skills|key skills)(.*?)(\n\n|\Z)', text_lower, re.DOTALL)
    if skills_section:
        lines = skills_section.group(2).split('\n')
        for line in lines:
            for w in re.split(r'[,;/\-]', line):
                w = w.strip()
                if 2 < len(w) < 32:
                    possible_skills.append(w)
    for skill in possible_skills:
        found_skills.add(skill)
    return sorted(found_skills)

def pred(input_resume):
    cleaned_text = clean_resume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]

def get_recommended_skills_and_courses(domain, existing_skills):
    domain_courses = df[df['domain'].str.contains(domain, case=False, na=False)]
    domain_skills = set()
    for s in domain_courses['Skills']:
        if pd.notnull(s):
            for skill in s.split(','):
                domain_skills.add(skill.strip().lower())
    missing_skills = domain_skills - set([s.lower() for s in existing_skills])
    recommended_courses = []
    for idx, row in domain_courses.iterrows():
        for mskill in missing_skills:
            if mskill in (row['Skills'] or '').lower():
                recommended_courses.append((row['Course Name'], row['Course URL'], mskill))
    seen = set()
    top_courses = []
    for cname, curl, skill in recommended_courses:
        if (cname, skill) not in seen and len(top_courses) < 5:
            top_courses.append({'course': cname, 'url': curl, 'skill': skill})
            seen.add((cname, skill))
    return list(missing_skills), top_courses

def calculate_ats_score(text, skills_list):
    score = 0
    criteria = 6
    skills_found = [skill for skill in skills_list if skill in text.lower()]
    score += len(skills_found) / max(len(skills_list), 1)
    if extract_email(text) != "Not found":
        score += 1
    if extract_phone(text) != "Not found":
        score += 1
    if "summary" in text.lower():
        score += 1
    if 300 < len(text.split()) < 1500:
        score += 1
    if "experience" in text.lower() or "work" in text.lower():
        score += 1
    ats_percent = int((score / criteria) * 100)
    return min(ats_percent, 100), skills_found

def ats_score_chart(score):
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        [score, 100 - score],
        labels=['ATS Matched', 'Improve'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#4CAF50', '#FF7043'],
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    plt.setp(autotexts, size=14, weight="bold")
    ax.set(aspect="equal")
    plt.title('ATS Score', fontsize=16, fontweight='bold')
    return fig

def get_resume_suggestions(text, all_skills_set):
    suggestions = []
    if extract_email(text) == "Not found":
        suggestions.append("Add your email address.")
    if extract_phone(text) == "Not found":
        suggestions.append("Add a contact phone number.")
    if "summary" not in text.lower():
        suggestions.append("Add a professional summary at the top.")
    if len(text.split()) < 350:
        suggestions.append("Expand your resume. Add more details about projects, internships, etc.")
    skills_found = [skill for skill in all_skills_set if skill in text.lower()]
    missing_skills = set(all_skills_set) - set(skills_found)
    if missing_skills:
        suggestions.append("Consider highlighting more industry-relevant skills.")
    if not suggestions:
        suggestions.append("Your resume looks strong! 🚀")
    return suggestions

# ------------- THE MAIN FUNCTION -----------------
def main():
    st.set_page_config(page_title="Premium Resume ATS Analyzer", page_icon="🧑‍💼", layout="wide")
    st.title("🧑‍💼 Resume ATS Analyzer & Career Path Recommender")
    st.markdown("Upload your resume (PDF, DOCX, TXT) and get a **premium ATS score, instant feedback, job category prediction, missing skills & recommended Coursera courses!**")

    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    all_skills_set = build_skills_set_from_coursera(df)

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Successfully extracted text from the uploaded resume.")

            # Extraction & Prediction
            name = extract_name(resume_text)
            email = extract_email(resume_text)
            phone = extract_phone(resume_text)
            matched_skills = extract_all_skills(resume_text, all_skills_set)
            category = pred(resume_text)
            ats_percent, skills_found = calculate_ats_score(resume_text, all_skills_set)
            suggestions = get_resume_suggestions(resume_text, all_skills_set)
            missing_skills, course_suggestions = get_recommended_skills_and_courses(category, matched_skills)

            # Show extracted info
            with st.expander("Show Extracted Resume Text"):
                st.text_area("Resume Text", resume_text, height=300)

            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("📝 Candidate Info")
                st.write(f"**Name:** {name}")
                st.write(f"**Email:** {email}")
                st.write(f"**Phone:** {phone}")
                st.write(f"**Skills:** {', '.join(matched_skills)}")
                st.subheader("🔖 Predicted Job Category")
                st.write(f"`{category}`")

            with col2:
                st.subheader("📊 ATS Score")
                fig = ats_score_chart(ats_percent)
                st.pyplot(fig)
                st.markdown(f"### **ATS Score: {ats_percent}%**")
                st.subheader("💡 Resume Improvement Suggestions")
                for tip in suggestions:
                    st.markdown(f"- {tip}")

                if missing_skills:
                    st.markdown("### 🔥 **Recommended Skills for Your Role:**")
                    st.write(", ".join(missing_skills))
                if course_suggestions:
                    st.markdown("### 📚 **Top Coursera Courses to Learn These Skills:**")
                    for c in course_suggestions:
                        st.markdown(f"- **{c['skill'].title()}**: [{c['course']}]({c['url']})")

        except Exception as e:
            st.error(f"❌ Error processing the file: {str(e)}")

# End of main()
if __name__ == "__main__":
    main()
