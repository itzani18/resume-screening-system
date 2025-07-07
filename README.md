If you try to run this file Github doesn't allow to upload more than 25mb so you have to first run .ipynb file one by one then you can run app.py
# Resume Screening and Job Category Prediction App

A Streamlit-based web application that extracts key information from resumes (PDF, DOCX, or TXT) and predicts the most suitable job category using Machine Learning.

## ✨ Features

- **Resume Parsing:** Extracts Name, Email, Phone, and Skills from uploaded resumes.
- **Category Prediction:** Uses a pre-trained ML model to predict the best-fit job domain for the candidate.
- **Skill Extraction:** Matches candidate skills with a predefined list.
- **Supports Multiple Formats:** Accepts PDF, DOCX, and TXT resumes.
- **User-friendly Interface:** Simple, clean, and responsive UI built with Streamlit.

## 🚀 Getting Started

### 1. Clone the Repository

git clone https://github.com/your-username/resume-screening-app.git
cd resume-screening-app

###2.Run .ipynb file so you can get the pkl files
Run one by one each cell dataset is already uploaded but also check the path.

###3. Now you are ready to run app.py
cmd- streamlit run app.py
