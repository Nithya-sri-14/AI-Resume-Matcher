import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import PyPDF2

st.set_page_config(page_title="AI Resume Matcher", layout="centered")

st.title("📄 AI Resume Matcher")
st.write("Upload a resume and paste a job description to get a match score")

# ✅ Load model safely (CPU + cache)
@st.cache_resource
def load_model():
    return SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu"
    )

model = load_model()

# ✅ PDF text extraction
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description")

if st.button("Check Match"):
    if resume_file and job_description.strip():
        resume_text = extract_text_from_pdf(resume_file)

        embeddings = model.encode(
            [resume_text, job_description],
            convert_to_numpy=True
        )

        similarity = cosine_similarity(
            [embeddings[0]],
            [embeddings[1]]
        )[0][0]

        score = round(similarity * 100, 2)

        st.success(f"✅ Match Score: {score}%")

        if score > 75:
            st.balloons()
            st.write("🎯 Strong Match")
        elif score > 50:
            st.write("⚠️ Moderate Match")
        else:
            st.write("❌ Weak Match")
    else:
        st.error("Please upload a resume and enter job description")
