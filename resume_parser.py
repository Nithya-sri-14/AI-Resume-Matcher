import pdfplumber
import re

def extract_resume_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    text = re.sub(r'\s+', ' ', text)
    return text.lower()