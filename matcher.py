from sklearn.metrics.pairwise import cosine_similarity
from embedding_engine import generate_embedding

def match_resume_with_jd(resume_text, jd_text):
    """
    Input: resume text & JD text
    Output: match score in percentage
    """
    resume_emb = generate_embedding(resume_text)
    jd_emb = generate_embedding(jd_text)

    score = cosine_similarity([resume_emb], [jd_emb])[0][0]
    return round(score * 100, 2)
