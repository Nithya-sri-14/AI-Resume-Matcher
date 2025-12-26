import openai

# 1️⃣ Set your OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"

def generate_explanation(resume_text, jd_text, score):
    """
    Generates a human-readable explanation of why the resume matches the job description
    """
    prompt = f"""
    The following is a candidate's resume:

    {resume_text}

    The following is a job description:

    {jd_text}

    The resume has a match score of {score}%.

    Explain in simple, HR-friendly language why this resume matches or does not match the job description.
    """
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.5,
        max_tokens=250
    )
    
    return response.choices[0].message.content.strip()
