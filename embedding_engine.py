from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text):
    """
    Input: text (string)
    Output: embedding vector (list of floats)
    """
    return model.encode(text)