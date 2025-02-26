import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model for text representation
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') #https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
embedding_dim = model.get_sentence_embedding_dimension()
# Initialize a FAISS index 
index = faiss.IndexFlatL2(embedding_dim)

# Email Storage
emails = {} # Dictionary to map faiss indices to email content

def store_email_in_faiss(email_text, email_id):
    """
    Converts email text into embeddings and stores them in FAISS.
    email_text: The extracted email content.
    email_id: Unique identifier of the email.
    """
    global email_store
    embedding = model.encode(email_text)
    embedding = np.array([embedding]).astype("float32")

    index.add(embedding)  # Add embedding to FAISS
    email_store[len(emails)] = {"id": email_id, "text": email_text}