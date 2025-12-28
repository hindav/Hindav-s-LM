from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

model = SentenceTransformer("all-MiniLM-L6-v2")

profile_chunks = [
    "Hindav Deshmukh is a Computer Science engineering graduate (2024) from Jawaharlal Darda Institute of Engineering and Technology, Pune.",
    "His primary focus is applied machine learning and Python backend development.",
    "He built an AI-based movie character recognition system using Python and OpenCV.",
    "He developed an AI-powered stock market forecasting system using LSTM models.",
    "He works with Python, FastAPI, Flask, MongoDB, SQL, and Linux."
]

chunk_embeddings = model.encode(profile_chunks)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(data: Question):
    question_embedding = model.encode([data.question])
    similarities = cosine_similarity(question_embedding, chunk_embeddings)
    best_idx = int(np.argmax(similarities))
    return {
        "answer": profile_chunks[best_idx]
    }
