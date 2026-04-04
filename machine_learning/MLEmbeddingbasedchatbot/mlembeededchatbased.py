
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# Load CSV
data = pd.read_csv("MLEmbeddingbasedchatbot/faq.csv")

# Load embedding model (local HuggingFace model)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode questions
embeddings = model.encode(data['question'].tolist())

# Store in FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings))

def chatbot(query):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k=1)  # get nearest answer
    return data.iloc[I[0][0]]['answer']

# Chat loop
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    print("Bot:", chatbot(query))
