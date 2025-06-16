import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle

df = pd.read_csv("data/patient_therapist.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["prompt"].tolist(), show_progress_bar=True)

# FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save
os.makedirs("model/faiss_index", exist_ok=True)
faiss.write_index(index, "model/faiss_index/therapy.index")
df.to_csv("model/faiss_index/data.csv", index=False)
