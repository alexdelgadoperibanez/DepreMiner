import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

def build_faiss_index(texts, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings, model

def search(query, model, index, texts, k=100):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [texts[i] for i in I[0]], D[0]
