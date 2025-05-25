import os
import numpy as np
import faiss
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import pandas as pd

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "PubMedDB"
COLLECTION = "major_depression_abstracts"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = MongoClient(MONGO_URI)
coll = client[DB_NAME][COLLECTION]

def load_documents_from_mongo() -> tuple[list[str], list[str]]:
    """
    Recupera documentos y sus PMIDs desde MongoDB.

    Returns:
        tuple[list[str], list[str]]: Una tupla que contiene listas de textos y PMIDs.
    """
    docs = list(coll.find({"processed": {"$exists": True, "$ne": []}}))
    texts = [doc.get("abstract1", "") + doc.get("abstract2", "") for doc in docs]
    pmids = [doc.get("pmid") for doc in docs]
    return texts, pmids

def create_and_save_embeddings(texts: list[str]) -> np.ndarray:
    """
    Genera y guarda embeddings para los textos utilizando un modelo de SentenceTransformer.

    Args:
        texts (list[str]): Lista de textos para generar embeddings.

    Returns:
        np.ndarray: Matriz de embeddings generados.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
    return embeddings

def save_pmids_and_texts(pmids: list[str], texts: list[str]) -> None:
    """
    Guarda los PMIDs y textos en archivos CSV.

    Args:
        pmids (list[str]): Lista de PMIDs.
        texts (list[str]): Lista de textos.
    """
    pd.DataFrame({"pmid": pmids}).to_csv(os.path.join(OUTPUT_DIR, "embedding_pmids.csv"), index=False)
    pd.DataFrame({"pmid": pmids, "text": texts}).to_csv(os.path.join(OUTPUT_DIR, "embedding_texts.csv"), index=False)

if __name__ == "__main__":
    print("🔍 Generando embeddings y creando índice FAISS...")

    texts, pmids = load_documents_from_mongo()
    if not texts:
        print("⚠️ No se encontraron textos procesados en MongoDB.")
        exit(1)

    embeddings = create_and_save_embeddings(texts)
    save_pmids_and_texts(pmids, texts)

    print(f"✅ Embeddings guardados en '{OUTPUT_DIR}'")
    print(f"✅ Total de documentos: {len(texts)}")
