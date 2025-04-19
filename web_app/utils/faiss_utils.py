import numpy as np
import pandas as pd
import faiss
import os

def load_index_and_docs(embedding_path="web_app/data/embeddings.npy",
                         pmid_path="web_app/data/embedding_pmids.csv",
                         docs_path="web_app/data/embedding_texts.csv"):
    """
    Carga el índice FAISS y los textos asociados a los embeddings.
    """
    embeddings = np.load(embedding_path)
    pmid_df = pd.read_csv(pmid_path)
    pmids = pmid_df["pmid"].tolist()

    if docs_path and os.path.exists(docs_path):
        texts_df = pd.read_csv(docs_path)
        texts = texts_df["text"].tolist()
    else:
        texts = ["Texto no disponible" for _ in pmids]

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, texts, pmids


def search_similar(query, model, index, texts, pmids, k=100):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)

    results = []
    for i, dist in zip(I[0], D[0]):
        results.append({
            "pmid": pmids[i],
            "distance": dist,
            "snippet": texts[i][:300],
            "title": f"PMID {pmids[i]}"
        })
    return results
