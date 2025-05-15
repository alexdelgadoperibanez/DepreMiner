import numpy as np
import pandas as pd
import faiss
import os
from typing import Tuple, List, Dict, Any


def load_index_and_docs(
        embedding_path: str = "web_app/data/embeddings.npy",
        pmid_path: str = "web_app/data/embedding_pmids.csv",
        docs_path: str = "web_app/data/embedding_texts.csv"
) -> Tuple[faiss.IndexFlatL2, List[str], List[str]]:
    """
    Carga el índice FAISS y los textos asociados a los embeddings.

    Args:
        embedding_path (str): Ruta del archivo de embeddings (.npy).
        pmid_path (str): Ruta del archivo CSV con los PMIDs.
        docs_path (str): Ruta del archivo CSV con los textos.

    Returns:
        Tuple[faiss.IndexFlatL2, List[str], List[str]]:
            - Índice FAISS cargado.
            - Lista de textos asociados a los embeddings.
            - Lista de PMIDs asociados.
    """
    print("📦 Cargando embeddings y construyendo índice FAISS...")
    embeddings = np.load(embedding_path)
    pmid_df = pd.read_csv(pmid_path)
    pmids = pmid_df["pmid"].tolist()

    if docs_path and os.path.exists(docs_path):
        texts_df = pd.read_csv(docs_path)
        texts = texts_df["text"].tolist()
    else:
        texts = ["Texto no disponible" for _ in pmids]
        print("[WARN] Textos no disponibles. Usando placeholders.")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    print("✅ Índice FAISS cargado correctamente.")

    return index, texts, pmids


def search_similar(
        query: str, model: Any, index: faiss.IndexFlatL2, texts: List[str], pmids: List[str], k: int = 100
) -> List[Dict[str, Any]]:
    """
    Realiza una búsqueda semántica en el índice FAISS y devuelve los resultados.

    Args:
        query (str): Consulta de búsqueda.
        model (Any): Modelo para generar el embedding de la consulta.
        index (faiss.IndexFlatL2): Índice FAISS para la búsqueda.
        texts (List[str]): Lista de textos asociados a los embeddings.
        pmids (List[str]): Lista de PMIDs asociados.
        k (int): Número de resultados a devolver.

    Returns:
        List[Dict[str, Any]]: Lista de resultados de búsqueda, cada uno con:
            - pmid (str): Identificador del documento.
            - distance (float): Distancia de similitud.
            - snippet (str): Fragmento del texto.
            - title (str): Título generado (PMID).
    """
    print("🔍 Realizando búsqueda semántica...")
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)

    results = []
    for i, dist in zip(I[0], D[0]):
        results.append({
            "pmid": pmids[i],
            "distance": dist,
            "snippet": texts[i][:300] if len(texts[i]) > 0 else "Texto no disponible",
            "title": f"PMID {pmids[i]}"
        })

    print(f"✅ Búsqueda completada. {len(results)} resultados encontrados.")
    return results
