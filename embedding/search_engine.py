import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

def build_faiss_index(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> (
        tuple)[faiss.IndexFlatL2, np.ndarray, SentenceTransformer]:
    """
    Construye un índice FAISS y genera embeddings utilizando el modelo especificado.

    Args:
        texts (list[str]): Lista de textos para generar embeddings.
        model_name (str): Nombre del modelo de SentenceTransformer a utilizar.

    Returns:
        tuple[faiss.IndexFlatL2, np.ndarray, SentenceTransformer]:
        Índice FAISS, matriz de embeddings y el modelo de embeddings.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings, model

def search(query: str, model: SentenceTransformer, index: faiss.IndexFlatL2, texts: list[str], k: int = 100) -> (
        tuple)[list[str], np.ndarray]:
    """
    Realiza una búsqueda semántica en el índice FAISS utilizando el modelo especificado.

    Args:
        query (str): Consulta de texto para buscar.
        model (SentenceTransformer): Modelo de embeddings para convertir la consulta.
        index (faiss.IndexFlatL2): Índice FAISS para realizar la búsqueda.
        texts (list[str]): Lista de textos para recuperar los resultados.
        k (int): Número de resultados a devolver.

    Returns:
        tuple[list[str], np.ndarray]: Lista de textos encontrados y sus distancias.
    """
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)
    return [texts[i] for i in I[0]], D[0]
