import os
import pandas as pd
import matplotlib.pyplot as plt
from pymongo import MongoClient
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "PubMedDB"
COLLECTION = "major_depression_abstracts"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = MongoClient(MONGO_URI)
coll = client[DB_NAME][COLLECTION]

def load_documents() -> list[dict]:
    """
    Carga documentos procesados desde MongoDB.

    Returns:
        list[dict]: Lista de documentos procesados.
    """
    docs = list(coll.find({"processed": {"$exists": True, "$ne": []}}))
    print(f"Documentos cargados: {len(docs)}")
    return docs

def generate_term_frequencies(docs: list[dict]) -> pd.DataFrame:
    """
    Genera un DataFrame de frecuencias de términos a partir de los documentos.

    Args:
        docs (list[dict]): Lista de documentos procesados.

    Returns:
        pd.DataFrame: DataFrame con términos y sus frecuencias.
    """
    counter = Counter()
    for doc in docs:
        for sent in doc["processed"]:
            counter.update(sent)

    freq_df = pd.DataFrame(counter.most_common(50), columns=["token", "freq"])
    freq_df.to_csv(os.path.join(OUTPUT_DIR, "term_frequencies.csv"), index=False)
    return freq_df

def generate_wordcloud(counter: Counter) -> None:
    """
    Genera y guarda una imagen de WordCloud basada en las frecuencias de términos.

    Args:
        counter (Counter): Contador de frecuencias de términos.
    """
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(counter)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("WordCloud - Términos más frecuentes")
    plt.savefig(os.path.join(OUTPUT_DIR, "wordcloud.png"))
    plt.show()

def generate_tfidf_matrix(docs: list[dict]) -> pd.DataFrame:
    """
    Calcula y guarda la matriz TF-IDF a partir de los textos de los documentos.

    Args:
        docs (list[dict]): Lista de documentos procesados.

    Returns:
        pd.DataFrame: DataFrame con la matriz TF-IDF.
    """
    texts = [" ".join(token for sent in doc["processed"] for token in sent) for doc in docs]
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    features = vectorizer.get_feature_names_out()
    df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=features)
    df_tfidf.to_csv(os.path.join(OUTPUT_DIR, "tfidf_matrix.csv"), index=False)
    return df_tfidf

def perform_clustering(tfidf_matrix: pd.DataFrame, docs: list[dict]) -> pd.DataFrame:
    """
    Realiza clustering de los documentos y guarda los resultados.

    Args:
        tfidf_matrix (pd.DataFrame): Matriz TF-IDF.
        docs (list[dict]): Lista de documentos procesados.

    Returns:
        pd.DataFrame: DataFrame con PMIDs y sus clusters asignados.
    """
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    df_clusters = pd.DataFrame({"pmid": [doc.get("pmid") for doc in docs], "cluster": clusters})
    df_clusters.to_csv(os.path.join(OUTPUT_DIR, "clusters.csv"), index=False)
    print("Clustering completado y guardado.")
    return df_clusters

if __name__ == "__main__":
    print("📊 Ejecutando análisis de métricas...")

    docs = load_documents()
    if not docs:
        print("⚠️ No hay documentos procesados en la colección.")
        exit(1)

    # Frecuencia de términos
    freq_df = generate_term_frequencies(docs)
    print("✅ Frecuencias de términos guardadas.")

    # WordCloud
    counter = Counter(dict(zip(freq_df["token"], freq_df["freq"])))
    generate_wordcloud(counter)
    print("✅ WordCloud generada.")

    # TF-IDF
    tfidf_df = generate_tfidf_matrix(docs)
    print("✅ Matriz TF-IDF guardada.")

    # Clustering
    perform_clustering(tfidf_df, docs)
    print("✅ Clustering completado.")

