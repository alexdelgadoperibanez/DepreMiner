#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import configparser
from collections import Counter, defaultdict
from pymongo import MongoClient

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

##############################################################################
# 1) Lectura de config
##############################################################################

def load_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "config", "config.conf")
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    return cfg

##############################################################################
# 2) Funciones para cargar datos de Mongo
##############################################################################

def load_processed_docs(uri, db_name, collection_name):
    """
    Retorna un cursor (o lista) de documentos con 'processed'.
    'processed' se asume es una lista de oraciones, 
    donde cada oración es una lista de tokens (strings).
    """
    client = MongoClient(uri)
    db = client[db_name]
    coll = db[collection_name]

    cursor = coll.find({"processed": {"$exists": True, "$ne": []}})
    # Si esperas muchos docs, quizás quieras iterar en streaming, pero 
    # por sencillez convertimos a lista:
    docs = list(cursor)
    return docs

##############################################################################
# 3) Análisis de Frecuencia de Términos
##############################################################################

def term_frequency(docs):
    """
    Calcula cuántas veces aparece cada token en todo el corpus.
    docs: lista de documentos, cada uno con doc["processed"] = [ [tokens], [tokens], ... ]
    Retorna un Counter con la frecuencia global de cada término.
    """
    counter = Counter()
    for doc in docs:
        processed = doc["processed"]  # lista de oraciones
        for sentence in processed:
            for token in sentence:
                counter[token] += 1
    return counter

##############################################################################
# 4) Análisis de Co-ocurrencia (Bigramas dentro de la misma oración)
##############################################################################

def co_occurrence(docs):
    """
    Extrae bigramas que aparecen en la misma oración. 
    Retorna un Counter con el par (token1, token2) -> frecuencia.
    """
    pair_counter = Counter()
    for doc in docs:
        processed = doc["processed"]
        for sentence in processed:
            # Generamos pares adyacentes (bigramas)
            for i in range(len(sentence) - 1):
                t1 = sentence[i]
                t2 = sentence[i+1]
                pair_counter[(t1, t2)] += 1
    return pair_counter

##############################################################################
# 5) Análisis de Tendencias Temporales
##############################################################################

def yearly_term_frequency(docs, year_field="year"):
    """
    Supongamos que cada doc tiene doc["year"] (o doc["date"] que convertiste a un año).
    Calculamos la frecuencia de cada token por año, devolviendo un dict:
      { <year>: Counter({token: freq, ...}), ... }
    Si no existe 'year_field' en el doc, lo ignoramos o lo situamos en 'unknown'.
    """
    freq_by_year = defaultdict(Counter)
    for doc in docs:
        processed = doc["processed"]
        year = doc.get(year_field, "unknown")  # Si no existe, "unknown" o None
        # Contar tokens
        for sentence in processed:
            for token in sentence:
                freq_by_year[year][token] += 1
    return freq_by_year

##############################################################################
# 6) Ejemplo de uso
##############################################################################

if __name__ == "__main__":
    # 1) Cargar config
    cfg = load_config()
    mongo_uri = cfg["db"].get("uri", "mongodb://localhost:27017")
    db_name = cfg["db"].get("db_name", "PubMedDB")
    collection_name = cfg["db"].get("collection_name", "major_depression_abstracts")

    # 2) Cargar documentos con processed
    docs = load_processed_docs(mongo_uri, db_name, collection_name)
    print(f"[INFO] Se han cargado {len(docs)} documentos de Mongo con campo 'processed'.")

    # 3) Frecuencia de términos global
    freq = term_frequency(docs)
    print(f"[INFO] Se han encontrado {len(freq)} términos distintos en total.")

    # Muestra los 10 términos más frecuentes
    print("\n=== TÉRMINOS MÁS FRECUENTES ===")
    for token, count in freq.most_common(10):
        print(f"{token}: {count}")

    # 4) Co-ocurrencia (bigramas)
    bigrams = co_occurrence(docs)
    print(f"\n[INFO] Se han encontrado {len(bigrams)} bigramas distintos.")
    print("=== BIGRAMAS MÁS FRECUENTES ===")
    for pair, count in bigrams.most_common(10):
        print(f"{pair}: {count}")

    # 5) Tendencias temporales (opcional)
    # Asumiendo que tus docs tienen un campo doc["year"]
    freq_year = yearly_term_frequency(docs, year_field="year")
    print(f"\n[INFO] Se han identificado {len(freq_year)} años distintos.")
    # Ejemplo: para 2022, ver top 10 tokens
    if "2022" in freq_year:
        print("\n=== TÉRMINOS MÁS FRECUENTES EN 2022 ===")
        c_2022 = freq_year["2022"]
        for token, count in c_2022.most_common(10):
            print(f"{token}: {count}")
    else:
        print("[WARN] Ningún documento con year=2022 encontrado.")


    def flatten_docs(docs):
        return [" ".join(token for sent in doc["processed"] for token in sent) for doc in docs]


    def compute_tfidf(docs, max_features=1000):
        texts = flatten_docs(docs)
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(texts)
        return vectorizer, tfidf_matrix


    def tfidf_by_year(docs):
        grouped = defaultdict(list)
        for doc in docs:
            year = doc.get("year", "unknown")
            grouped[year].append(" ".join(token for sent in doc["processed"] for token in sent))
        result = {}
        for year, texts in grouped.items():
            vectorizer = TfidfVectorizer(max_features=500)
            tfidf = vectorizer.fit_transform(texts)
            result[year] = (vectorizer, tfidf)
        return result
