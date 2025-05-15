#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import configparser
from collections import Counter, defaultdict
from pymongo import MongoClient

from sklearn.feature_extraction.text import TfidfVectorizer


def load_config() -> configparser.ConfigParser:
    """
    Carga el archivo de configuración config.conf y devuelve el objeto ConfigParser.

    Returns:
        configparser.ConfigParser: Objeto de configuración cargado.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "config", "config.conf")
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    return cfg

def load_processed_docs(uri: str, db_name: str, collection_name: str) -> list[dict]:
    """
    Carga documentos procesados desde MongoDB.

    Args:
        uri (str): URI de conexión a MongoDB.
        db_name (str): Nombre de la base de datos.
        collection_name (str): Nombre de la colección.

    Returns:
        list[dict]: Lista de documentos procesados.
    """
    client = MongoClient(uri)
    db = client[db_name]
    coll = db[collection_name]
    return list(coll.find({"processed": {"$exists": True, "$ne": []}}))

def term_frequency(docs: list[dict]) -> Counter:
    """
    Calcula la frecuencia global de términos en el corpus.

    Args:
        docs (list[dict]): Lista de documentos procesados.

    Returns:
        Counter: Contador de frecuencias de términos.
    """
    counter = Counter()
    for doc in docs:
        for sentence in doc["processed"]:
            counter.update(sentence)
    return counter

def co_occurrence(docs: list[dict]) -> Counter:
    """
    Calcula las co-ocurrencias (bigramas) de términos en las oraciones.

    Args:
        docs (list[dict]): Lista de documentos procesados.

    Returns:
        Counter: Contador de bigramas y su frecuencia.
    """
    pair_counter = Counter()
    for doc in docs:
        for sentence in doc["processed"]:
            for i in range(len(sentence) - 1):
                pair_counter[(sentence[i], sentence[i+1])] += 1
    return pair_counter

def yearly_term_frequency(docs: list[dict], year_field: str = "year") -> dict[str, Counter]:
    """
    Calcula la frecuencia de términos por año.

    Args:
        docs (list[dict]): Lista de documentos procesados.
        year_field (str): Campo que contiene el año en cada documento.

    Returns:
        dict[str, Counter]: Diccionario de frecuencias de términos por año.
    """
    freq_by_year = defaultdict(Counter)
    for doc in docs:
        year = doc.get(year_field, "unknown")
        for sentence in doc["processed"]:
            freq_by_year[year].update(sentence)
    return freq_by_year

def flatten_docs(docs: list[dict]) -> list[str]:
    """
    Aplana los documentos procesados en una lista de textos.

    Args:
        docs (list[dict]): Lista de documentos procesados.

    Returns:
        list[str]: Lista de textos concatenados.
    """
    return [" ".join(token for sent in doc["processed"] for token in sent) for doc in docs]

def compute_tfidf(docs: list[dict], max_features: int = 1000) -> tuple[TfidfVectorizer, list[list[float]]]:
    """
    Calcula la matriz TF-IDF para el corpus de documentos.

    Args:
        docs (list[dict]): Lista de documentos procesados.
        max_features (int): Número máximo de características TF-IDF.

    Returns:
        tuple[TfidfVectorizer, list[list[float]]]: El vectorizador y la matriz TF-IDF.
    """
    texts = flatten_docs(docs)
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts).toarray()
    return vectorizer, tfidf_matrix

def tfidf_by_year(docs: list[dict], max_features: int = 500) -> dict[str, tuple[TfidfVectorizer, list[list[float]]]]:
    """
    Calcula la matriz TF-IDF por año.

    Args:
        docs (list[dict]): Lista de documentos procesados.
        max_features (int): Número máximo de características TF-IDF.

    Returns:
        dict[str, tuple[TfidfVectorizer, list[list[float]]]]: Diccionario con TF-IDF por año.
    """
    grouped = defaultdict(list)
    for doc in docs:
        year = doc.get("year", "unknown")
        grouped[year].append(" ".join(token for sent in doc["processed"] for token in sent))

    result = {}
    for year, texts in grouped.items():
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf = vectorizer.fit_transform(texts).toarray()
        result[year] = (vectorizer, tfidf)

    return result
