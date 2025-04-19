#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import configparser
import spacy
from pymongo import MongoClient
from spacy.lang.en import STOP_WORDS


def load_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "config", "config.conf")
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    return cfg

def load_spacy_model(model_name: str):
    print(f"[INFO] Cargando modelo spaCy: {model_name}")
    nlp = spacy.load(model_name)
    return nlp

def clean_text(text: str) -> str:
    """
    Limpieza previa:
      - Sustituye / por espacio para separar tokens como 'PURPOSE/BACKGROUND'
      - Reemplaza saltos de línea con espacios
      - Elimina espacios múltiples
    """
    # Reemplazar todas las barras '/' por espacio
    text = text.replace("/", " ")

    # Saltos de línea a espacio
    text = text.replace("\n", " ")

    # Espacios múltiples
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize_and_remove_symbols(doc, nlp) -> list[list[str]]:
    """
    - Segmenta 'doc' en oraciones con spaCy.
    - Para cada token:
      * minúsculas
      * eliminamos todo lo no alfanumérico
      * descartamos tokens vacíos
    """
    spacy_doc = nlp(doc)
    sentences = []
    for sent in spacy_doc.sents:
        tokens_clean = []
        for token in sent:
            # Convertimos a minúsculas
            lower_text = token.text.lower()
            # Quitamos stopwords spaCy
            if lower_text in STOP_WORDS:
                continue
            # Eliminamos símbolos no alfanuméricos
            cleaned = re.sub(r'[^a-z0-9]', '', lower_text)
            # Omitimos vacíos o muy cortos
            if len(cleaned) < 3:
                continue
            tokens_clean.append(cleaned)
        if tokens_clean:
            sentences.append(tokens_clean)
    return sentences

def preprocess_abstract(abstract: str, nlp) -> list[list[str]]:
    """
    1. Limpieza (sustituye '/', saltos de línea, etc.)
    2. Segmentación y tokenización con spaCy
    3. Eliminación de símbolos no alfanuméricos 
    """
    abstract_clean = clean_text(abstract)
    return tokenize_and_remove_symbols(abstract_clean, nlp)

def preprocess_and_update_mongo(db_name, collection_name, uri, nlp):
    client = MongoClient(uri)
    db = client[db_name]
    coll = db[collection_name]

    cursor = coll.find({"abstract": {"$exists": True, "$ne": ""}})
    documents = list(cursor)
    total_docs = len(documents)
    print(f"[INFO] Recuperados {total_docs} documentos con 'abstract' en '{collection_name}'.")

    count_processed = 0
    for doc in documents:
        pmid = doc.get("pmid")
        abstract_text = doc["abstract"]

        processed_sentences = preprocess_abstract(abstract_text, nlp)

        coll.update_one({"_id": doc["_id"]},
                        {"$set": {"processed": processed_sentences}})
        count_processed += 1

    print(f"[INFO] Se han actualizado {count_processed} documentos con el nuevo campo 'processed'.")

def main():
    cfg = load_config()
    spacy_model_name = cfg["preprocessor"].get("spacy_model", "en_core_sci_sm")

    mongo_uri = cfg["db"].get("uri", "mongodb://localhost:27017")
    db_name = cfg["db"].get("db_name", "PubMedDB")
    collection_name = cfg["db"].get("collection_name", "major_depression_abstracts")

    nlp = load_spacy_model(spacy_model_name)

    start_time = time.time()
    preprocess_and_update_mongo(db_name, collection_name, mongo_uri, nlp)
    end_time = time.time()

    print(f"[DONE] Preprocesado completado en {end_time - start_time:.2f} seg.")

if __name__ == "__main__":
    main()
