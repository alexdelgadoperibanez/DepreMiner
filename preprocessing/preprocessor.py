import os
import re
import time
import configparser
import spacy
from pymongo import MongoClient
from spacy.lang.en import STOP_WORDS
from typing import List


def load_config(config_path: str) -> configparser.ConfigParser:
    """
    Carga el archivo de configuración config.conf y devuelve el objeto ConfigParser.

    Args:
        config_path (str): Ruta del archivo de configuración.

    Returns:
        configparser.ConfigParser: Objeto de configuración cargado.
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    return cfg


def load_spacy_model(model_name: str) -> spacy.language.Language:
    """
    Carga el modelo de spaCy especificado.

    Args:
        model_name (str): Nombre del modelo de spaCy a cargar.

    Returns:
        spacy.language.Language: Modelo spaCy cargado.
    """
    print(f"[INFO] Cargando modelo spaCy: {model_name}")
    return spacy.load(model_name)


def clean_text(text: str) -> str:
    """
    Limpia el texto eliminando caracteres no deseados.

    Args:
        text (str): Texto a limpiar.

    Returns:
        str: Texto limpio.
    """
    text = text.replace("/", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_and_remove_symbols(doc: str, nlp: spacy.language.Language) -> List[List[str]]:
    """
    Segmenta el texto en oraciones y tokeniza cada oración.

    Args:
        doc (str): Texto a procesar.
        nlp (spacy.language.Language): Modelo spaCy para tokenización.

    Returns:
        List[List[str]]: Lista de oraciones tokenizadas.
    """
    spacy_doc = nlp(doc)
    sentences = []
    for sent in spacy_doc.sents:
        tokens_clean = []
        for token in sent:
            lower_text = token.text.lower()
            if lower_text in STOP_WORDS:
                continue
            cleaned = re.sub(r'[^a-z0-9]', '', lower_text)
            if len(cleaned) < 3:
                continue
            tokens_clean.append(cleaned)
        if tokens_clean:
            sentences.append(tokens_clean)
    return sentences


def preprocess_abstract(abstract: str, nlp: spacy.language.Language) -> List[List[str]]:
    """
    Preprocesa un abstract:
    1. Limpieza del texto.
    2. Tokenización y eliminación de símbolos no alfanuméricos.

    Args:
        abstract (str): Texto del abstract.
        nlp (spacy.language.Language): Modelo spaCy para tokenización.

    Returns:
        List[List[str]]: Lista de oraciones procesadas.
    """
    abstract_clean = clean_text(abstract)
    return tokenize_and_remove_symbols(abstract_clean, nlp)


def preprocess_and_update_mongo(
        db_name: str, collection_name: str, uri: str, nlp: spacy.language.Language
) -> None:
    """
    Preprocesa abstracts de documentos en MongoDB y actualiza cada documento.

    Args:
        db_name (str): Nombre de la base de datos.
        collection_name (str): Nombre de la colección.
        uri (str): URI de conexión a MongoDB.
        nlp (spacy.language.Language): Modelo spaCy para tokenización.

    Returns:
        None
    """
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
    """
    Función principal del script:
    1. Carga la configuración.
    2. Carga el modelo spaCy.
    3. Preprocesa los abstracts de MongoDB y los actualiza.
    """
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.conf")
    cfg = load_config(config_path)

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