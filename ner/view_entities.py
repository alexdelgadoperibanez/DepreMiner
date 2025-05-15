import os
from pymongo import MongoClient
import pandas as pd
from typing import List, Dict, Any

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "PubMedDB"
COLLECTION = "major_depression_abstracts"

def connect_to_mongo(uri: str, db_name: str, collection_name: str) -> MongoClient:
    """
    Establece una conexión a MongoDB y devuelve la colección especificada.

    Args:
        uri (str): URI de conexión a MongoDB.
        db_name (str): Nombre de la base de datos.
        collection_name (str): Nombre de la colección.

    Returns:
        MongoClient: Conexión a la colección especificada.
    """
    client = MongoClient(uri)
    db = client[db_name]
    return db[collection_name]

def load_entities(collection) -> List[Dict[str, Any]]:
    """
    Carga documentos con entidades extraídas desde MongoDB.

    Args:
        collection (MongoClient): Colección de MongoDB.

    Returns:
        List[Dict[str, Any]]: Lista de documentos con entidades.
    """
    docs = list(collection.find({"entities": {"$exists": True, "$ne": []}}))
    print(f"Documentos con entidades: {len(docs)}\n")
    return docs

def display_entities(docs: List[Dict[str, Any]], num_docs: int = 5) -> None:
    """
    Muestra las entidades extraídas de los primeros documentos.

    Args:
        docs (List[Dict[str, Any]]): Lista de documentos con entidades.
        num_docs (int): Número de documentos a mostrar.
    """
    for doc in docs[:num_docs]:
        print(f"PMID: {doc.get('pmid', 'N/A')}")
        print(f"Title: {doc.get('title', '')}")
        print("=" * 60)
        df = pd.json_normalize(doc["entities"])
        print(df[["entity_group", "word", "occurrences", "overall_combined_score", "models"]])
        print("\n")

if __name__ == "__main__":
    # Conectar a MongoDB y cargar documentos con entidades
    coll = connect_to_mongo(MONGO_URI, DB_NAME, COLLECTION)
    docs_with_entities = load_entities(coll)

    # Mostrar entidades de los primeros documentos
    display_entities(docs_with_entities)
