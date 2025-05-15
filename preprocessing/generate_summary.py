import os
import configparser
from pymongo import MongoClient
from transformers import pipeline
from tqdm import tqdm
from typing import List, Dict, Any

def load_config(config_path: str) -> configparser.ConfigParser:
    """
    Carga el archivo de configuración y devuelve el objeto ConfigParser.

    Args:
        config_path (str): Ruta del archivo de configuración.

    Returns:
        configparser.ConfigParser: Objeto de configuración cargado.
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

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

def initialize_summarizer(model_name: str = "sshleifer/distilbart-cnn-12-6") -> Any:
    """
    Inicializa el modelo de resumen (summarizer).

    Args:
        model_name (str): Nombre del modelo de resumen.

    Returns:
        Any: Pipeline de resumen.
    """
    print("📦 Cargando modelo de resumen...")
    return pipeline("summarization", model=model_name)

def generate_summaries(coll: MongoClient, summarizer: Any, max_length: int = 80, min_length: int = 30) -> None:
    """
    Genera resúmenes para documentos sin resumen en la colección de MongoDB.

    Args:
        coll (MongoClient): Colección de MongoDB.
        summarizer (Any): Pipeline de resumen.
        max_length (int): Longitud máxima del resumen.
        min_length (int): Longitud mínima del resumen.

    Returns:
        None
    """
    docs = list(coll.find({
        "abstract1": {"$exists": True, "$ne": ""},
        "summary": {"$exists": False}
    }))

    print(f"✍️ Se encontraron {len(docs)} documentos sin resumen.")

    for doc in tqdm(docs, desc="Generando resúmenes"):
        pmid = doc.get("pmid")
        text = doc.get("abstract1", "") + doc.get("abstract2", "")

        if len(text.split()) < 30:
            summary = "Resumen no generado (abstract demasiado corto)."
        else:
            try:
                result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
                summary = result[0]["summary_text"]
            except Exception as e:
                summary = f"Error al generar resumen: {e}"

        coll.update_one({"_id": doc["_id"]}, {"$set": {"summary": summary}})

if __name__ == "__main__":
    # === Cargar configuración ===
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.conf")
    config = load_config(config_path)

    # Parámetros de base de datos
    MONGO_URI = config["db"].get("uri", "mongodb://localhost:27017")
    DB_NAME = config["db"].get("db_name", "PubMedDB")
    COLLECTION_NAME = config["db"].get("collection_name", "major_depression_abstracts")

    # Inicializar conexión Mongo
    coll = connect_to_mongo(MONGO_URI, DB_NAME, COLLECTION_NAME)

    # Inicializar modelo de resumen
    summarizer = initialize_summarizer()

    # Generar resúmenes
    generate_summaries(coll, summarizer)
