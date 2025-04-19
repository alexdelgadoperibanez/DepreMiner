import os
import configparser
from pymongo import MongoClient
from transformers import pipeline
from tqdm import tqdm

# === Cargar configuración ===
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.conf")
config.read(config_path)

# Parámetros de base de datos
MONGO_URI = config["db"].get("uri", "mongodb://localhost:27017")
DB_NAME = config["db"].get("db_name", "PubMedDB")
COLLECTION_NAME = config["db"].get("collection_name", "major_depression_abstracts")

# Inicializar conexión Mongo
client = MongoClient(MONGO_URI)
coll = client[DB_NAME][COLLECTION_NAME]

# Inicializar modelo de resumen
print("📦 Cargando modelo de resumen...")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Buscar documentos sin resumen
docs = list(coll.find({
    "abstract1": {"$exists": True, "$ne": ""},
    "summary": {"$exists": False}
}))

print(f"✍️ Se encontraron {len(docs)} documentos sin resumen.")

for doc in tqdm(docs):
    pmid = doc.get("pmid")
    text = doc.get("abstract1", "") + doc.get("abstract2", "")
    if len(text.split()) < 30:
        summary = "Resumen no generado (abstract demasiado corto)."
    else:
        try:
            result = summarizer(text, max_length=80, min_length=30, do_sample=False)
            summary = result[0]["summary_text"]
        except Exception as e:
            summary = f"Error al generar resumen: {e}"

    coll.update_one({"_id": doc["_id"]}, {"$set": {"summary": summary}})
