
import os
from pymongo import MongoClient
import pandas as pd

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "PubMedDB"
COLLECTION = "major_depression_abstracts"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
coll = db[COLLECTION]

docs = list(coll.find({"entities": {"$exists": True, "$ne": []}}))

print(f"Documentos con entidades: {len(docs)}\n")

for doc in docs[:5]:
    print(f"PMID: {doc.get('pmid', 'N/A')}")
    print(f"Title: {doc.get('title', '')}")
    print("=" * 60)
    df = pd.json_normalize(doc["entities"])
    print(df[["entity_group", "word", "occurrences", "overall_combined_score", "models"]])
    print("\n")
