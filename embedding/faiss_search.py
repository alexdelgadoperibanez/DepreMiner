import os
import numpy as np
import faiss
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import pandas as pd

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "PubMedDB"
COLLECTION = "major_depression_abstracts"
OUTPUT_DIR = "output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = MongoClient(MONGO_URI)
coll = client[DB_NAME][COLLECTION]

# Recuperar documentos y reconstruir el texto real usado en NER
docs = list(coll.find({"processed": {"$exists": True, "$ne": []}}))
texts = [doc.get("abstract1", "") + doc.get("abstract2", "") for doc in docs]
pmids = [doc.get("pmid") for doc in docs]

# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Guardar embeddings y PMIDs
np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
pd.DataFrame({"pmid": pmids}).to_csv(os.path.join(OUTPUT_DIR, "embedding_pmids.csv"), index=False)
pd.DataFrame({"pmid": pmids, "text": texts}).to_csv(os.path.join(OUTPUT_DIR, "embedding_texts.csv"), index=False)

# Búsqueda de prueba
query = "efficacy of SSRIs in older adults"
q_embed = model.encode([query])
D, I = index.search(np.array(q_embed), k=5)

print(f"\n=== Resultados para query: '{query}' ===\n")

results = []
for idx, dist in zip(I[0], D[0]):
    title = docs[idx].get("title", "")
    snippet = texts[idx][:300]
    pmid = docs[idx].get("pmid", "")
    print("-", title)
    print(snippet, "...\n")
    print("PMID:", pmid)
    print("Distance:", dist)
    print()
    results.append({
        "pmid": pmid,
        "title": title,
        "abstract_snippet": snippet,
        "distance": dist
    })

# Guardar resultados en CSV
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(OUTPUT_DIR, "semantic_search_results.csv"), index=False)

print("✅ Embeddings y resultados de búsqueda guardados.")
