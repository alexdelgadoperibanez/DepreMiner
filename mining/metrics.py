
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

docs = list(coll.find({"processed": {"$exists": True, "$ne": []}}))
print(f"Documentos cargados: {len(docs)}")

# --- FRECUENCIA DE TÉRMINOS ---
counter = Counter()
for doc in docs:
    for sent in doc["processed"]:
        counter.update(sent)

freq_df = pd.DataFrame(counter.most_common(50), columns=["token", "freq"])
freq_df.to_csv(os.path.join(OUTPUT_DIR, "term_frequencies.csv"), index=False)

# --- WORDCLOUD ---
wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(counter)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud - Términos más frecuentes")
plt.savefig(os.path.join(OUTPUT_DIR, "wordcloud.png"))
plt.show()

# --- TF-IDF GLOBAL ---
texts = [" ".join(token for sent in doc["processed"] for token in sent) for doc in docs]
vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(texts)
features = vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=features)
df_tfidf.to_csv(os.path.join(OUTPUT_DIR, "tfidf_matrix.csv"), index=False)

# --- CLUSTERING ---
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(tfidf_matrix)
df_clusters = pd.DataFrame({"pmid": [doc.get("pmid") for doc in docs], "cluster": clusters})
df_clusters.to_csv(os.path.join(OUTPUT_DIR, "clusters.csv"), index=False)
print("Clustering completado y guardado.")
