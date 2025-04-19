
# PubMed Mining Project

Este proyecto permite realizar análisis de texto y visualización de entidades a partir de abstracts de PubMed relacionados con la depresión mayor y tratamientos farmacológicos.

## Estructura

```
pubmed_project_scripts/
├── ner/
│   └── view_entities.py         # Visualiza entidades NER desde Mongo
├── text_mining/
│   └── metrics.py               # Métricas: TF, WordCloud, TF-IDF, clustering
├── embedding/
│   └── faiss_search.py          # Embeddings y búsqueda semántica
├── output/                      # Carpeta de salida para CSVs, imágenes, vectores
```

## Requisitos

Instala las dependencias necesarias:

```bash
pip install pymongo pandas matplotlib seaborn scikit-learn wordcloud faiss-cpu sentence-transformers
```

## Conexión

Todos los scripts se conectan por defecto a:

- Mongo URI: `mongodb://localhost:27017`
- Base de datos: `PubMedDB`
- Colección: `major_depression_abstracts`

Asegúrate de que los abstracts estén previamente almacenados con los campos `abstract`, `processed`, `entities`.

---

## Scripts

### 🔹 `ner/view_entities.py`

Imprime por consola las entidades reconocidas por los modelos biomédicos.

```bash
python ner/view_entities.py
```

---

### 🔹 `text_mining/metrics.py`

Genera:

- `output/term_frequencies.csv`
- `output/wordcloud.png`
- `output/tfidf_matrix.csv`
- `output/clusters.csv`

Y muestra la nube de palabras al ejecutar.

```bash
python text_mining/metrics.py
```

---

### 🔹 `embedding/faiss_search.py`

- Crea `embeddings.npy` y `embedding_pmids.csv`
- Muestra los 5 abstracts más similares a una frase de búsqueda

```bash
python embedding/faiss_search.py
```

---

## Output

Todos los resultados relevantes se guardan en la carpeta `output/`.

