
# PubMed TFM - Análisis de Abstracts de Depresión Mayor

Este proyecto permite realizar análisis avanzados de textos biomédicos de PubMed relacionados con la depresión mayor y sus tratamientos farmacológicos. Incluye extracción de textos, procesamiento de lenguaje natural (NER), análisis de métricas de texto, generación de embeddings y búsqueda semántica, así como un chatbot biomédico basado en BioGPT.

---

## 🚀 Estructura del Proyecto

```
TFM_UOC/
├── api/
│   └── pubmed_api.py          # Extracción de abstracts desde PubMed
├── config/
│   └── config.conf            # Archivo de configuración del proyecto
├── embedding/
│   ├── faiss_search.py        # Generación de embeddings y búsqueda semántica
│   └── search_engine.py       # Motor de búsqueda semántica con FAISS
├── mining/
│   ├── metrics.py             # Análisis de métricas de texto (TF, TF-IDF, WordCloud)
│   └── text_mining.py         # Minería de texto y co-ocurrencias
├── ner/
│   ├── ner.py                 # Extracción de entidades (NER)
│   ├── view_entities.py       # Visualización de entidades NER
│   └── visualization.py       # Visualización gráfica de entidades
├── output/                    # Carpeta de salida para CSVs, imágenes, vectores
├── preprocessing/
│   ├── preprocessor.py        # Limpieza y preprocesamiento de textos
│   └── generate_summary.py    # Generación de resúmenes automáticos
├── query/
│   └── search_query.txt       # Consulta de búsqueda para PubMed
├── web_app/
│   ├── app.py                 # Aplicación web (Streamlit)
│   └── utils/                 # Utilidades para la app
└── sync_components.py         # Script maestro para gestionar el flujo del proyecto
```

---

## ⚡ Requisitos

Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

- **MongoDB**: Asegúrate de tener MongoDB ejecutando en `mongodb://localhost:27017`.
- **Configuración:** Los parámetros de conexión y configuración están definidos en `config/config.conf`.

---

## 🚀 Uso del Proyecto

### 1️⃣ Sincronización Completa (Automática)

```bash
python sync_components.py --refresh --ner --preprocess --metrics --embeddings --summary
```

### 2️⃣ Ejecución Manual de Scripts

#### 🔹 Extracción de Abstracts (PubMed)
```bash
python api/pubmed_api.py
```

#### 🔹 Extracción de Entidades (NER)
```bash
python ner/ner.py
```

#### 🔹 Preprocesamiento de Abstracts
```bash
python preprocessing/preprocessor.py
```

#### 🔹 Generación de Métricas de Texto
```bash
python mining/metrics.py
```

#### 🔹 Generación de Embeddings y Búsqueda Semántica
```bash
python embedding/faiss_search.py
```

#### 🔹 Generación de Resúmenes Automáticos
```bash
python preprocessing/generate_summary.py
```

---

## 🌐 Aplicación Web (Streamlit)

```bash
streamlit run web_app/app.py
```

---

## 🔧 Configuración (config/config.conf)

```ini
[db]
uri = mongodb://localhost:27017
db_name = PubMedDB
collection_name = major_depression_abstracts

[pubmed]
email = TU_CORREO@ejemplo.com
batch_size = 100

[preprocessor]
spacy_model = en_core_sci_sm
```

---

## ✅ Output (Resultados)

Los resultados se guardan en la carpeta `output/` y se sincronizan automáticamente con `web_app/data`.

---

## 📌 Autor

- **Nombre:** Alejandro Delgado Peribáñez
- **Máster en Bioinformática & Bioestadistica - UOC**

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT.