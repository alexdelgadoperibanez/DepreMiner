# 🧠 DepreMiner – Análisis de Abstracts de Depresión Mayor en PubMed

Este proyecto permite realizar análisis avanzados de textos biomédicos de PubMed relacionados con la depresión mayor y sus tratamientos farmacológicos. Incluye extracción de textos, procesamiento de lenguaje natural (NER), análisis de métricas de texto, generación de embeddings y búsqueda semántica, así como un chatbot biomédico basado en BioGPT.

---

## 🗂️ Estructura del Proyecto

```
DepreMiner/
├── api/                   # Extracción de abstracts desde PubMed
├── ner/                   # Extracción de entidades con modelos NER
├── preprocessing/         # Preprocesamiento de texto
├── mining/                # Análisis de métricas y clustering
├── embedding/             # Generación de embeddings y búsqueda semántica
├── web_app/               # Aplicación web para visualización
├── config/                # Archivos de configuración
├── query/                 # Consultas para PubMed
├── scripts/               # Scripts adicionales
├── output/                # Resultados intermedios
├── sync_components.py     # Orquestador de tareas
├── app_runner.py          # Lanzador de la aplicación web
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Este archivo
```

---

## ⚙️ Requisitos

- Python 3.8 o superior
- MongoDB en ejecución local (`mongodb://localhost:27017`)
- Conexión a internet para descargar modelos y datos de PubMed

---

## 🚀 Instalación

1. **Clonar el repositorio:**

   ```bash
   git clone https://github.com/alexdelgadoperibanez/DepreMiner.git
   cd DepreMiner
   ```

2. **Crear y activar un entorno virtual:**

   ```bash
   python -m venv env
   source env/bin/activate  # En Windows: env\Scripts\activate
   ```

3. **Instalar las dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar el archivo de configuración:**

   Copiar el archivo de ejemplo y editar según sea necesario:

   ```bash
   cp config/config_example.conf config/config.conf
   ```

   Asegúrate de configurar correctamente los parámetros como el correo electrónico para Entrez, la URI de MongoDB, y los modelos NER a utilizar.

---

## 🔄 Ejecución del Flujo Completo

Para ejecutar todo el flujo de procesamiento, desde la extracción de datos hasta la generación de embeddings y métricas, utiliza el script `sync_components.py` con el flag `--all`:

```bash
python sync_components.py --all
```

Este comando ejecutará secuencialmente las siguientes tareas:

- Extracción de abstracts desde PubMed
- Extracción de entidades con modelos NER
- Preprocesamiento de texto
- Análisis de métricas y clustering
- Generación de embeddings y búsqueda semántica
- Generación de resúmenes

---

## 🧩 Ejecución de Componentes Individuales

También puedes ejecutar componentes específicos utilizando los siguientes flags:

- `--refresh`: Extraer abstracts desde PubMed
- `--ner`: Ejecutar extracción de entidades NER
- `--preprocess`: Preprocesar textos
- `--metrics`: Generar métricas y clustering
- `--embeddings`: Generar embeddings y búsqueda semántica
- `--summary`: Generar resúmenes

Ejemplo:

```bash
python sync_components.py --ner --preprocess
```

---

## 🌐 Ejecutar la Aplicación Web

Una vez que hayas procesado los datos, puedes lanzar la aplicación web para visualizar los resultados:

```bash
python app_runner.py --mode local
```

Esto iniciará un servidor local accesible en `http://localhost:8501`, donde podrás explorar las visualizaciones y análisis generados.

---

## 📌 Autor

- **Nombre:** Alejandro Delgado Peribáñez
- **Máster en Bioinformática & Bioestadistica - UOC**

---

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.