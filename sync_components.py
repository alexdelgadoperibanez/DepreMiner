import os
import shutil
import subprocess
import sys

def run_and_sync(script_path, output_folder, target_folder, files_to_copy):
    subprocess.run(["python", script_path], check=True)
    os.makedirs(target_folder, exist_ok=True)
    for fname in files_to_copy:
        src = os.path.join(output_folder, fname)
        dst = os.path.join(target_folder, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"✅ Copiado: {fname}")
        else:
            print(f"⚠️ No encontrado: {fname}")

# === FLAGS activados por argumentos ===
args = [arg.lower() for arg in sys.argv]

do_refresh = "--refresh" in args
do_ner = "--ner" in args
do_preprocess = "--preprocess" in args
do_metrics = "--metrics" in args
do_embeddings = "--embeddings" in args
do_summary = "--summary" in args

print("🚀 Iniciando sincronización con configuración dinámica...\n")

# 0. Refresh desde PubMed
if do_refresh:
    print("📥 Actualizando abstracts desde PubMed...")
    subprocess.run(["python", "api/pubmed_api.py"], check=True)

# 1. NER
if do_ner:
    print("🧠 Ejecutando extracción de entidades NER...")
    subprocess.run(["python", "ner/ner.py"], check=True)

# 2. Preprocesamiento
if do_preprocess:
    print("🔧 Ejecutando preprocesamiento...")
    subprocess.run(["python", "preprocessing/preprocessor.py"], check=True)

# 3. Métricas
if do_metrics:
    print("📊 Ejecutando métricas de texto...")
    run_and_sync(
        script_path="mining/metrics.py",
        output_folder="mining/output",
        target_folder="web_app/data",
        files_to_copy=[
            "clusters.csv", "term_frequencies.csv", "tfidf_matrix.csv", "wordcloud.png"
        ]
    )

# 4. Embeddings
if do_embeddings:
    print("🔍 Ejecutando generación de embeddings y FAISS...")
    run_and_sync(
        script_path="embedding/faiss_search.py",
        output_folder="embedding/output",
        target_folder="web_app/data",
        files_to_copy=[
            "embeddings.npy", "embedding_pmids.csv", "embedding_texts.csv", "semantic_search_results.csv"
        ]
    )

# 5. Resúmenes
if do_summary:
    print("✍️ Generando resúmenes automáticos...")
    subprocess.run(["python", "scripts/generate_summary.py"], check=True)

print("\n✅ Finalizado. Si quieres lanzar la app, ejecuta:")
print("   streamlit run web_app/app.py")
