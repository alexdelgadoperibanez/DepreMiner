import os
import shutil
import subprocess
import sys
from typing import List
import configparser


def load_config() -> configparser.ConfigParser:
    """
    Carga el archivo de configuración config.conf y devuelve el objeto ConfigParser.

    Returns:
        configparser.ConfigParser: Objeto de configuración cargado.
    """
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.conf")
    config.read(config_path)
    return config


def run_script(script_path: str) -> None:
    """
    Ejecuta un script de Python.

    Args:
        script_path (str): Ruta del script a ejecutar.
    """
    print(f"🚀 Ejecutando: {script_path}...")
    try:
        subprocess.run(["python", script_path], check=True)
        print(f"✅ Ejecución completada: {script_path}\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar {script_path}: {e}")


def sync_output_files(output_folder: str, target_folder: str, files_to_copy: List[str]) -> None:
    """
    Sincroniza archivos desde la carpeta de salida a la carpeta de destino.

    Args:
        output_folder (str): Carpeta de salida de los archivos generados.
        target_folder (str): Carpeta de destino para sincronizar archivos.
        files_to_copy (List[str]): Lista de archivos a copiar.
    """
    os.makedirs(target_folder, exist_ok=True)
    for fname in files_to_copy:
        src = os.path.join(output_folder, fname)
        dst = os.path.join(target_folder, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"✅ Copiado: {fname} a {target_folder}")
        else:
            print(f"⚠️ No encontrado: {fname}")


def main():
    """
    Script principal que gestiona el flujo del proyecto TFM.
    Permite ejecutar y sincronizar componentes específicos.
    """
    config = load_config()
    print("\n🚀 Gestión del flujo del proyecto TFM\n")

    # === FLAGS activados por argumentos ===
    args = [arg.lower() for arg in sys.argv]
    do_refresh = "--refresh" in args
    do_ner = "--ner" in args
    do_preprocess = "--preprocess" in args
    do_metrics = "--metrics" in args
    do_embeddings = "--embeddings" in args
    do_summary = "--summary" in args

    # === Ejecución de componentes ===
    if do_refresh:
        print("📥 Actualizando abstracts desde PubMed...")
        run_script("api/pubmed_api.py")

    if do_ner:
        print("🧠 Ejecutando extracción de entidades NER...")
        run_script("ner/ner.py")

    if do_preprocess:
        print("🔧 Ejecutando preprocesamiento...")
        run_script("preprocessing/preprocessor.py")

    if do_metrics:
        print("📊 Ejecutando métricas de texto...")
        run_script("mining/metrics.py")
        sync_output_files(
            output_folder="mining/output",
            target_folder="web_app/data",
            files_to_copy=["clusters.csv", "term_frequencies.csv", "tfidf_matrix.csv", "wordcloud.png"]
        )

    if do_embeddings:
        print("🔍 Ejecutando generación de embeddings y FAISS...")
        run_script("embedding/faiss_search.py")
        sync_output_files(
            output_folder="embedding/output",
            target_folder="web_app/data",
            files_to_copy=[
                "embeddings.npy",
                "embedding_pmids.csv",
                "embedding_texts.csv",
                "semantic_search_results.csv"
            ]
        )

    if do_summary:
        print("✍️ Generando resúmenes automáticos...")
        run_script("scripts/generate_summary.py")

    print("\n✅ Sincronización completada.")
    print("🚀 Para lanzar la aplicación: streamlit run web_app/app.py")


if __name__ == "__main__":
    main()
