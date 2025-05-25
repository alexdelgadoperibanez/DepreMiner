import os
import time
import configparser
from Bio import Entrez, Medline
from pymongo import MongoClient

def load_query_from_file(query_path: str) -> str:
    """
    Lee el contenido completo de un archivo de texto y lo retorna como string.

    Args:
        query_path (str): Ruta al archivo de texto que contiene la query.

    Returns:
        str: Contenido del archivo como una cadena de texto.
    """
    with open(query_path, "r", encoding="utf-8") as f:
        query_text = f.read().strip()
    return query_text

def get_all_ids(query: str, batch_size: int = 100) -> list[str]:
    """
    Recupera todos los IDs de artículos que coinciden con la query en PubMed.

    Args:
        query (str): Término de búsqueda en PubMed.
        batch_size (int): Número de IDs a recuperar por lote (paginación).

    Returns:
        list[str]: Lista de IDs únicos de artículos en PubMed.
    """
    handle = Entrez.esearch(db="pubmed", term=query, retmax=0)
    record = Entrez.read(handle)
    handle.close()

    total_count = int(record["Count"])
    print(f"Total de artículos para la query:\n{query}\n=> {total_count} artículos.")

    all_ids = []
    for start in range(0, total_count, batch_size):
        handle = Entrez.esearch(db="pubmed", term=query, retmax=batch_size, retstart=start)
        record = Entrez.read(handle)
        handle.close()

        id_list = record.get("IdList", [])
        all_ids.extend(id_list)

        time.sleep(0.3)

    unique_ids = list(set(all_ids))
    print(f"IDs recuperados (sin duplicados): {len(unique_ids)}")
    return unique_ids

def fetch_pubmed_abstracts(id_list: list[str]) -> list[dict]:
    """
    Descarga y retorna los abstracts de PubMed para una lista de IDs.

    Args:
        id_list (list): Lista de PMIDs a recuperar.

    Returns:
        list[dict]: Lista de diccionarios con 'pmid', 'title', 'abstract' y 'date'.
    """
    step = 50
    results = []
    for i in range(0, len(id_list), step):
        sub_ids = id_list[i:i+step]
        try:
            handle = Entrez.efetch(db="pubmed", id=sub_ids, rettype="medline", retmode="text")
            records = Medline.parse(handle)
            for record in records:
                pmid = record.get("PMID", "")
                title = record.get("TI", "")
                abstract = record.get("AB", "")
                publication_date = record.get("DP", "")
                if pmid:
                    results.append({"pmid": pmid, "title": title, "abstract": abstract, "date": publication_date})
            handle.close()
            time.sleep(0.3)
        except Exception as e:
            print("Error al recuperar abstracts:", e)
    return results

def store_abstracts_in_mongo(abstracts: list[dict], db_name: str, collection_name: str, uri: str) -> None:
    """
    Almacena una lista de abstracts en una colección de MongoDB.

    Args:
        abstracts (list): Lista de diccionarios con abstracts a almacenar.
        db_name (str): Nombre de la base de datos en MongoDB.
        collection_name (str): Nombre de la colección en MongoDB.
        uri (str): URI de conexión a MongoDB.

    Returns:
        None
    """
    client = MongoClient(uri)
    db = client[db_name]
    coll = db[collection_name]

    for doc in abstracts:
        pmid = doc["pmid"]
        coll.update_one({"pmid": pmid}, {"$set": doc}, upsert=True)

    print(f"Se han almacenado/actualizado {len(abstracts)} documentos en '{collection_name}'.")

if __name__ == "__main__":
    print("🚀 Ejecutando descarga de abstracts desde PubMed...")

    # Cargar configuración
    config = configparser.ConfigParser()
    config.read("config/config.conf")

    Entrez.email = config.get("pubmed", "email")
    query_path = config.get("pubmed", "query_path")
    db_uri = config.get("db", "uri")
    db_name = config.get("db", "database")
    collection_name = config.get("db", "collection")

    # Leer query y lanzar flujo completo
    query = load_query_from_file(query_path)
    ids = get_all_ids(query)
    abstracts = fetch_pubmed_abstracts(ids)
    store_abstracts_in_mongo(abstracts, db_name, collection_name, db_uri)

    print("✅ Descarga y almacenamiento completados.")

