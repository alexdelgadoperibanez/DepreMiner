#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import configparser
import pprint

from Bio import Entrez, Medline
from pymongo import MongoClient

def load_query_from_file(query_path):
    """Lee el contenido completo de un archivo de texto y lo retorna como string."""
    with open(query_path, "r", encoding="utf-8") as f:
        query_text = f.read().strip()
    return query_text

def get_all_ids(query, batch_size=100):
    """
    Recupera TODOS los IDs que concuerdan con la 'query' en PubMed,
    iterando en páginas (retstart) hasta cubrir la cantidad total.
    """
    # Búsqueda inicial para saber cuántos hay en total
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

        time.sleep(0.3)  # para no saturar la API

    # Eliminar duplicados en caso de solapamientos
    unique_ids = list(set(all_ids))
    print(f"IDs recuperados (sin duplicados): {len(unique_ids)}")
    return unique_ids

def fetch_pubmed_abstracts(id_list):
    """
    Dada una lista de PMIDs, descarga los datos (title, abstract, fecha de publicación) en bloques.
    Retorna una lista de diccionarios con 'pmid', 'title', 'abstract' y 'date'.
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
                # Extraemos la fecha de publicación; el campo "DP" es el Date of Publication
                publication_date = record.get("DP", "")
                if pmid:
                    results.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "date": publication_date
                    })
            handle.close()
            time.sleep(0.3)
        except Exception as e:
            print("Error al recuperar abstracts:", e)
    return results

def store_abstracts_in_mongo(abstracts, db_name, collection_name, uri):
    """
    Almacena la lista de abstracts en MongoDB, usando update_one con upsert=True
    para no duplicar PMIDs.
    """
    client = MongoClient(uri)
    db = client[db_name]
    coll = db[collection_name]

    for doc in abstracts:
        pmid = doc["pmid"]
        coll.update_one({"pmid": pmid}, {"$set": doc}, upsert=True)

    print(f"Se han almacenado/actualizado {len(abstracts)} documentos en '{collection_name}'.")

if __name__ == "__main__":
    # Directorio base de este archivo
    base_dir = os.path.dirname(os.path.abspath(__file__))

    ####################################################################
    # 1) Cargamos la configuración desde config.ini
    ####################################################################
    config_path = os.path.join(base_dir, "..", "config", "config.conf")
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    pubmed_email = cfg["pubmed"].get("email", "TU_CORREO@ejemplo.com")
    batch_size = cfg["pubmed"].getint("batch_size", 100)

    mongo_uri = cfg["db"].get("uri", "mongodb://localhost:27017")
    db_name = cfg["db"].get("db_name", "PubMedDB")
    collection_name = cfg["db"].get("collection_name", "major_depression")

    # Establecemos Entrez.email
    Entrez.email = pubmed_email

    ####################################################################
    # 2) Cargamos la query (search_query.txt)
    ####################################################################
    query_path = os.path.join(base_dir, "..", "query", "search_query.txt")
    pubmed_query = load_query_from_file(query_path)

    ####################################################################
    # 3) Obtenemos todos los IDs (paginación)
    ####################################################################
    all_ids = get_all_ids(pubmed_query, batch_size=batch_size)

    ####################################################################
    # 4) Descargamos los abstracts
    ####################################################################
    data = fetch_pubmed_abstracts(all_ids)
    print(f"Se han descargado {len(data)} abstracts.")


    ####################################################################
    # 5) Guardamos en MongoDB
    ####################################################################
    store_abstracts_in_mongo(data, db_name, collection_name, mongo_uri)
