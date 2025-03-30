#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import configparser
import logging
import numpy as np
from pymongo import MongoClient
from transformers import pipeline

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Umbral para filtrar entidades por score
SCORE_THRESHOLD = 0.5
# Límite de tokens por segmento
MAX_TOKENS = 512

def load_config():
    """
    Carga la configuración del archivo config/conf.conf.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "config", "config.conf")
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    return cfg

def load_ner_pipeline(model_name: str):
    """
    Carga el pipeline de NER utilizando el modelo finetuneado.
    Se utiliza aggregation_strategy="simple" para agrupar las entidades.
    """
    logging.info(f"Cargando el modelo NER: {model_name}")
    try:
        ner_pipe = pipeline("ner", model=model_name, tokenizer=model_name, aggregation_strategy="simple")
        logging.info("Modelo NER cargado correctamente.")
    except Exception as e:
        logging.error(f"Error al cargar el modelo {model_name}: {e}")
        raise e
    return ner_pipe

def convert_numpy_types(obj):
    """
    Convierte tipos numpy (por ejemplo, numpy.float32) a tipos nativos de Python.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def split_abstract(abstract: str, tokenizer, max_tokens=MAX_TOKENS):
    """
    Divide el abstract en dos partes si el número de tokens (sin contar tokens especiales)
    supera max_tokens. Se usa el tokenizer para contar tokens y se decodifican las dos mitades.
    Si el abstract es corto, se devuelve en abstract1 y abstract2 queda vacío.
    """
    encoding = tokenizer(abstract, add_special_tokens=False)
    input_ids = encoding["input_ids"]
    if len(input_ids) > max_tokens:
        half = len(input_ids) // 2
        tokens1 = input_ids[:half]
        tokens2 = input_ids[half:]
        abstract1 = tokenizer.decode(tokens1, skip_special_tokens=True)
        abstract2 = tokenizer.decode(tokens2, skip_special_tokens=True)
    else:
        abstract1 = abstract
        abstract2 = ""
    return abstract1, abstract2

def extract_entities_from_text(text: str, ner_pipe, model_name: str):
    """
    Extrae entidades de un fragmento de texto usando un pipeline NER.
    Se añade el nombre del modelo a cada entidad para luego identificar duplicados
    provenientes de distintos pipelines.
    """
    try:
        entities = ner_pipe(text)
    except Exception as e:
        logging.error(f"Error al extraer entidades: {e}")
        entities = []
    entities = convert_numpy_types(entities)
    # Añadimos el nombre del modelo a cada entidad extraída
    for ent in entities:
        ent["model"] = model_name
    return entities


def combine_and_filter_entities(entities_all, threshold=SCORE_THRESHOLD, tolerance=5):
    """
    Agrupa las entidades de la lista entities_all por (entity_group, word),
    pero también separa las ocurrencias si sus offsets difieren más que 'tolerance'.

    Para cada entidad se crea una lista de ocurrencias, donde cada ocurrencia:
      - Tiene un start y end representativo.
      - Se acumula el score (para calcular el promedio de score) y la cantidad de apariciones.
      - Se registra la lista de modelos que contribuyeron a esa ocurrencia.

    Devuelve una lista con una entrada por cada (entity_group, word) y dentro de cada
    entrada se incluye:
      - "occurrences": número de ocurrencias distintas encontradas.
      - "overall_combined_score": promedio de scores de cada ocurrencia.
      - "models": la unión de los modelos que detectaron alguna ocurrencia.
      - "positions": una lista de ocurrencias con su posición, score y modelos.
    """
    grouped = {}
    for ent in entities_all:
        score = ent.get("score", 0)
        if score < threshold:
            continue
        entity_group = ent.get("entity_group")
        word = ent.get("word")
        model = ent.get("model", "unknown")
        start = ent.get("start")
        end = ent.get("end")
        key = (entity_group, word)
        if key not in grouped:
            grouped[key] = []
        # Buscamos si ya existe una ocurrencia cercana (dentro de la tolerancia)
        found = False
        for occ in grouped[key]:
            if abs(occ["start"] - start) <= tolerance and abs(occ["end"] - end) <= tolerance:
                # Se considera la misma ocurrencia
                occ["score_sum"] += score
                occ["count"] += 1
                occ["combined_score"] = occ["score_sum"] / occ["count"]
                if model not in occ["models"]:
                    occ["models"].append(model)
                found = True
                break
        if not found:
            # Nueva ocurrencia
            grouped[key].append({
                "start": start,
                "end": end,
                "score_sum": score,
                "count": 1,
                "combined_score": score,
                "models": [model]
            })
    # Ahora combinamos la información para cada (entity_group, word)
    result = []
    for (entity_group, word), occ_list in grouped.items():
        # Calculamos un score global como el promedio de los scores de cada ocurrencia
        overall_score = sum(occ["combined_score"] for occ in occ_list) / len(occ_list)
        # Unificamos todos los modelos que han contribuido
        models = list({m for occ in occ_list for m in occ["models"]})
        result.append({
            "entity_group": entity_group,
            "word": word,
            "occurrences": len(occ_list),
            "overall_combined_score": overall_score,
            "models": models,
            "positions": occ_list
        })
    return result


def update_documents_with_entities(uri: str, db_name: str, collection_name: str, ner_model_infos, tokenizer):
    """
    Recorre los documentos en MongoDB que tienen un 'abstract' y que aún no tienen el campo 'entities'.
    Si el abstract es largo, se divide en dos partes (abstract1 y abstract2).
    Se extraen entidades con cada modelo y se combinan los resultados evitando contar la misma
    entidad repetidamente si la detectan distintos modelos.
    """
    client = MongoClient(uri)
    db = client[db_name]
    coll = db[collection_name]

    cursor = coll.find({"abstract": {"$exists": True, "$ne": ""}, "entities": {"$exists": False}})
    docs = list(cursor)
    total_docs = len(docs)
    logging.info(f"Se encontraron {total_docs} documentos para extracción de entidades.")

    processed_count = 0
    for doc in docs:
        abstract = doc["abstract"]
        abstract1, abstract2 = split_abstract(abstract, tokenizer, max_tokens=MAX_TOKENS)
        update_fields = {"abstract1": abstract1, "abstract2": abstract2}

        entities_total = []
        for model_name, ner_pipe in ner_model_infos:
            # Procesamos la primera parte
            ents1 = extract_entities_from_text(abstract1, ner_pipe, model_name)
            # Si hay segunda parte, extraer y ajustar offsets (sumar la longitud de abstract1 en caracteres)
            if abstract2:
                ents2 = extract_entities_from_text(abstract2, ner_pipe, model_name)
                offset = len(abstract1)
                for ent in ents2:
                    ent["start"] += offset
                    ent["end"] += offset
                combined = ents1 + ents2
            else:
                combined = ents1
            entities_total.extend(combined)

        # Combinar y filtrar sin duplicar ocurrencias de distintos modelos
        entities_filtered = combine_and_filter_entities(entities_total, threshold=SCORE_THRESHOLD)
        update_fields["entities"] = entities_filtered

        coll.update_one({"_id": doc["_id"]}, {"$set": update_fields})
        processed_count += 1
        logging.info(f"Documento PMID: {doc.get('pmid', 'N/A')} procesado, entidades totales: {len(entities_filtered)}")
        time.sleep(0.1)

    logging.info(f"Extracción de entidades completada para {processed_count} documentos.")

def main():
    cfg = load_config()
    mongo_uri = cfg["db"].get("uri", "mongodb://localhost:27017")
    db_name = cfg["db"].get("db_name", "PubMedDB")
    collection_name = cfg["db"].get("collection_name", "major_depression_abstracts")

    # Definimos los modelos a usar
    model_names = [
        "judithrosell/JNLPBA_PubMedBERT_NER",
        "judithrosell/BC5CDR_PubMedBERT_NER",
        "judithrosell/BioNLP13CG_PubMedBERT_NER"
    ]
    # Cargamos los pipelines y asociamos cada uno con su nombre
    ner_model_infos = [(name, load_ner_pipeline(name)) for name in model_names]
    # Usamos el tokenizer del primer pipeline (se asume compatibilidad)
    tokenizer = ner_model_infos[0][1].tokenizer

    update_documents_with_entities(mongo_uri, db_name, collection_name, ner_model_infos, tokenizer)

if __name__ == "__main__":
    main()
