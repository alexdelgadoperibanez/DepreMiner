#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import configparser
import logging
import re
import numpy as np
from pymongo import MongoClient
from transformers import pipeline
from typing import List, Dict, Tuple, Optional, Any

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

SCORE_THRESHOLD = 0.61
MAX_TOKENS = 512


def load_config() -> configparser.ConfigParser:
    """
    Carga la configuración del archivo config/conf.conf.

    Returns:
        configparser.ConfigParser: Objeto de configuración cargado.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "config", "config.conf")
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    return cfg


def load_ner_pipeline(model_name: str) -> Any:
    """
    Carga el pipeline de NER utilizando el modelo especificado.

    Args:
        model_name (str): Nombre del modelo NER.

    Returns:
        Any: Objeto de pipeline NER cargado.
    """
    logging.info(f"Cargando el modelo NER: {model_name}")
    try:
        ner_pipe = pipeline("ner", model=model_name, tokenizer=model_name, aggregation_strategy="simple")
        logging.info("Modelo NER cargado correctamente.")
    except Exception as e:
        logging.error(f"Error al cargar el modelo {model_name}: {e}")
        raise e
    return ner_pipe


def convert_numpy_types(obj: Any) -> Any:
    """
    Convierte tipos numpy a tipos nativos de Python.

    Args:
        obj (Any): Objeto que puede contener tipos numpy.

    Returns:
        Any: Objeto convertido a tipos nativos de Python.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def normalize_entity(word: str) -> Optional[str]:
    """
    Normaliza un término de entidad:
    - A minúsculas
    - Quita símbolos no alfabéticos
    - Elimina palabras con longitud <= 2

    Args:
        word (str): Término a normalizar.

    Returns:
        Optional[str]: Término normalizado o None si queda vacío.
    """
    word = re.sub(r"[^a-z\s]", "", word.lower().strip())
    tokens = [t for t in word.split() if len(t) > 2]
    return " ".join(tokens) if tokens else None


def split_abstract(abstract: str, tokenizer: Any, max_tokens: int = MAX_TOKENS) -> Tuple[str, str]:
    """
    Divide el abstract en dos partes si supera el límite de tokens.

    Args:
        abstract (str): Texto del abstract.
        tokenizer (Any): Tokenizador del modelo NER.
        max_tokens (int): Máximo de tokens permitidos.

    Returns:
        Tuple[str, str]: Abstract dividido en dos partes.
    """
    encoding = tokenizer(abstract, add_special_tokens=False)
    if len(encoding["input_ids"]) > max_tokens:
        half = len(encoding["input_ids"]) // 2
        return (tokenizer.decode(encoding["input_ids"][:half], skip_special_tokens=True),
                tokenizer.decode(encoding["input_ids"][half:], skip_special_tokens=True))
    return abstract, ""


def extract_entities_from_text(text: str, ner_pipe: Any, model_name: str) -> List[Dict[str, Any]]:
    """
    Extrae entidades de texto usando un pipeline NER.

    Args:
        text (str): Texto de entrada.
        ner_pipe (Any): Pipeline NER.
        model_name (str): Nombre del modelo.

    Returns:
        List[Dict[str, Any]]: Lista de entidades extraídas.
    """
    try:
        entities = ner_pipe(text)
    except Exception as e:
        logging.error(f"Error al extraer entidades: {e}")
        entities = []

    entities = convert_numpy_types(entities)
    for ent in entities:
        ent["model"] = model_name
    return entities


def combine_and_filter_entities(
        entities_all: List[Dict[str, Any]], threshold: float = SCORE_THRESHOLD, tolerance: int = 5
) -> List[Dict[str, Any]]:
    """
    Combina y filtra entidades extraídas de texto.

    Args:
        entities_all (List[Dict[str, Any]]): Lista de entidades sin filtrar.
        threshold (float): Umbral mínimo de score para considerar una entidad.
        tolerance (int): Tolerancia para combinar entidades similares.

    Returns:
        List[Dict[str, Any]]: Lista de entidades combinadas y filtradas.
    """
    from collections import defaultdict

    grouped = defaultdict(list)
    for ent in entities_all:
        if ent.get("score", 0) >= threshold:
            key = (ent.get("entity_group"), ent.get("word").lower())
            grouped[key].append(ent)

    result = []
    for (entity_group, word), occurrences in grouped.items():
        combined_score = sum(ent["score"] for ent in occurrences) / len(occurrences)
        result.append({
            "entity_group": entity_group,
            "word": normalize_entity(word),
            "occurrences": len(occurrences),
            "average_score": combined_score,
            "models": list(set(ent.get("model") for ent in occurrences))
        })

    return result


def update_documents_with_entities(
        uri: str, db_name: str, collection_name: str, ner_model_infos: List[Tuple[str, Any]], tokenizer: Any
) -> None:
    """
    Actualiza documentos en MongoDB con entidades extraídas.

    Args:
        uri (str): URI de conexión a MongoDB.
        db_name (str): Nombre de la base de datos.
        collection_name (str): Nombre de la colección.
        ner_model_infos (List[Tuple[str, Any]]): Lista de modelos NER.
        tokenizer (Any): Tokenizador para dividir abstracts.
    """
    client = MongoClient(uri)
    db = client[db_name]
    coll = db[collection_name]
    cursor = coll.find({"abstract": {"$exists": True}, "entities": {"$exists": False}})

    for doc in cursor:
        abstract1, abstract2 = split_abstract(doc.get("abstract", ""), tokenizer)
        entities_total = []

        for model_name, ner_pipe in ner_model_infos:
            entities_total += extract_entities_from_text(abstract1, ner_pipe, model_name)
            if abstract2:
                entities_total += extract_entities_from_text(abstract2, ner_pipe, model_name)

        entities_filtered = combine_and_filter_entities(entities_total)
        coll.update_one({"_id": doc["_id"]}, {"$set": {"entities": entities_filtered}})

if __name__ == "__main__":
    print("🧠 Iniciando extracción de entidades NER...")

    # 1. Cargar configuración
    cfg = load_config()
    mongo_uri = cfg.get("db", "uri")
    db_name = cfg.get("db", "database")
    collection = cfg.get("db", "collection")

    model_list = [m.strip() for m in cfg.get("ner", "models").split(",")]
    threshold = float(cfg.get("ner", "score_threshold", fallback="0.61"))

    # 2. Cargar modelos
    ner_models = []
    for model_name in model_list:
        pipe = load_ner_pipeline(model_name)
        tokenizer = pipe.tokenizer  # mismo para todos si son i
