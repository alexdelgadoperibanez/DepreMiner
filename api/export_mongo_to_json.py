import os
import json
from pymongo import MongoClient
from bson import json_util
from configparser import ConfigParser

def get_mongo_config(config_path="../config/config.conf"):
    config = ConfigParser()
    config.read(config_path)
    uri = config.get("db", "uri")
    db_name = config.get("db", "db_name")
    return uri, db_name

def export_database_to_json(config_path="config.conf", output_dir="mongo_exports", batch_size=1000):
    uri, db_name = get_mongo_config(config_path)
    client = MongoClient(uri)
    db = client[db_name]
    os.makedirs(output_dir, exist_ok=True)

    collections = db.list_collection_names()
    print(f"Exportando {len(collections)} colecciones...")

    for collection_name in collections:
        collection = db[collection_name]
        cursor = collection.find()

        i = 0
        part = 1
        batch = []

        for doc in cursor:
            batch.append(json.loads(json_util.dumps(doc)))
            i += 1
            if i % batch_size == 0:
                output_path = os.path.join(output_dir, f"{collection_name}_part{part}.json")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(batch, f, indent=2, ensure_ascii=False)
                print(f"✔️ {output_path} ({len(batch)} documentos)")
                batch = []
                part += 1

        if batch:
            output_path = os.path.join(output_dir, f"{collection_name}_part{part}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(batch, f, indent=2, ensure_ascii=False)
            print(f"✔️ {output_path} ({len(batch)} documentos)")

    client.close()
    print("✅ Exportación completada.")

if __name__ == "__main__":
    export_database_to_json(config_path="../config/config.conf")
