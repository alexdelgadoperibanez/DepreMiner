import json
from glob import glob
from typing import List, Dict, Optional, Any
import re


class LocalMongoCollection:
    def __init__(self, data: List[Dict]):
        self.docs = data

    def find(self, query: Optional[Dict] = None, projection: Optional[Dict] = None):
        if not query:
            return self.docs
        return [doc for doc in self.docs if self._match_query(doc, query)]

    def find_one(self, query: Dict):
        for doc in self.find(query):
            return doc
        return None

    def count_documents(self, query: Dict):
        return len(self.find(query))

    def _get_nested_value(self, doc: Dict, path: str) -> Any:
        """Accede a claves anidadas como 'entities.0.word'."""
        keys = path.split(".")
        current = doc
        try:
            for key in keys:
                if isinstance(current, list) and key.isdigit():
                    current = current[int(key)]
                elif isinstance(current, dict):
                    current = current.get(key)
                else:
                    return None
            return current
        except (IndexError, KeyError, TypeError):
            return None

    def _match_query(self, doc: Dict, query: Dict) -> bool:
        for key, condition in query.items():
            value = self._get_nested_value(doc, key)

            if isinstance(condition, dict):
                # $exists
                if "$exists" in condition:
                    exists = value is not None
                    if exists != condition["$exists"]:
                        return False

                # $regex
                if "$regex" in condition:
                    pattern = re.compile(condition["$regex"], re.IGNORECASE if condition.get("$options") == "i" else 0)
                    if not isinstance(value, str) or not pattern.search(value):
                        return False

            else:
                # Comparación directa
                if value != condition:
                    return False

        return True


def load_local_collection(json_folder: str = "api/mongo_exports") -> LocalMongoCollection:
    all_docs = []
    for path in sorted(glob(f"{json_folder}/major_depression_abstracts_part*.json")):
        with open(path, "r", encoding="utf-8") as f:
            all_docs.extend(json.load(f))
    return LocalMongoCollection(all_docs)
