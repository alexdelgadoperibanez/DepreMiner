import re
import pandas as pd
from collections import Counter, defaultdict


def extract_contextual_chemical_outcomes(mongo_coll) -> pd.DataFrame:
    """
    Extrae relaciones Chemical–Outcome con validación cruzada contra entidades 'Chemical'.

    Returns:
        pd.DataFrame: ['Chemical', 'Outcome', 'Count']
    """
    PATTERNS = [
        r"treated with (?P<drug>\w+).*?\b(?P<outcome>remission|improvement|response|recovery|relapse)\b",
        r"\b(?P<drug>\w+) led to (?P<outcome>remission|improvement|response|recovery|relapse)",
        r"\b(?P<outcome>remission|improvement|response|recovery|relapse) observed after (?P<drug>\w+)",
        r"\b(?P<drug>\w+) resulted in (?P<outcome>remission|improvement|response|recovery|relapse)",
        r"\b(?P<outcome>remission|improvement|response|recovery|relapse) achieved with (?P<drug>\w+)"
    ]

    counter = Counter()

    for doc in mongo_coll.find({"abstract": {"$exists": True}}, {"abstract": 1, "entities": 1}):
        abstract = doc["abstract"].lower()
        # lista de entidades Chemical válidas en el abstract
        valid_chems = {e["word"].lower().strip() for e in doc.get("entities", []) if
                       e.get("entity_group") == "Chemical" and e.get("word")}

        for pattern in PATTERNS:
            for match in re.finditer(pattern, abstract):
                drug = match.group("drug").lower().strip()
                outcome = match.group("outcome").lower().strip()

                if drug in valid_chems:
                    counter[(drug, outcome)] += 1

    df = pd.DataFrame(
        [(chem, outc, count) for (chem, outc), count in counter.items()],
        columns=["Chemical", "Outcome", "Count"]
    ).sort_values("Count", ascending=False)

    return df



