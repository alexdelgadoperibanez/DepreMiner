from spacy.tokens import Doc, Span
from spacy.lang.en import English
from spacy import displacy
from typing import List, Dict, Any


def visualize_ner(text: str, entities: List[Dict[str, Any]]) -> None:
    """
    Muestra el texto con las entidades resaltadas usando displacy.

    Args:
        text (str): El texto completo en el que se resaltarán las entidades.
        entities (List[Dict[str, Any]]): Lista de entidades, cada una con:
            - start (int): Posición inicial de la entidad.
            - end (int): Posición final de la entidad.
            - entity_group (str): Etiqueta de la entidad.

    Returns:
        None: Muestra visualmente las entidades en Jupyter Notebook.
    """
    nlp = English()
    doc = nlp(text)
    spans = []

    for ent in entities:
        start = ent.get("start")
        end = ent.get("end")
        label = ent.get("entity_group", "ENTITY")

        char_span = doc.char_span(start, end)
        if char_span is not None:
            spans.append(Span(doc, char_span.start, char_span.end, label=label))
        else:
            print(f"[WARN] No se pudo crear span para entidad: {ent}")

    doc.ents = spans
    displacy.render(doc, style="ent", jupyter=True)
