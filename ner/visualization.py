from spacy.tokens import Doc, Span
from spacy.lang.en import English
from spacy import displacy

def visualize_ner(text, entities):
    """
    Muestra el texto con las entidades resaltadas usando displacy.
    `entities` debe tener start, end, label.
    """
    nlp = English()
    doc = nlp(text)

    spans = []
    for ent in entities:
        start = ent["start"]
        end = ent["end"]
        label = ent["entity_group"]
        spans.append(Span(doc, doc.char_span(start, end).start, doc.char_span(start, end).end, label=label))

    doc.ents = spans
    displacy.render(doc, style="ent", jupyter=True)
