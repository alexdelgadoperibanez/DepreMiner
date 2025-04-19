from spacy.lang.en import English
from spacy.tokens import Span
from spacy.util import filter_spans


# Registrar la extensión solo si no está ya registrada
if not Span.has_extension("custom_color"):
    Span.set_extension("custom_color", default=None)

def render_ner_html(text, entities):
    """
    Renderiza el texto con entidades resaltadas sin solapamientos,
    aplicando colores personalizados según la clase de entidad.
    """
    label_colors = {
        "Chemical": "#f94144",  # Rojo fuerte
        "Disease": "#f9c74f",  # Amarillo brillante
        "Gene": "#90be6d",  # Verde suave
        "Entity": "#577590",  # Azul acero
        "Protein": "#43aa8b",  # Verde esmeralda
    }

    nlp = English()
    doc = nlp(text)
    spans = []

    for ent in entities:
        label = ent.get("entity_group", "Entity")
        for pos in ent.get("positions", []):
            start = pos.get("start")
            end = pos.get("end")
            if start is not None and end is not None:
                span = doc.char_span(start, end, label=label)
                if span:
                    spans.append(span)

    spans = filter_spans(spans)
    sorted_spans = sorted(spans, key=lambda x: x.start_char)

    rendered = ""
    last_idx = 0
    for span in sorted_spans:
        rendered += text[last_idx:span.start_char]
        rendered += (
            f"<span style='background-color:{label_colors.get(span.label_, '#ccc')}; "
            f"padding:2px; border-radius:4px;'>"
            f"{text[span.start_char:span.end_char]} "
            f"<span style='font-size:0.8em; color:#333;'>[{span.label_}]</span>"
            f"</span>"
        )
        last_idx = span.end_char

    rendered += text[last_idx:]
    return f"<div style='line-height:1.6; font-family:sans-serif;'>{rendered}</div>"

