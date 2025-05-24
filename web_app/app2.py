import streamlit as st
from pymongo import MongoClient
from utils.faiss_utils import load_index_and_docs, search_similar
from utils.bio_chat import generate_biomedical_answer
from utils.pattern_extractor import extract_contextual_chemical_outcomes
from utils.ner_utils import render_ner_html
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# Configuración inicial
st.set_page_config(page_title="TFM - Eficacia de Tratamientos", layout="wide")

# Cargar index FAISS y modelo
index, docs_texts, pmids = load_index_and_docs()

# Cargar conexión MongoDB
mongo_client = MongoClient("mongodb://localhost:27017")
mongo_coll = mongo_client["PubMedDB"]["major_depression_abstracts"]

# Función de búsqueda semántica
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# === Página 1: Exploración clínica ===
def page_exploracion():
    st.title("🔍 Exploración Clínica de Abstracts")
    query = st.text_input("Escribe tu consulta (en inglés):", placeholder="ej. efficacy of SSRIs in elderly patients")
    year_filter = st.slider("Filtrar por año:", 1990, 2025, (1990, 2025))

    if query:
        st.markdown("### Resultados más similares")
        results = search_similar(query, model, index, docs_texts, pmids)

        if not results:
            st.info("No se encontraron resultados para tu consulta.")
            return

        # Filtrar resultados por año antes de la paginación
        filtered_results = []
        for result in results:
            doc = mongo_coll.find_one({"pmid": str(result["pmid"])})
            if doc:
                date = doc.get("date", "")
                if date:
                    year = int(date.split()[0]) if date.split()[0].isdigit() else None
                    if year and (year_filter[0] <= year <= year_filter[1]):
                        result["year"] = year
                        result["relevance"] = determine_relevance(result["distance"])
                        filtered_results.append(result)

        if not filtered_results:
            st.info("No se encontraron resultados en el rango de años seleccionado.")
            return

        # Ordenar por año (opcional)
        filtered_results = sorted(filtered_results, key=lambda x: x.get("year", 0))

        # Paginación: mostrar 10 resultados por pestaña
        page_size = 10
        total = len(filtered_results)
        num_pages = (total + page_size - 1) // page_size
        page_tabs = st.tabs([f"Página {i + 1}" for i in range(num_pages)])

        for page_num, tab in enumerate(page_tabs):
            with tab:
                start = page_num * page_size
                end = min(start + page_size, total)
                st.markdown(f"Mostrando resultados {start + 1} – {end} de {total}")

                for i in range(start, end):
                    result = filtered_results[i]
                    doc = mongo_coll.find_one({"pmid": str(result["pmid"])})

                    # Mostrar directamente la relevancia en el título del artículo
                    with st.expander(f"📄 {result['title'][:100]}... - {result['relevance']}"):
                        st.markdown(f"**PMID:** `{result['pmid']}`  \n**Distancia FAISS:** `{result['distance']:.4f}`")

                        full_text = doc.get("abstract1", "") + doc.get("abstract2", "")
                        st.markdown("#### Título del abstract:")
                        st.write(doc.get("title", ""))

                        # Mostrar resumen si existe
                        summary = doc.get("summary")
                        if summary:
                            st.markdown("#### ✍️ Resumen automático del abstract")
                            st.write(summary)

                        st.markdown("#### 🧠 Entidades reconocidas:")
                        if "entities" in doc and len(doc["entities"]) > 0:
                            try:
                                html = render_ner_html(full_text, doc["entities"])
                                st.markdown(html, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"❌ Error al mostrar entidades: {e}")
                        else:
                            st.info("ℹ️ No se encontraron entidades NER para este documento.")

                    st.markdown("#### Acceso directo al artículo:")
                    st.markdown(f"🔗 [Ver en PubMed](https://pubmed.ncbi.nlm.nih.gov/{result['pmid']}/)")


def determine_relevance(distance: float) -> str:
    """
    Determina la relevancia de un resultado basado en su distancia.

    Args:
        distance (float): Distancia FAISS del resultado.

    Returns:
        str: Etiqueta de relevancia.
    """
    if distance <= 0.5:
        return "✅ Muy Relevante"
    elif distance <= 1.0:
        return "🟢 Relevante"
    elif distance <= 1.5:
        return "🟡 Poco Relevante"
    else:
        return "🔴 No Relevante"
# === Página 2: Análisis agregado ===
def page_analisis():
    st.title("📊 Análisis Agregado de Literatura")

    df = pd.DataFrame(list(mongo_coll.find({"entities.0": {"$exists": True}}, {"entities": 1, "date": 1})))
    records = []
    for row in df.itertuples():
        date = getattr(row, "date", "")
        year = int(date.split()[0]) if date and date.split()[0].isdigit() else None
        for ent in row.entities:
            label = ent.get("entity_group")
            word = ent.get("word")
            if label and word:
                records.append((label, word.lower().strip(), year))

    ent_df = pd.DataFrame(records, columns=["type", "word", "year"])

    tabs = st.tabs(["💊 Top Fármacos", "💥 Fármacos vs Resultados", "📈 Evolución Temporal"])

    with tabs[0]:
        st.markdown("### 💊 Top medicamentos mencionados")
        top_chems = ent_df[ent_df["type"] == "Chemical"]["word"].value_counts().head(20)
        st.bar_chart(top_chems)

    with tabs[1]:
        st.markdown("### 💥 Co-ocurrencias Chemical – Outcome")
        df_out = extract_contextual_chemical_outcomes(mongo_coll)
        st.dataframe(df_out.head(20))
        pivot_df = df_out.pivot_table(index="Chemical", columns="Outcome", values="Count", fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_df, cmap="Blues", ax=ax)
        st.pyplot(fig)

    with tabs[2]:
        st.markdown("### 📈 Menciones de tratamientos farmacológicos por Año")
        chem_df = ent_df[(ent_df["type"] == "Chemical") & (ent_df["year"].notna())]
        year_counts = chem_df.groupby("year").size()
        st.line_chart(year_counts)
        st.caption("Número de menciones de fármacos (entidades 'Chemical') por año en los abstracts.")

        st.markdown("### 🔍 Evolución temporal de uno o más fármacos")
        selected_drugs = st.multiselect("Selecciona uno o más fármacos", sorted(chem_df["word"].unique()))
        if selected_drugs:
            multi_df = chem_df[chem_df["word"].isin(selected_drugs)]
            multi_year_counts = multi_df.groupby(["word", "year"]).size().unstack(fill_value=0)
            st.line_chart(multi_year_counts.T)
            st.caption("Comparativa de menciones por año entre los fármacos seleccionados.")
# === Página 3: Caso clínico ===
def page_caso():
    st.title("📋 Caso de Uso Clínico: Exploración de un Fármaco")
    fármaco = st.text_input("Introduce el nombre del fármaco (minúsculas):")
    if fármaco:
        total = mongo_coll.count_documents(
            {"entities.entity_group": "Chemical", "entities.word": {"$regex": fármaco, "$options": "i"}})
        st.markdown(f"Este fármaco aparece en **{total} abstracts**.")

        OUTCOME_KEYWORDS = {"remission", "improvement", "response", "recovery", "relapse"}
        count = 0
        ejemplos = []
        for doc in mongo_coll.find(
                {"abstract": {"$exists": True}, "entities.word": {"$regex": fármaco, "$options": "i"}},
                {"abstract": 1, "pmid": 1}):
            abstract = doc.get("abstract", "")
            pmid = doc.get("pmid")
            if any(o in abstract.lower() for o in OUTCOME_KEYWORDS):
                count += 1
                link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
                ejemplos.append((abstract[:300] + "...", link))

        st.markdown(f"De ellos, **{count} abstracts** mencionan términos de eficacia clínica.")
        st.markdown("#### Ejemplos de contexto clínico:")
        for ex, url in ejemplos[:5]:
            st.info(ex)
            if url:
                st.markdown(f"🔗 [Ver en PubMed]({url})")


# === Página 4: Chat clínico ===
def page_chat():
    st.title("🤖 Asistente Clínico (Beta)")
    pregunta = st.text_input("Haz una pregunta médica:")
    if pregunta:
        results = search_similar(pregunta, model, index, docs_texts, pmids, k=3)
        contexto = "\n\n".join([
            mongo_coll.find_one({"pmid": str(r["pmid"])}).get("abstract", "") for r in results
        ])
        prompt = f"""Based on the following abstracts related to major depression treatments:\n\n{contexto}\n\nAnswer this: {pregunta}"""
        with st.spinner("Pensando como un experto clínico..."):
            respuesta = generate_biomedical_answer(prompt)
        st.success(respuesta)

# === Navegación ===
page = st.sidebar.radio("📌 Navegación", ["🔍 Exploración clínica", "📊 Análisis agregado", "📋 Caso clínico", "🤖 Chat clínico"])

if page == "🔍 Exploración clínica":
    page_exploracion()
elif page == "📊 Análisis agregado":
    page_analisis()
elif page == "📋 Caso clínico":
    page_caso()
elif page == "🤖 Chat clínico":
    page_chat()
