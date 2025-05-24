import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

from sentence_transformers import SentenceTransformer
from pymongo import MongoClient

from utils.ner_utils import render_ner_html
from utils.faiss_utils import load_index_and_docs, search_similar
from utils.bio_chat import generate_biomedical_answer
from utils.pattern_extractor import extract_contextual_chemical_outcomes



# Configuración de la app
st.set_page_config(page_title="PubMed TFM - Búsqueda Semántica", layout="wide")

# Cargar FAISS + textos reales
index, docs_texts, pmids = load_index_and_docs()
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Cargar MongoDB
mongo_client = MongoClient("mongodb://localhost:27017")
mongo_coll = mongo_client["PubMedDB"]["major_depression_abstracts"]

# ========================
# FUNCIONES
# ========================

def show_search():
    st.title("🔍 Búsqueda Semántica de Abstracts en PubMed")
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


def show_dashboard():
    st.title("📊 Dashboard de Análisis de Abstracts")

    # Datos base
    total_docs = mongo_coll.count_documents({"abstract": {"$exists": True}})
    total_entities = mongo_coll.count_documents({"entities.0": {"$exists": True}})
    st.markdown(f"- 🧾 Abstracts con texto: **{total_docs}**")
    st.markdown(f"- 🧠 Abstracts con entidades NER: **{total_entities}**")

    # Preparar datos base una sola vez
    all_entities = []
    entity_map = {}
    cursor = mongo_coll.find({"entities.0": {"$exists": True}}, {"entities": 1, "pmid": 1, "title": 1, "date": 1})
    for doc in cursor:
        pmid = str(doc.get("pmid"))
        chemicals = set()
        diseases = set()
        for ent in doc["entities"]:
            label = ent.get("entity_group")
            word_original = ent.get("word")
            if not word_original:
                continue
            word_original = word_original.strip()
            date = doc.get("date", "")
            all_entities.append((label, word_original, date))
            if label == "Chemical":
                chemicals.add(word_original)
            elif label == "Disease":
                diseases.add(word_original)
        entity_map[pmid] = {
            "chemical": chemicals,
            "disease": diseases,
            "title": doc.get("title", "Sin título"),
            "date": doc.get("date")
        }

    df = pd.DataFrame(all_entities, columns=["type", "word", "date"])

    tabs = st.tabs([
        "🧪 Top Entidades", "📈 Análisis Temporal",
        "🔎 Comparativa de Tratamientos", "📈 Abstracts por Año",
        "📝 Metodología de Estudios", "💊 Tratamientos Farmacológicos",
        "💥 Co-ocurrencias Chemical – Outcome"
    ])

    # === Top Entidades ===
    with tabs[0]:
        st.markdown("### 🧪 Top entidades por tipo")
        selected_type = st.selectbox("Selecciona tipo de entidad", df["type"].unique(), key="top_entities_select")
        top_entities = df[df["type"] == selected_type]["word"].value_counts().head(20)
        st.bar_chart(top_entities)

    # === Análisis Temporal ===
    with tabs[1]:
        st.markdown("### 📈 Análisis Temporal por Entidad")
        selected_type = st.selectbox("Selecciona tipo de entidad", df["type"].unique(), key="temporal_select")
        filtered_df = df[df["type"] == selected_type].copy()
        filtered_df["year"] = pd.to_numeric(
            filtered_df["date"].str.extract(r"(\d{4})")[0], errors="coerce"
        ).dropna().astype(int)
        if filtered_df["year"].empty:
            st.warning("No se encontraron fechas válidas para esta entidad.")
        else:
            year_counts = filtered_df["year"].value_counts().sort_index()
            fig, ax = plt.subplots()
            ax.plot(year_counts.index, year_counts.values, marker="o", linestyle="-")
            ax.set_xlabel("Año")
            ax.set_ylabel("Número de menciones")
            ax.set_title(f"Menciones de {selected_type} por Año")
            ax.grid(True)
            st.pyplot(fig)

    # === Comparativa entre entidades ===
    with tabs[2]:
        st.markdown("### 📊 Comparar Entidades por Año")
        selected_type = st.selectbox("Selecciona tipo de entidad para comparar", df["type"].unique(),
                                     key="compare_select")
        comparison_df = df[df["type"] == selected_type].copy()
        entities_to_compare = st.multiselect("Selecciona entidades para comparar",
                                             comparison_df["word"].unique(), key="compare_entities")
        if entities_to_compare:
            comp_df = comparison_df[comparison_df["word"].isin(entities_to_compare)]
            comp_df["year"] = pd.to_numeric(
                comp_df["date"].str.extract(r"(\d{4})")[0], errors="coerce"
            ).dropna().astype(int)
            comp_counts = comp_df.groupby(["word", "year"]).size().unstack(fill_value=0)
            st.line_chart(comp_counts.T)

        st.download_button("📥 Descargar Datos", data=comparison_df.to_csv(index=False),
                           file_name="comparison_analysis.csv", key="download_comparison_csv")

    # === Abstracts por Año ===
    with tabs[3]:
        st.markdown("### 📈 Número de Abstracts por Año")
        df["year"] = pd.to_numeric(df["date"].str.extract(r"(\d{4})")[0], errors="coerce").dropna().astype(int)
        doc_counts = df.drop_duplicates(subset=["word", "year"]).groupby("year").size()
        st.bar_chart(doc_counts)

    # === Metodología de Estudios ===
    with tabs[4]:
        st.markdown("### 📝 Análisis de Metodología de Estudios")
        method_keywords = ["randomized controlled trial", "meta-analysis", "double-blind", "placebo-controlled"]
        for keyword in method_keywords:
            count = mongo_coll.count_documents({"abstract": {"$regex": keyword, "$options": "i"}})
            st.markdown(f"- **{keyword.title()}:** {count} abstracts")

    # === Tratamientos Farmacológicos ===
    with tabs[5]:
        st.markdown("### 💊 Análisis de Tratamientos Farmacológicos")
        chemical_df = df[df["type"] == "Chemical"]
        top_chemicals = chemical_df["word"].value_counts().head(10)
        st.markdown("#### 📌 Top 10 Medicamentos más Mencionados")
        st.bar_chart(top_chemicals)

        st.markdown("#### 📊 Comparativa entre Medicamentos")
        meds_to_compare = st.multiselect("Selecciona medicamentos para comparar", top_chemicals.index)
        if meds_to_compare:
            compare_df = chemical_df[chemical_df["word"].isin(meds_to_compare)]
            compare_counts = compare_df["word"].value_counts()
            st.bar_chart(compare_counts)

    # === Co-ocurrencias Chemical – Outcome ===
    with tabs[6]:
        st.markdown("### 💥 Co-ocurrencias Chemical – Outcome")
        from collections import Counter

        # Palabras clave que indican resultados clínicos positivos
        OUTCOME_KEYWORDS = {"remission", "improvement", "response", "recovery", "relapse"}

        def detect_outcomes_in_text(text: str) -> set:
            return {kw for kw in OUTCOME_KEYWORDS if kw in text.lower()}

        cooc_counter = Counter()

        for doc in mongo_coll.find({"entities.0": {"$exists": True}}, {"abstract": 1, "entities": 1}):
            abstract = doc.get("abstract", "")
            if not abstract:
                continue

            # Detectar medicamentos
            chems = {
                e["word"].lower().strip()
                for e in doc["entities"]
                if e.get("entity_group") == "Chemical" and e.get("word")
            }

            # Detectar outcomes manualmente desde el texto
            outcomes = detect_outcomes_in_text(abstract)

            for chem in chems:
                for outc in outcomes:
                    cooc_counter[(chem, outc)] += 1

        # Crear DataFrame de co-ocurrencias
        cooc_df = pd.DataFrame(
            [(chem, outc, count) for (chem, outc), count in cooc_counter.items()],
            columns=["Chemical", "Outcome", "Count"]
        ).sort_values("Count", ascending=False)

        st.markdown("#### 🔝 Top 10 combinaciones más frecuentes")
        if not cooc_df.empty:
            st.dataframe(cooc_df.head(10))

            # Heatmap
            pivot_df = cooc_df.pivot_table(index="Chemical", columns="Outcome", values="Count", fill_value=0)
            if not pivot_df.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(pivot_df, cmap="Blues", ax=ax)
                st.pyplot(fig)
            else:
                st.warning("No hay suficientes co-ocurrencias para mostrar un heatmap.")
        else:
            st.warning("No se encontraron combinaciones de Chemical y Outcome en los abstracts.")



def show_chatbot():
    st.title("🤖 Biomedical Chatbot - Treatments for Major Depressive Disorder")

    query = st.text_input("Ask your question:", placeholder="e.g. What antidepressants are used in elderly patients?")
    if query:
        # Retrieve most relevant abstracts
        results = search_similar(query, model, index, docs_texts, pmids, k=3)
        context = "\n\n".join([
            mongo_coll.find_one({"pmid": str(r["pmid"])}).get("abstract", "") for r in results
        ])

        prompt = f"""The following excerpts have been extracted from scientific articles about pharmacological treatments for Major Depressive Disorder:

        {context}
        
        Researcher's question: {query}
    
        Answer based on the articles:"""

        with st.spinner("🧠 Generating response with BioGPT..."):
            answer = generate_biomedical_answer(prompt)

        st.markdown("### 💬 Chatbot response:")
        st.success(answer)

# ========================
# NAVEGACIÓN
# ========================

page = st.sidebar.radio("🔧 Navegación", ["🔍 Búsqueda", "📊 Dashboard", "🤖 Chatbot (beta version)"])

if page == "🔍 Búsqueda":
    show_search()
elif page == "📊 Dashboard":
    show_dashboard()
elif page == "🤖 Chatbot (beta version)":
    show_chatbot()
