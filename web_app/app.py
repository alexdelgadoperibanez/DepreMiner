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


# Configuración de la app
st.set_page_config(page_title="PubMed TFM - Búsqueda Semántica", layout="wide")

# Cargar FAISS + textos reales
index, docs_texts, pmids = load_index_and_docs()
model = SentenceTransformer("all-MiniLM-L6-v2")

# Cargar Mongo
mongo_client = MongoClient("mongodb://localhost:27017")
mongo_coll = mongo_client["PubMedDB"]["major_depression_abstracts"]

# ========================
# FUNCIONES
# ========================

def show_search():
    st.title("🔍 Búsqueda Semántica de Abstracts en PubMed")
    query = st.text_input("Escribe tu consulta (en inglés):", placeholder="ej. efficacy of SSRIs in elderly patients")

    if query:
        st.markdown("### Resultados más similares")
        results = search_similar(query, model, index, docs_texts, pmids)

        if not results:
            st.info("No se encontraron resultados para tu consulta.")
            return

        # Paginación: mostrar 10 resultados por pestaña
        page_size = 10
        total = len(results)
        num_pages = (total + page_size - 1) // page_size
        page_tabs = st.tabs([f"Página {i+1}" for i in range(num_pages)])

        for page_num, tab in enumerate(page_tabs):
            with tab:
                start = page_num * page_size
                end = min(start + page_size, total)
                st.markdown(f"Mostrando resultados {start+1} – {end} de {total}")

                for i in range(start, end):
                    result = results[i]
                    with st.expander(f"📄 {result['title'][:100]}..."):
                        st.markdown(f"**PMID:** `{result['pmid']}`  \n**Distancia FAISS:** `{result['distance']:.4f}`")

                        doc = mongo_coll.find_one({"pmid": str(result["pmid"])})
                        if doc:
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
                        else:
                            st.warning("⚠️ No se encontró el documento en Mongo para este PMID.")

                        st.markdown("\n")
                        st.markdown("#### Acceso directo al artículo:")
                        st.markdown(f"🔗 [Ver en PubMed](https://pubmed.ncbi.nlm.nih.gov/{result['pmid']}/)")


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
            label = ent["entity_group"]
            word = ent.get("word")
            if not word:
                continue
            word = word.strip().lower()
            all_entities.append((label, word))
            if label == "Chemical":
                chemicals.add(word)
            elif label == "Disease":
                diseases.add(word)
        entity_map[pmid] = {
            "chemical": chemicals,
            "disease": diseases,
            "title": doc.get("title", "Sin título"),
            "date": doc.get("date")
        }

    df = pd.DataFrame(all_entities, columns=["type", "word"])

    # Pestañas
    tabs = st.tabs([
        "🧪 Top Entidades",
        "☁️ WordCloud",
        "🔥 Heatmap",
        "📈 Abstracts por Año",
        "🔎 Buscar Pares"
    ])

    # === Top Entidades ===
    with tabs[0]:
        st.markdown("### 🧪 Top entidades por tipo")
        top_by_type = df.groupby("type")["word"].value_counts().groupby(level=0).head(10).reset_index(name="count")
        for ent_type in top_by_type["type"].unique():
            subset = top_by_type[top_by_type["type"] == ent_type]
            st.markdown(f"**🔹 {ent_type}**")
            st.dataframe(subset[["word", "count"]].reset_index(drop=True), use_container_width=True)

    # === WordCloud ===
    with tabs[1]:
        st.markdown("### ☁️ WordCloud por tipo de entidad")
        selected_type = st.selectbox("Selecciona tipo de entidad", df["type"].unique())
        words = df[df["type"] == selected_type]["word"].value_counts().to_dict()
        wc = WordCloud(width=800, height=400, background_color="black", colormap="Set2").generate_from_frequencies(
            words)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # === Heatmap ===
    with tabs[2]:
        st.markdown("### 🔥 Heatmap de Co-ocurrencias: Chemical – Disease")
        from itertools import product
        pair_counter = Counter()
        for entry in entity_map.values():
            for chem, dis in product(entry["chemical"], entry["disease"]):
                pair_counter[(chem, dis)] += 1
        top_pairs = Counter(pair_counter).most_common(20)
        chem_set = sorted({pair[0] for pair, _ in top_pairs if pair[0]})
        dis_set = sorted({pair[1] for pair, _ in top_pairs if pair[1]})
        heat_data = pd.DataFrame(index=chem_set, columns=dis_set).fillna(0)
        for (chem, dis), count in top_pairs:
            if chem in heat_data.index and dis in heat_data.columns:
                heat_data.loc[chem, dis] = count
        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(heat_data.fillna(0).astype(int), annot=True, fmt="d", cmap="rocket_r", linewidths=0.5,
                    linecolor='gray', ax=ax)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.title("Top 20 Co-ocurrencias entre Chemical y Disease", fontsize=14, weight='bold')
        st.pyplot(fig)

    # === Abstracts por año ===
    with tabs[3]:
        st.markdown("### 📈 Abstracts por Año")
        years = []
        for info in entity_map.values():
            raw_date = info.get("date", "")
            if isinstance(raw_date, str):
                parts = raw_date.strip().split()
                if parts and parts[0].isdigit():
                    years.append(int(parts[0]))
        if years:
            year_df = pd.DataFrame(years, columns=["year"])
            year_counts = year_df["year"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(year_counts.index, year_counts.values, marker="o", linestyle="-")
            ax.set_xlabel("Año")
            ax.set_ylabel("Número de Abstracts")
            ax.set_title("Publicaciones por Año")
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.info("No se encontraron fechas válidas.")

    # === Buscar Pares ===
    with tabs[4]:
        st.markdown("### 🔎 Buscar abstracts con un par Chemical–Disease específico")
        input_chem = st.text_input("Introduce un Chemical", placeholder="ej. fluoxetine").lower().strip()
        input_dis = st.text_input("Introduce un Disease", placeholder="ej. depression").lower().strip()
        if input_chem and input_dis:
            matched_pmids = [pmid for pmid, ent in entity_map.items()
                             if input_chem in ent["chemical"] and input_dis in ent["disease"]]
            st.markdown(f"Se encontraron **{len(matched_pmids)}** abstracts con ambos términos:")
            for pmid in matched_pmids[:10]:
                st.markdown(f"🔗 [PMID {pmid} - {entity_map[pmid]['title']}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")

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

page = st.sidebar.radio("🔧 Navegación", ["🔍 Búsqueda", "📊 Dashboard", "🤖 Chatbot"])

if page == "🔍 Búsqueda":
    show_search()
elif page == "📊 Dashboard":
    show_dashboard()
elif page == "🤖 Chatbot":
    show_chatbot()
