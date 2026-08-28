from datetime import datetime, timedelta, timezone
import feedparser
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import streamlit as st

# 1. Configurazione Interfaccia Giornale
st.set_page_config(
    page_title="Il Clarino - Ground News Edition",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📰 Il Clarino | Ground News Daily")
st.caption("Prima pagina automatica con analisi del bias e copertura mediatica delle ultime 24 ore.")

# Sidebar per configurazione
st.sidebar.header("Parametri")
api_key = st.sidebar.text_input("Gemini API Key", type="password")
categoria = st.sidebar.selectbox(
    "Sezione Giornale",
    ["Prima Pagina (Top News)", "Politica", "Economia", "Tecnologia"],
)
esegui = st.sidebar.button("Genera Edizione del Giorno", type="primary")

# Mappa Bias Testate Italiane (Estendibile)
BIAS_DB = {
    "ANSA.it": "Centro",
    "La Repubblica": "Centro-Sinistra",
    "Il Giornale": "Centro-Destra",
    "Corriere della Sera": "Centro",
    "Il Fatto Quotidiano": "Sinistra",
    "La Stampa": "Centro-Sinistra",
    "Libero Quotidiano": "Destra",
    "Il Sole 24 Ore": "Centro",
    "Domani": "Sinistra",
    "Agi": "Centro",
}

# 2. Funzione Ingestione Notizie 24h
def fetch_top_news_24h(cat: str):
    urls = {
        "Prima Pagina (Top News)": "https://news.google.com/rss?hl=it&gl=IT&ceid=IT:it",
        "Politica": "https://news.google.com/rss/headlines/section/topic/POLITICS?hl=it&gl=IT&ceid=IT:it",
        "Economia": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=it&gl=IT&ceid=IT:it",
        "Tecnologia": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=it&gl=IT&ceid=IT:it",
    }
    feed = feedparser.parse(urls[cat])

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)

    articles = []
    for entry in feed.entries:
        # Controllo data di pubblicazione (ultime 24 ore)
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            if pub_date < cutoff:
                continue

        source = entry.source.title if hasattr(entry, "source") else "Sconosciuto"
        articles.append(
            {"title": entry.title, "link": entry.link, "source": source}
        )

    return articles[:30]  # Limita ai primi 30 articoli per velocità


# 3. Logica Principale
if esegui:
    if not api_key:
        st.error("Inserisci una chiave API Gemini nella barra laterale.")
        st.stop()

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    with st.spinner("Raccolta notizie ed elaborazione del bias in corso..."):
        articles = fetch_top_news_24h(categoria)

        if not articles:
            st.warning("Nessuna notizia recente trovata nelle ultime 24h.")
            st.stop()

        # Clustering vettoriale dei titoli
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        titles = [a["title"] for a in articles]
        embeddings = embedder.encode(titles)

        dbscan = DBSCAN(eps=0.38, min_samples=2, metric="cosine")
        labels = dbscan.fit_predict(embeddings)

        clusters = {}
        for idx, label in enumerate(labels):
            if label != -1:
                clusters.setdefault(label, []).append(articles[idx])

    st.divider()

    # 4. Rendering in Stile Giornale
    if not clusters:
        st.info(
            "Non sono stati individuati raggruppamenti significativi nelle ultime notizie."
        )

    for c_id, group in clusters.items():
        sources_list = []
        bias_counts = {
            "Sinistra": 0,
            "Centro-Sinistra": 0,
            "Centro": 0,
            "Centro-Destra": 0,
            "Destra": 0,
            "Non classificato": 0,
        }

        for item in group:
            b = BIAS_DB.get(item["source"], "Non classificato")
            bias_counts[b] += 1
            sources_list.append(
                f"- **{item['source']}** ({b}): [{item['title']}]({item['link']})"
            )

        # Prompt per la sintesi e l'analisi del framing
        prompt = f"""
        Sei un redattore capo di una testata super-partes stile Ground News.
        Analizza questi articoli relativi allo stesso evento:

        {chr(10).join(sources_list)}

        Fornisci un output in formato Markdown strutturato così:
        **Titolo Notizia**: (Crea un titolo unico e neutrale)
        **Sintesi Fattuale**: (Spiega i fatti principali in 2-3 frasi)
        **Framing & Tono**: (Come varia la narrazione tra i giornali di diversa fazione)
        **Blindspot**: (Indica se la notizia è ignorata o sovra-rappresentata da un orientamento specifico, oppure "Nessuno")
        """

        res = model.generate_content(prompt)

        # Card Stile Giornale
        with st.container():
            st.markdown(f"### 📰 Notizia #{c_id + 1}")

            col_main, col_stats = st.columns([2, 1])

            with col_main:
                st.markdown(res.text)

            with col_stats:
                st.markdown("**Copertura Fonti**")
                st.metric("Fonti Totali", len(group))

                # Grafico rapido distribuzione bias
                st.write("**Distribuzione Bias:**")
                for b_type, count in bias_counts.items():
                    if count > 0:
                        st.caption(f"{b_type}: {count}")
                        st.progress(count / len(group))

            with st.expander("Vedi articoli correlati e fonti"):
                st.markdown("\n".join(sources_list))

            st.divider()
