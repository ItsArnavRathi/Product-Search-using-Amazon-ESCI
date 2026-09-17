"""
Stage 5 — Search interface (Streamlit web app).

Run it with:
    streamlit run src/app.py

Type a shopping query (or pick an example), choose a ranking method, and see the
top products. If your query matches one of the dataset's queries, each result
also shows its human ESCI label (E/S/C/I) so you can eyeball the quality.
"""

import os
import pandas as pd
import streamlit as st

from ranking import TfidfRanker, BM25Ranker, TransformerRanker, RRFRanker
from search import search_catalog

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

LABEL_TEXT = {"E": "Exact", "S": "Substitute", "C": "Complement", "I": "Irrelevant"}
LABEL_COLOR = {"E": "#54A24B", "S": "#4C78A8", "C": "#F58518", "I": "#B0B0B0"}


# ----------------------------------------------------------------------
# Load data + build the rankers ONCE (cached across reruns)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data and building search index...")
def load_everything():
    prod = pd.read_csv(os.path.join(DATA, "products.csv")).fillna("")
    judg = pd.read_csv(os.path.join(DATA, "judgments.csv"))
    pids = prod["product_id"].tolist()
    titles = prod["product_title"].tolist()

    tfidf = TfidfRanker(pids, titles)
    bm25 = BM25Ranker(pids, titles)
    st_ranker = TransformerRanker(pids, titles)
    rrf = RRFRanker([bm25, st_ranker])
    rankers = {r.name: r for r in [tfidf, bm25, st_ranker, rrf]}

    title_of = dict(zip(prod["product_id"], prod["product_title"]))
    # ground-truth labels keyed by (query, product_id) for the demo
    labels_by_query = {}
    for q, g in judg.groupby("query"):
        labels_by_query[q] = dict(zip(g["product_id"], g["esci_label"]))
    example_queries = sorted(labels_by_query.keys())
    return rankers, pids, title_of, labels_by_query, example_queries


def main():
    st.set_page_config(page_title="Amazon ESCI Product Search", page_icon="🔎",
                       layout="centered")
    st.title("🔎 Amazon ESCI Product Search")
    st.caption("Compare four IR ranking methods on real shopping queries.")

    if not os.path.exists(os.path.join(DATA, "products.csv")):
        st.error("No data found. Run `python src/download_data.py` first.")
        return

    rankers, pids, title_of, labels_by_query, example_queries = load_everything()

    # --- controls ---
    col1, col2 = st.columns([2, 1])
    with col1:
        method = st.selectbox("Ranking method", list(rankers.keys()), index=3)
    with col2:
        top_n = st.slider("Results", 5, 20, 10)

    pick = st.selectbox("Pick an example query (or type your own below)",
                        ["— type my own —"] + example_queries)
    typed = st.text_input("Search query",
                          value="" if pick == "— type my own —" else pick,
                          placeholder="e.g. wireless bluetooth headphones")
    query = typed.strip()

    if not query:
        st.info("Enter a query or pick an example to search.")
        return

    labels = labels_by_query.get(query)   # None for free-typed queries
    results = search_catalog(rankers[method], query, pids, title_of,
                             top_n=top_n, labels=labels)

    st.subheader(f"Top {len(results)} results — {method}")
    if labels is None:
        st.caption("(No ground-truth labels: this query isn't in the dataset.)")

    for r in results:
        badge = ""
        if r["label"]:
            c = LABEL_COLOR.get(r["label"], "#888")
            badge = (f"<span style='background:{c};color:white;padding:2px 8px;"
                     f"border-radius:10px;font-size:12px'>"
                     f"{LABEL_TEXT.get(r['label'], r['label'])}</span>")
        st.markdown(
            f"**{r['rank']}. {r['title']}** &nbsp; {badge}<br>"
            f"<span style='color:gray;font-size:13px'>score: {r['score']}</span>",
            unsafe_allow_html=True)


if __name__ == "__main__":
    main()
