# Amazon ESCI Product Search (Information Retrieval Project)

A small, clean **e-commerce product search engine** built on the
[Amazon ESCI dataset](https://github.com/amazon-science/esci-data).
Given a shopping query, it ranks products by relevance using four methods and
compares them with standard IR metrics.

## What it does
- **Preprocessing:** clean product titles (lowercase, remove punctuation, tokenize, stem)
- **Indexing:** build a searchable product catalog
- **Four ranking methods:**
  - **TF-IDF** — keyword matching (baseline)
  - **BM25** — improved keyword ranking (implemented from scratch)
  - **Sentence Transformer** — semantic / meaning-based ranking
  - **Hybrid (RRF)** — Reciprocal Rank Fusion of BM25 + Transformer
- **Evaluation:** Precision@10, Recall, MAP, nDCG → comparison table + precision-recall graph
- **Search interface:** a simple Streamlit web app

## The ESCI labels (our "answer key")
Every query–product pair was judged by humans as one of:
`E` Exact · `S` Substitute · `C` Complement · `I` Irrelevant.

## Setup
```bash
python -m venv venv && source venv/bin/activate   # (optional)
pip install -r requirements.txt
```

## How to run (in order)
```bash
python src/download_data.py      # Stage 1: download data + build small sample
python src/preprocess.py         # Stage 2: clean text (built next)
python src/evaluate.py           # Stage 4: run all methods + save table & graph
streamlit run src/app.py         # Stage 5: launch the search app
```

## Project structure
```
esci-product-search/
├── data/            # products.csv, judgments.csv (sample); raw/ is git-ignored
├── src/             # all the code
├── results/         # comparison table + precision-recall graph
├── requirements.txt
└── README.md
```

## Team
4th-year B.Tech CSE — Information Retrieval project.
