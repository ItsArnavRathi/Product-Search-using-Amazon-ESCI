"""
Stage 3 — Ranking methods.

Four ways to rank products for a query. Each ranker is built once over the whole
product catalog, then `.score_all(query)` returns a {product_id: score} dict
(higher = more relevant). The evaluation stage and the search app both use that.

    1. TfidfRanker      - keyword matching with TF-IDF + cosine similarity
    2. BM25Ranker       - improved keyword ranking (implemented from scratch)
    3. TransformerRanker - semantic ranking with a Sentence-Transformer model
    4. reciprocal_rank_fusion() - the Hybrid model that fuses two rankings (RRF)
"""

import math
from collections import defaultdict

import numpy as np

from preprocess import preprocess, preprocess_to_string


# ======================================================================
# 1. TF-IDF  (uses scikit-learn)
# ======================================================================
class TfidfRanker:
    name = "TF-IDF"

    def __init__(self, product_ids, titles):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.product_ids = list(product_ids)
        # our own preprocess() does tokenizing + stopwords + stemming
        self.vectorizer = TfidfVectorizer(
            tokenizer=preprocess, lowercase=False, token_pattern=None)
        self.matrix = self.vectorizer.fit_transform(titles)   # (N_products, vocab)

    def score_all(self, query):
        from sklearn.metrics.pairwise import linear_kernel
        qv = self.vectorizer.transform([query])
        # matrix is L2-normalised by TfidfVectorizer, so dot product == cosine
        scores = linear_kernel(qv, self.matrix).ravel()
        return dict(zip(self.product_ids, scores))


# ======================================================================
# 2. BM25  (from scratch — the classic Okapi BM25)
# ======================================================================
class BM25Ranker:
    name = "BM25"

    def __init__(self, product_ids, titles, k1=1.5, b=0.75):
        self.product_ids = list(product_ids)
        self.k1 = k1
        self.b = b
        self.N = len(titles)

        # tokenize every product title with the same preprocessing
        docs = [preprocess(t) for t in titles]
        self.doc_len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = self.doc_len.mean() if self.N else 0.0

        # inverted index:  term -> list of (doc_index, term_frequency)
        self.postings = defaultdict(list)
        df = defaultdict(int)                       # document frequency
        for i, doc in enumerate(docs):
            tf = defaultdict(int)
            for term in doc:
                tf[term] += 1
            for term, freq in tf.items():
                self.postings[term].append((i, freq))
                df[term] += 1

        # idf for every term:  ln(1 + (N - df + 0.5)/(df + 0.5))
        self.idf = {t: math.log(1 + (self.N - dfi + 0.5) / (dfi + 0.5))
                    for t, dfi in df.items()}

    def score_all(self, query):
        scores = np.zeros(self.N)
        for term in preprocess(query):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for i, tf in self.postings[term]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return dict(zip(self.product_ids, scores))


# ======================================================================
# 3. Sentence Transformer  (semantic / meaning-based)
# ======================================================================
class TransformerRanker:
    name = "Sentence Transformer"

    def __init__(self, product_ids, titles, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.product_ids = list(product_ids)
        self.model = SentenceTransformer(model_name)
        # raw titles (no stemming) — the model does its own tokenizing.
        # normalize so a dot product equals cosine similarity.
        self.embeddings = self.model.encode(
            list(titles), normalize_embeddings=True, show_progress_bar=True)

    def score_all(self, query):
        qv = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ qv
        return dict(zip(self.product_ids, scores))


# ======================================================================
# 4. Hybrid — Reciprocal Rank Fusion (RRF)
# ======================================================================
def ranking_from_scores(score_dict):
    """Turn a {pid: score} dict into a ranked list of pids (best first)."""
    return [pid for pid, _ in sorted(
        score_dict.items(), key=lambda kv: kv[1], reverse=True)]


def reciprocal_rank_fusion(score_dicts, k=60):
    """
    Combine several rankings into one. Each product gets points based on its
    POSITION in each ranking: 1 / (k + rank).  Points are summed across methods.
    A product ranked high by several methods rises to the top.

    score_dicts : list of {pid: score} dicts (e.g. BM25's and the Transformer's)
    Returns a fused {pid: rrf_score} dict.
    """
    fused = defaultdict(float)
    for sd in score_dicts:
        for rank, pid in enumerate(ranking_from_scores(sd)):
            fused[pid] += 1.0 / (k + rank + 1)
    return dict(fused)


class RRFRanker:
    """Convenience wrapper: fuses the rankings of the given rankers."""
    name = "Hybrid (RRF)"

    def __init__(self, rankers, k=60):
        self.rankers = rankers
        self.k = k

    def score_all(self, query):
        return reciprocal_rank_fusion(
            [r.score_all(query) for r in self.rankers], k=self.k)
