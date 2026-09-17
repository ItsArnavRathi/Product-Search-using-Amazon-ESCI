"""
Stage 4 — Run every ranking method on every query, score them, and produce:
    results/comparison_table.csv   (Precision@10, Recall@10, MAP, nDCG@10)
    results/pr_curve.png           (precision-recall graph)

Usage:
    python src/evaluate.py
"""

import os
import pandas as pd
import numpy as np

from ranking import (TfidfRanker, BM25Ranker, TransformerRanker,
                     RRFRanker, ranking_from_scores)
import metrics as M

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
DATA = os.path.join(PROJECT, "data")
RESULTS = os.path.join(PROJECT, "results")

K = 10   # the "@k" cut-off used for Precision, Recall and nDCG


# ----------------------------------------------------------------------
# Rank each query's own candidate products and collect their labels
# ----------------------------------------------------------------------
def ranked_labels_per_query(ranker, judg):
    """For each query: sort its judged products with the ranker and return
    the list of their ESCI labels in ranked order."""
    out = {}
    for qid, group in judg.groupby("query_id"):
        query = group["query"].iloc[0]
        cand_ids = group["product_id"].tolist()
        label_of = dict(zip(group["product_id"], group["esci_label"]))
        scores = ranker.score_all(query)                 # {pid: score} over catalog
        ranked = ranking_from_scores({p: scores.get(p, 0.0) for p in cand_ids})
        out[qid] = [label_of[p] for p in ranked]
    return out


def evaluate_one(ranker, judg, k=K):
    """Return average metrics for a single ranker."""
    per_q = ranked_labels_per_query(ranker, judg)
    p, r, ap, nd = [], [], [], []
    for labels in per_q.values():
        p.append(M.precision_at_k(labels, k))
        rec = M.recall_at_k(labels, k)
        if rec is not None:
            r.append(rec)
        a = M.average_precision(labels)
        if a is not None:
            ap.append(a)
        n = M.ndcg_at_k(labels, k)
        if n is not None:
            nd.append(n)
    return {
        f"Precision@{k}": np.mean(p),
        f"Recall@{k}": np.mean(r),
        "MAP": np.mean(ap),
        f"nDCG@{k}": np.mean(nd),
    }, per_q


# ----------------------------------------------------------------------
# Precision-Recall curve data: sweep the cut-off k and average across queries
# ----------------------------------------------------------------------
def pr_curve_points(per_q, max_k=20):
    xs, ys = [], []
    for k in range(1, max_k + 1):
        precs, recs = [], []
        for labels in per_q.values():
            precs.append(M.precision_at_k(labels, k))
            rec = M.recall_at_k(labels, k)
            if rec is not None:
                recs.append(rec)
        xs.append(np.mean(recs))
        ys.append(np.mean(precs))
    return xs, ys


def plot_pr(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"TF-IDF": "#4C78A8", "BM25": "#F58518",
              "Sentence Transformer": "#54A24B", "Hybrid (RRF)": "#E45756"}
    plt.figure(figsize=(7, 5))
    for name, (xs, ys) in curves.items():
        plt.plot(xs, ys, marker="o", markersize=3,
                 color=colors.get(name), label=name, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall by ranking method (averaged over queries)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"  saved {path}")


# ----------------------------------------------------------------------
def build_rankers(prod):
    pids = prod["product_id"].tolist()
    titles = prod["product_title"].fillna("").tolist()
    print("Building rankers (Transformer will download a small model once)...")
    tfidf = TfidfRanker(pids, titles)
    bm25 = BM25Ranker(pids, titles)
    st = TransformerRanker(pids, titles)
    rrf = RRFRanker([bm25, st])     # fuse the two strongest signals
    return [tfidf, bm25, st, rrf]


def main():
    os.makedirs(RESULTS, exist_ok=True)
    prod = pd.read_csv(os.path.join(DATA, "products.csv"))
    judg = pd.read_csv(os.path.join(DATA, "judgments.csv"))
    print(f"Loaded {prod['product_id'].nunique()} products, "
          f"{judg['query_id'].nunique()} queries.\n")

    rankers = build_rankers(prod)

    table, curves = {}, {}
    for r in rankers:
        print(f"Evaluating {r.name} ...")
        row, per_q = evaluate_one(r, judg)
        table[r.name] = row
        curves[r.name] = pr_curve_points(per_q)

    df = pd.DataFrame(table).T.round(4)
    df.to_csv(os.path.join(RESULTS, "comparison_table.csv"))
    plot_pr(curves, os.path.join(RESULTS, "pr_curve.png"))

    print("\n=== Comparison table ===")
    print(df.to_string())


if __name__ == "__main__":
    main()
