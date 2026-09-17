"""
Small search helper shared by the Streamlit app (Stage 5).
Kept free of any streamlit import so it can be tested on its own.
"""

from ranking import ranking_from_scores


def search_catalog(ranker, query, product_ids, title_of, top_n=10,
                   labels=None):
    """
    Rank the WHOLE catalog for `query` and return the top_n results.

    ranker      : one of the ranker objects from ranking.py
    product_ids : list of all product ids
    title_of    : {product_id: title}
    labels      : optional {product_id: 'E'/'S'/'C'/'I'} for the current query,
                  used only to show the ground-truth label in the demo.

    Returns a list of dicts: rank, product_id, title, score, label.
    """
    scores = ranker.score_all(query)
    ranked = ranking_from_scores(scores)[:top_n]
    results = []
    for i, pid in enumerate(ranked, start=1):
        results.append({
            "rank": i,
            "product_id": pid,
            "title": title_of.get(pid, ""),
            "score": round(float(scores.get(pid, 0.0)), 4),
            "label": (labels or {}).get(pid, ""),
        })
    return results
