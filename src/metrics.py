"""
Stage 4a — Evaluation metrics.

Every metric takes `ranked_labels`: the ESCI labels of a query's products,
sorted by the ranker from best to worst. Example: ['E','E','S','I','C', ...].

Relevance definitions (kept in ONE place so they're easy to change/explain):
  - For Precision / Recall / MAP (yes-or-no relevance) we count Exact and
    Substitute as relevant, Complement and Irrelevant as not.
  - For nDCG (graded relevance) we give each label a "gain":
        Exact=3, Substitute=2, Complement=1, Irrelevant=0
    so getting an Exact right is worth more than a Complement.
"""

import numpy as np

RELEVANT_LABELS = {"E", "S"}          # counts as relevant for P / R / MAP
GAIN = {"E": 3, "S": 2, "C": 1, "I": 0}   # graded gains for nDCG


def _rel(labels):
    """1 if relevant else 0, for each label."""
    return [1 if l in RELEVANT_LABELS else 0 for l in labels]


def precision_at_k(ranked_labels, k=10):
    """Of the top-k results, what fraction are relevant."""
    rel = _rel(ranked_labels[:k])
    if not rel:
        return 0.0
    return sum(rel) / len(rel)


def recall_at_k(ranked_labels, k=10):
    """Of all relevant products, how many appear in the top-k."""
    rel_all = _rel(ranked_labels)
    total_relevant = sum(rel_all)
    if total_relevant == 0:
        return None                    # undefined; caller skips it
    return sum(rel_all[:k]) / total_relevant


def average_precision(ranked_labels):
    """Average precision — rewards putting relevant items high up. (Used for MAP.)"""
    rel = _rel(ranked_labels)
    total_relevant = sum(rel)
    if total_relevant == 0:
        return None
    hits = 0
    score = 0.0
    for i, r in enumerate(rel):
        if r:
            hits += 1
            score += hits / (i + 1)    # precision at this position
    return score / total_relevant


def dcg(gains):
    """Discounted Cumulative Gain: reward relevant items, discount by position."""
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_labels, k=10):
    """Normalised DCG: how close our ranking is to the perfect ranking (0..1)."""
    gains = [GAIN[l] for l in ranked_labels[:k]]
    ideal = sorted([GAIN[l] for l in ranked_labels], reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return None
    return dcg(gains) / idcg
