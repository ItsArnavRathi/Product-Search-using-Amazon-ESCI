"""
Stage 2 — Text preprocessing.

Cleans raw product titles / queries into a normalised list of word "tokens"
so that every ranking method compares like with like. Steps:

    lowercase  ->  remove punctuation  ->  tokenize  ->  remove stopwords  ->  stem

Example:
    "Sony WH-1000XM4 Wireless Bluetooth Headphones, Noise-Cancelling!"
      -> ['soni', 'wh', '1000xm4', 'wireless', 'bluetooth', 'headphon', 'nois', 'cancel']

Stemming reduces a word to its root ("running", "runs", "ran"* -> "run") so that
"headphones" and "headphone" match. We use NLTK's Porter stemmer (the classic
IR choice); if NLTK isn't installed the code falls back to a small built-in
stemmer so it still runs.
"""

import re

# ----------------------------------------------------------------------
# 1. Stopwords: extremely common words that carry little search meaning.
#    A short hand-picked list keeps things simple and dependency-free.
# ----------------------------------------------------------------------
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with", "or", "this", "these", "those", "your", "you",
    "i", "we", "they", "them", "our", "us", "but", "not", "so", "if", "up",
    "out", "about", "into", "over", "then", "than", "too", "very", "can",
}

# ----------------------------------------------------------------------
# 2. Stemmer: prefer NLTK's Porter stemmer, else a light fallback.
# ----------------------------------------------------------------------
try:
    from nltk.stem import PorterStemmer
    _porter = PorterStemmer()

    def stem(word):
        return _porter.stem(word)

    STEMMER_NAME = "nltk.PorterStemmer"
except Exception:  # NLTK not available -> simple suffix stripper
    def stem(word):
        for suffix in ("ing", "edly", "ies", "ied", "ly", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)]
        return word

    STEMMER_NAME = "builtin-fallback"

# match runs of letters/digits; drops punctuation automatically
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase and split text into word tokens (punctuation removed)."""
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())


def preprocess(text, do_stem=True, remove_stopwords=True, min_len=2):
    """
    Full cleaning pipeline. Returns a list of processed tokens.

    do_stem          : apply the stemmer
    remove_stopwords : drop common stopwords
    min_len          : drop tokens shorter than this (keeps '32', 'oz', drops 'x')
    """
    tokens = tokenize(text)
    out = []
    for tok in tokens:
        if len(tok) < min_len:
            continue
        if remove_stopwords and tok in STOPWORDS:
            continue
        if do_stem:
            tok = stem(tok)
        out.append(tok)
    return out


def preprocess_to_string(text, **kwargs):
    """Same as preprocess() but joined back into a single string
    (handy for scikit-learn's TF-IDF vectorizer)."""
    return " ".join(preprocess(text, **kwargs))


# ----------------------------------------------------------------------
# Demo: run `python src/preprocess.py` to see before/after on real titles.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import pandas as pd

    print(f"Stemmer in use: {STEMMER_NAME}\n")

    samples = [
        "Sony WH-1000XM4 Wireless Bluetooth Headphones, Noise-Cancelling!",
        "Nike Revolution 6 Men's Road Running Shoes (Size 10)",
        "Hydro Flask 32 oz. Stainless-Steel Insulated Water Bottle",
    ]
    for s in samples:
        print("RAW  :", s)
        print("CLEAN:", preprocess(s))
        print()

    # If the sample data exists, show it working on the real catalog
    data_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "products.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} products from data/products.csv")
        print("Example cleaned titles:")
        for _, row in df.head(3).iterrows():
            print("  ", row["product_title"])
            print("   ->", preprocess_to_string(row["product_title"]))
