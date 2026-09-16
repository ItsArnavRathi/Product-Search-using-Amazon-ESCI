"""
Stage 1 — Download the Amazon ESCI dataset and build a small English sample.

Run this ONCE on your own machine (needs internet):

    python src/download_data.py

It will:
  1. Download the two official ESCI parquet files into data/raw/
  2. Keep only English (US) rows from the "small" version of the dataset
  3. Randomly pick a manageable number of queries (default 300)
  4. Join in the product titles/descriptions for those queries
  5. Save two clean CSV files that the rest of the project uses:
        data/products.csv    -> product_id, product_title, product_description
        data/judgments.csv   -> query_id, query, product_id, esci_label

The ESCI label is one of: E (Exact), S (Substitute), C (Complement), I (Irrelevant).
These human labels are our "answer key" for evaluation later.
"""

import os
import pandas as pd

# ----------------------------------------------------------------------
# Settings you can tweak
# ----------------------------------------------------------------------
N_QUERIES = 300          # how many queries to keep in the sample
RANDOM_SEED = 42         # fixed seed => same sample every run (reproducible)
LOCALE = "us"            # english = "us"  (other options: "es", "jp")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
RAW_DIR = os.path.join(PROJECT, "data", "raw")
DATA_DIR = os.path.join(PROJECT, "data")

# Direct links to the official parquet files (git-LFS content).
BASE = ("https://media.githubusercontent.com/media/amazon-science/esci-data/"
        "main/shopping_queries_dataset/")
EXAMPLES_URL = BASE + "shopping_queries_dataset_examples.parquet"
PRODUCTS_URL = BASE + "shopping_queries_dataset_products.parquet"


def download(url, dest):
    """Download a file to `dest` if it isn't already there."""
    if os.path.exists(dest):
        print(f"  already have {os.path.basename(dest)}, skipping download")
        return
    import urllib.request
    print(f"  downloading {os.path.basename(dest)} ... (this can take a minute)")
    urllib.request.urlretrieve(url, dest)
    print(f"  saved -> {dest}")


def build_sample(examples, products, n_queries=N_QUERIES,
                 locale=LOCALE, seed=RANDOM_SEED):
    """
    Turn the full ESCI tables into a small sample.
    Kept as a separate function so it is easy to test.

    examples : DataFrame with columns
        example_id, query, query_id, product_id, product_locale,
        esci_label, small_version, large_version, split
    products : DataFrame with columns
        product_id, product_title, product_description, ... , product_locale

    Returns (products_out, judgments_out) as two DataFrames.
    """
    # 1. English + "small" version only
    ex = examples[(examples["product_locale"] == locale) &
                  (examples["small_version"] == 1)].copy()

    # 2. Pick a random set of query_ids
    unique_qids = ex["query_id"].drop_duplicates()
    n_queries = min(n_queries, len(unique_qids))
    chosen_qids = unique_qids.sample(n=n_queries, random_state=seed)
    ex = ex[ex["query_id"].isin(chosen_qids)]

    # 3. judgments table = query -> product -> label
    judgments = ex[["query_id", "query", "product_id", "esci_label"]].copy()

    # 4. products table = only the products referenced by our judgments
    prod = products[products["product_locale"] == locale]
    keep_ids = judgments["product_id"].unique()
    prod = prod[prod["product_id"].isin(keep_ids)]
    products_out = prod[["product_id", "product_title",
                         "product_description"]].drop_duplicates("product_id")

    # tidy up
    products_out = products_out.fillna("").reset_index(drop=True)
    judgments = judgments.reset_index(drop=True)
    return products_out, judgments


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    print("Step 1/3: downloading raw ESCI files")
    ex_path = os.path.join(RAW_DIR, "examples.parquet")
    pr_path = os.path.join(RAW_DIR, "products.parquet")
    download(EXAMPLES_URL, ex_path)
    download(PRODUCTS_URL, pr_path)

    print("Step 2/3: reading + sampling")
    examples = pd.read_parquet(ex_path)
    products = pd.read_parquet(pr_path)
    products_out, judgments = build_sample(examples, products)

    print("Step 3/3: saving clean CSVs")
    products_out.to_csv(os.path.join(DATA_DIR, "products.csv"), index=False)
    judgments.to_csv(os.path.join(DATA_DIR, "judgments.csv"), index=False)

    # quick summary
    print("\nDone! Sample summary")
    print(f"  queries : {judgments['query_id'].nunique()}")
    print(f"  products: {products_out['product_id'].nunique()}")
    print(f"  judgments (query-product pairs): {len(judgments)}")
    print("  label counts:")
    print(judgments["esci_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
