import os
import random
from beir.datasets.data_loader import GenericDataLoader

def load_beir_dataset(data_dir: str, dataset_name: str):
    """
    Loads a BEIR dataset (specified by name) from the given directory.
    Returns corpus, queries, and qrels.
    """
    data_path = os.path.join(data_dir, dataset_name)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="train")
    # corpus: dict of {doc_id: {"title": ..., "text": ...}}
    # queries: dict of {query_id: "query text"}
    # qrels: dict of {query_id: {doc_id: relevance_score}}, typically 1 for relevant docs.
    return corpus, queries, qrels

def generate_triplets(corpus, queries, qrels, num_negatives: int = 1):
    """
    Generate (query, positive_doc, negative_doc) triplets for training.
    For each query with at least one relevant doc, sample the specified number of negatives.
    """
    triplets = []
    all_doc_ids = list(corpus.keys())
    for qid, rels in qrels.items():
        if not rels:
            continue
        query_text = queries.get(qid)
        if query_text is None:
            continue
        # For each relevant doc for this query, create a triplet
        for pos_id, score in rels.items():
            if score <= 0:
                continue  # skip if not truly relevant
            pos_text = corpus[pos_id].get("text", "")
            # Combine title and text if both exist
            title = corpus[pos_id].get("title")
            if title:
                pos_text = title + " " + pos_text
            # Sample negatives (documents not in rels)
            neg_candidates = [doc_id for doc_id in all_doc_ids if doc_id not in rels]
            # If the corpus is huge, consider a smarter negative (e.g., BM25 top non-relevant) for better training
            for _ in range(num_negatives):
                neg_id = random.choice(neg_candidates)
                neg_text = corpus[neg_id].get("text", "")
                title = corpus[neg_id].get("title")
                if title:
                    neg_text = title + " " + neg_text
                triplets.append((query_text, pos_text, neg_text))
    return triplets

# Example usage:
# data_dir = "/path/to/beir/datasets"
# dataset_name = "scifact"  # e.g., one of BEIR's dataset
# corpus, queries, qrels = load_beir_dataset(data_dir, dataset_name)
# train_triplets = generate_triplets(corpus, queries, qrels, num_negatives=1)
