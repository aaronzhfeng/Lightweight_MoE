import torch
import numpy as np

def infonce_loss(query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Computes the InfoNCE contrastive loss given a batch of query and document embeddings.
    Uses in-batch negatives: each query's positive document is the matching index in doc_embeddings.
    query_embeddings: Tensor of shape [B, D]
    doc_embeddings: Tensor of shape [B, D]  (assumed aligned so that doc_embeddings[i] is the positive for query[i])
    """
    # Normalize embeddings if using cosine similarity (optional)
    # query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    # doc_embeddings   = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
    # Similarity scores between each query and each doc in the batch
    scores = torch.matmul(query_embeddings, doc_embeddings.T)  # shape [B, B]
    # Optionally divide by temperature to scale the logits
    if temperature != 1.0:
        scores = scores / temperature
    # Labels: 0,1,...,B-1 (the correct doc for each query is at the same index)
    labels = torch.arange(scores.size(0)).to(query_embeddings.device)
    # Use cross-entropy between softmax(scores) and labels (InfoNCE objective)
    loss = torch.nn.CrossEntropyLoss()(scores, labels)
    return loss

def recall_at_k(ranked_list, ground_truth: set, k: int) -> float:
    """Compute Recall@K: proportion of ground-truth documents in the top-K ranked_list."""
    top_k = ranked_list[:k]
    # If any of the ground truth docs is in top_k, then for single relevant it's binary,
    # If multiple, we count how many.
    hits = len(set(top_k) & ground_truth)
    # For recall metric, divide by total relevant count (for single relevant, it's hits itself).
    return hits / len(ground_truth) if ground_truth else 0.0

def dcg_at_k(ranked_list, ground_truth: set, k: int) -> float:
    """Compute Discounted Cumulative Gain at K."""
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_list[:k], start=1):
        rel = 1.0 if doc_id in ground_truth else 0.0  # assuming binary relevance
        # DCG formula: sum_{doc in topK} (2^rel - 1) / log2(1+rank). Here rel is binary (0 or 1).
        dcg += (2**rel - 1) / np.log2(rank + 1)
    return dcg

def ndcg_at_k(ranked_list, ground_truth: set, k: int) -> float:
    """Compute NDCG@K (Normalized DCG)."""
    dcg = dcg_at_k(ranked_list, ground_truth, k)
    # Ideal DCG (IDCG) at K: all relevant documents (up to K) ranked at top
    ideal_list = sorted([1.0]*len(ground_truth) + [0.0]* (k - len(ground_truth)), reverse=True)[:k]
    # Compute DCG for the ideal list
    ideal_dcg = 0.0
    for rank, rel in enumerate(ideal_list, start=1):
        ideal_dcg += (2**rel - 1) / np.log2(rank + 1)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def mean_reciprocal_rank(ranked_list, ground_truth: set) -> float:
    """Compute MRR (Mean Reciprocal Rank) for a single query. Returns 0 if no hit."""
    for rank, doc_id in enumerate(ranked_list, start=1):
        if doc_id in ground_truth:
            return 1.0 / rank
    return 0.0
