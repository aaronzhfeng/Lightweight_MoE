import faiss
import torch
import numpy as np
from transformers import AutoTokenizer
from model import TinyBERTWithMoE
from data_prep import load_beir_dataset
from utils import recall_at_k, ndcg_at_k, mean_reciprocal_rank

# Configuration
model_dir = "./retrieval_model"   # directory of the fine-tuned model
dataset_name = "scifact"          # evaluate on the same or another BEIR dataset
data_dir = "datasets"

# 1. Load the dataset (evaluation split)
corpus, queries, qrels = load_beir_dataset(data_dir, dataset_name)
# For evaluation, use BEIR's dev or test split if available. Adjust load_beir_dataset accordingly.

# 2. Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = TinyBERTWithMoE(tinybert_name=model_dir)  # load fine-tuned model weights
model.eval()

# 3. Encode all corpus documents into embeddings
doc_ids = list(corpus.keys())
doc_texts = []
for doc_id in doc_ids:
    text = corpus[doc_id].get("text", "")
    title = corpus[doc_id].get("title")
    if title:
        text = title + " " + text
    doc_texts.append(text)

# Tokenize documents in batches
batch_size = 128
doc_embeddings = []
with torch.no_grad():
    for i in range(0, len(doc_texts), batch_size):
        batch_texts = doc_texts[i:i+batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(model.device) for k,v in enc.items()}
        emb = model(enc["input_ids"], attention_mask=enc["attention_mask"])
        doc_embeddings.append(emb.cpu().numpy())
doc_embeddings = np.vstack(doc_embeddings)  # shape [N_docs, dim]
dim = doc_embeddings.shape[1]

# 4. Build a FAISS index for efficient retrieval
index = faiss.IndexFlatIP(dim)  # Inner-product similarity (FAISS uses IP or L2; IP for cosine if vectors are normalized)
# (Optional: if vectors are not normalized, consider IndexFlatL2 for Euclidean, or normalize them now for cosine sim.)
# Add document vectors to the index
index.add(doc_embeddings)
# (For very large corpora, one might use FaissIVFFlat or HNSW indexes for scalability, and possibly train PCA/OPQ.)

# 5. Retrieve top K documents for each query and evaluate
K = 10
all_recalls, all_ndcgs, all_mrrs = [], [], []
with torch.no_grad():
    for qid, query_text in queries.items():
        # Encode query
        enc = tokenizer(query_text, truncation=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(model.device) for k,v in enc.items()}
        q_emb = model(enc["input_ids"], attention_mask=enc["attention_mask"])
        q_vec = q_emb.cpu().numpy()
        # Search in Faiss index
        D, I = index.search(q_vec, K)  # I: indices of top K docs, shape [1, K]
        top_doc_indices = I[0]
        retrieved_doc_ids = [doc_ids[idx] for idx in top_doc_indices]
        # Compute metrics for this query
        gold_docs = set(qrels.get(qid, {}).keys())  # relevant doc IDs for this query
        all_recalls.append(recall_at_k(retrieved_doc_ids, gold_docs, K))
        all_ndcgs.append(ndcg_at_k(retrieved_doc_ids, gold_docs, 10))  # NDCG@10
        all_mrrs.append(mean_reciprocal_rank(retrieved_doc_ids, gold_docs))
# Aggregate metrics
mean_recall = np.mean(all_recalls)
mean_ndcg = np.mean(all_ndcgs)
mean_mrr = np.mean(all_mrrs)
print(f"Evaluation Results - Recall@{K}: {mean_recall:.4f}, NDCG@10: {mean_ndcg:.4f}, MRR: {mean_mrr:.4f}")
