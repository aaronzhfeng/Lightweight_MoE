import os
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import safetensors.torch

from model import TinyBERTWithMoE
from data_prep import load_beir_dataset
from utils import recall_at_k, ndcg_at_k, mean_reciprocal_rank

# ---------------------------
# Configuration
# ---------------------------
# Directories for the two fine-tuned models:
baseline_model_dir = "./retrieval_model_tinybert"  # newly fine-tuned baseline TinyBERT
moe_model_dir = "./retrieval_model"                # fine-tuned TinyBERT+MoE

base_model_name = "huawei-noah/TinyBERT_General_6L_768D"
dataset_name = "scifact"
data_dir = "datasets"

result_dir = "./evaluation_results"
os.makedirs(result_dir, exist_ok=True)

K = 1  # we'll compute Recall@10

# ---------------------------
# 1. Load the dataset
# ---------------------------
corpus, queries, qrels = load_beir_dataset(data_dir, dataset_name)

# Convert corpus to lists for easier indexing
doc_ids = list(corpus.keys())
doc_texts = []
for doc_id in doc_ids:
    text = corpus[doc_id].get("text", "")
    title = corpus[doc_id].get("title")
    if title:
        text = title + " " + text
    doc_texts.append(text)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# 2. Initialize Tokenizer
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# ============================================================
# Evaluate Baseline TinyBERT (Fine-Tuned)
# ============================================================
print("Evaluating Fine-Tuned Baseline TinyBERT...")

# Load the fine-tuned baseline from retrieval_model_tinybert
baseline_model = AutoModel.from_pretrained(baseline_model_dir)
baseline_model.eval()
baseline_model.to(device)

# Encode corpus
batch_size = 128
doc_embeddings_base = []
with torch.no_grad():
    for i in range(0, len(doc_texts), batch_size):
        batch_texts = doc_texts[i : i + batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = baseline_model(**enc)
        # Use [CLS] token embedding
        cls_emb = outputs.last_hidden_state[:, 0, :]
        doc_embeddings_base.append(cls_emb.cpu().numpy())
doc_embeddings_base = np.vstack(doc_embeddings_base)

# Build FAISS index for baseline embeddings
dim = doc_embeddings_base.shape[1]
index_base = faiss.IndexFlatIP(dim)
index_base.add(doc_embeddings_base)

# Retrieval and metrics
all_recalls_base, all_ndcgs_base, all_mrrs_base = [], [], []
with torch.no_grad():
    for qid, query_text in queries.items():
        enc = tokenizer(query_text, padding=True, truncation=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = baseline_model(**enc)
        q_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # Search
        D, I = index_base.search(q_emb, K)
        top_doc_indices = I[0]
        retrieved_doc_ids = [doc_ids[idx] for idx in top_doc_indices]
        gold_docs = set(qrels.get(qid, {}).keys())

        all_recalls_base.append(recall_at_k(retrieved_doc_ids, gold_docs, K))
        all_ndcgs_base.append(ndcg_at_k(retrieved_doc_ids, gold_docs, 10))
        all_mrrs_base.append(mean_reciprocal_rank(retrieved_doc_ids, gold_docs))

baseline_recall = np.mean(all_recalls_base)
baseline_ndcg   = np.mean(all_ndcgs_base)
baseline_mrr    = np.mean(all_mrrs_base)

print(f"[Fine-Tuned TinyBERT] Recall@{K}: {baseline_recall:.4f}, "
      f"NDCG@10: {baseline_ndcg:.4f}, MRR: {baseline_mrr:.4f}")

# ============================================================
# Evaluate Fine-Tuned TinyBERT+MoE
# ============================================================
print("Evaluating Fine-Tuned TinyBERT+MoE...")

# Load the MoE model from retrieval_model (saved by original train.py)
# We use safetensors to load the state dict (if that's how you saved it),
# otherwise you can do model.load_state_dict(torch.load(...))
moe_model_path = os.path.join(moe_model_dir, "model.safetensors")
state_dict = safetensors.torch.load_file(moe_model_path, device="cpu")

moe_model = TinyBERTWithMoE(tinybert_name=base_model_name, num_experts=8)
moe_model.load_state_dict(state_dict)
moe_model.eval()
moe_model.to(device)

# Encode corpus with MoE
doc_embeddings_moe = []
with torch.no_grad():
    for i in range(0, len(doc_texts), batch_size):
        batch_texts = doc_texts[i : i + batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        emb = moe_model(enc["input_ids"], attention_mask=enc["attention_mask"])
        # moe_model(...) returns [CLS] embedding directly
        doc_embeddings_moe.append(emb.cpu().numpy())
doc_embeddings_moe = np.vstack(doc_embeddings_moe)

# Build FAISS index for MoE embeddings
dim_moe = doc_embeddings_moe.shape[1]
index_moe = faiss.IndexFlatIP(dim_moe)
index_moe.add(doc_embeddings_moe)

# Retrieval and metrics for MoE
all_recalls_moe, all_ndcgs_moe, all_mrrs_moe = [], [], []
with torch.no_grad():
    for qid, query_text in queries.items():
        enc = tokenizer(query_text, padding=True, truncation=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        q_emb = moe_model(enc["input_ids"], attention_mask=enc["attention_mask"])
        q_emb = q_emb.cpu().numpy()

        D, I = index_moe.search(q_emb, K)
        top_doc_indices = I[0]
        retrieved_doc_ids = [doc_ids[idx] for idx in top_doc_indices]
        gold_docs = set(qrels.get(qid, {}).keys())

        all_recalls_moe.append(recall_at_k(retrieved_doc_ids, gold_docs, K))
        all_ndcgs_moe.append(ndcg_at_k(retrieved_doc_ids, gold_docs, 10))
        all_mrrs_moe.append(mean_reciprocal_rank(retrieved_doc_ids, gold_docs))

moe_recall = np.mean(all_recalls_moe)
moe_ndcg   = np.mean(all_ndcgs_moe)
moe_mrr    = np.mean(all_mrrs_moe)

print(f"[Fine-Tuned TinyBERT+MoE] Recall@{K}: {moe_recall:.4f}, "
      f"NDCG@10: {moe_ndcg:.4f}, MRR: {moe_mrr:.4f}")

# -----------------------------------------------------------
# Print side-by-side comparison
# -----------------------------------------------------------
print("\n==== Final Comparison ====")
print(f"Baseline TinyBERT  - Recall@{K}: {baseline_recall:.4f}, NDCG@10: {baseline_ndcg:.4f}, MRR: {baseline_mrr:.4f}")
print(f"TinyBERT+MoE       - Recall@{K}: {moe_recall:.4f}, NDCG@10: {moe_ndcg:.4f}, MRR: {moe_mrr:.4f}")

# Optionally write results to a single CSV
with open(os.path.join(result_dir, "comparison_metrics.csv"), "w") as f:
    f.write("Model,Recall@10,NDCG@10,MRR\n")
    f.write(f"FineTunedTinyBERT,{baseline_recall:.4f},{baseline_ndcg:.4f},{baseline_mrr:.4f}\n")
    f.write(f"FineTunedTinyBERT+MoE,{moe_recall:.4f},{moe_ndcg:.4f},{moe_mrr:.4f}\n")

print(f"Metrics written to {os.path.join(result_dir, 'comparison_metrics.csv')}")
