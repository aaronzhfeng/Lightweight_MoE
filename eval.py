import os
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from model import TinyBERTWithMoE
from data_prep import load_beir_dataset
from utils import recall_at_k, ndcg_at_k, mean_reciprocal_rank
import safetensors.torch

# ---------------------------
# Configuration
# ---------------------------
model_dir = "./retrieval_model"          # Directory for fine-tuned TinyBERT+MoE model
base_model_name = "huawei-noah/TinyBERT_General_6L_768D"  # Baseline TinyBERT without MoE
dataset_name = "scifact"
data_dir = "datasets"
result_dir = "./evaluation_results"
os.makedirs(result_dir, exist_ok=True)   # Create folder for evaluation results
K = 10

# ---------------------------
# 1. Load the dataset
# ---------------------------
corpus, queries, qrels = load_beir_dataset(data_dir, dataset_name)

# ---------------------------
# 2. Initialize Tokenizer (same for both models)
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# ---------------------------
# 3A. Evaluate Baseline TinyBERT (without MoE)
# ---------------------------
print("Evaluating Baseline TinyBERT...")
baseline_model = AutoModel.from_pretrained(base_model_name)
baseline_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
baseline_model.to(device)

# Encode all corpus documents using baseline model
doc_ids = list(corpus.keys())
doc_texts = []
for doc_id in doc_ids:
    text = corpus[doc_id].get("text", "")
    title = corpus[doc_id].get("title")
    if title:
        text = title + " " + text
    doc_texts.append(text)

batch_size = 128
doc_embeddings_base = []
with torch.no_grad():
    for i in range(0, len(doc_texts), batch_size):
        batch_texts = doc_texts[i:i+batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = baseline_model(**enc)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        doc_embeddings_base.append(cls_emb.cpu().numpy())
doc_embeddings_base = np.vstack(doc_embeddings_base)

# Build FAISS index for baseline embeddings
dim = doc_embeddings_base.shape[1]
index_base = faiss.IndexFlatIP(dim)
index_base.add(doc_embeddings_base)

# Retrieve and evaluate for baseline model
all_recalls_base, all_ndcgs_base, all_mrrs_base = [], [], []
with torch.no_grad():
    for qid, query_text in queries.items():
        enc = tokenizer(query_text, truncation=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = baseline_model(**enc)
        q_cls_emb = outputs.last_hidden_state[:, 0, :]
        q_vec = q_cls_emb.cpu().numpy()
        D, I = index_base.search(q_vec, K)
        top_doc_indices = I[0]
        retrieved_doc_ids = [doc_ids[idx] for idx in top_doc_indices]
        gold_docs = set(qrels.get(qid, {}).keys())
        all_recalls_base.append(recall_at_k(retrieved_doc_ids, gold_docs, K))
        all_ndcgs_base.append(ndcg_at_k(retrieved_doc_ids, gold_docs, 10))
        all_mrrs_base.append(mean_reciprocal_rank(retrieved_doc_ids, gold_docs))

baseline_recall = np.mean(all_recalls_base)
baseline_ndcg   = np.mean(all_ndcgs_base)
baseline_mrr    = np.mean(all_mrrs_base)
print(f"Baseline TinyBERT Results - Recall@{K}: {baseline_recall:.4f}, NDCG@10: {baseline_ndcg:.4f}, MRR: {baseline_mrr:.4f}")

with open(os.path.join(result_dir, "baseline_metrics.csv"), "w") as f:
    f.write("Model,Recall@10,NDCG@10,MRR\n")
    f.write(f"BaselineTinyBERT,{baseline_recall:.4f},{baseline_ndcg:.4f},{baseline_mrr:.4f}\n")

# ---------------------------
# 3B. Evaluate Fine-tuned TinyBERT+MoE
# ---------------------------
print("Evaluating TinyBERT+MoE...")
# Load fine-tuned MoE model; note: use same tokenizer as baseline
model_path = os.path.join(model_dir, "model.safetensors")
state_dict = safetensors.torch.load_file(model_path, device="cpu")
moe_model = TinyBERTWithMoE(tinybert_name=base_model_name, num_experts=4)
moe_model.load_state_dict(state_dict)
moe_model.eval()
moe_model.to(device)

# Encode corpus using MoE model
doc_embeddings_moe = []
with torch.no_grad():
    for i in range(0, len(doc_texts), batch_size):
        batch_texts = doc_texts[i:i+batch_size]
        enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        emb = moe_model(enc["input_ids"], attention_mask=enc["attention_mask"])
        doc_embeddings_moe.append(emb.cpu().numpy())
doc_embeddings_moe = np.vstack(doc_embeddings_moe)

# Build FAISS index for MoE embeddings
dim = doc_embeddings_moe.shape[1]
index_moe = faiss.IndexFlatIP(dim)
index_moe.add(doc_embeddings_moe)

# Retrieve and evaluate for MoE model
all_recalls_moe, all_ndcgs_moe, all_mrrs_moe = [], [], []
with torch.no_grad():
    for qid, query_text in queries.items():
        enc = tokenizer(query_text, truncation=True, max_length=256, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        q_emb = moe_model(enc["input_ids"], attention_mask=enc["attention_mask"])
        q_vec = q_emb.cpu().numpy()
        D, I = index_moe.search(q_vec, K)
        top_doc_indices = I[0]
        retrieved_doc_ids = [doc_ids[idx] for idx in top_doc_indices]
        gold_docs = set(qrels.get(qid, {}).keys())
        all_recalls_moe.append(recall_at_k(retrieved_doc_ids, gold_docs, K))
        all_ndcgs_moe.append(ndcg_at_k(retrieved_doc_ids, gold_docs, 10))
        all_mrrs_moe.append(mean_reciprocal_rank(retrieved_doc_ids, gold_docs))

moe_recall = np.mean(all_recalls_moe)
moe_ndcg   = np.mean(all_ndcgs_moe)
moe_mrr    = np.mean(all_mrrs_moe)
print(f"TinyBERT+MoE Results - Recall@{K}: {moe_recall:.4f}, NDCG@10: {moe_ndcg:.4f}, MRR: {moe_mrr:.4f}")

with open(os.path.join(result_dir, "moe_metrics.csv"), "w") as f:
    f.write("Model,Recall@10,NDCG@10,MRR\n")
    f.write(f"TinyBERT_MoE,{moe_recall:.4f},{moe_ndcg:.4f},{moe_mrr:.4f}\n")
