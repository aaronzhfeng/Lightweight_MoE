import os
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import safetensors.torch

from model import TinyBERTWithMoE, TinyBERTMoERegular
from data_prep import load_beir_dataset
from utils import recall_at_k, ndcg_at_k, mean_reciprocal_rank

# ---------------------------
# Configuration
# ---------------------------
# Directories for the fine-tuned models
baseline_model_dir = "./retrieval_model_tinybert"  # Fine-tuned baseline TinyBERT
sbmoe_model_dir    = "./retrieval_model_sbmoe"      # Fine-tuned SB-MoE TinyBERT
moe_model_dir      = "./retrieval_model_moe"        # Fine-tuned regular MoE TinyBERT

base_model_name = "huawei-noah/TinyBERT_General_6L_768D"
dataset_name    = "scifact"
data_dir        = "datasets"
result_dir      = "./evaluation_results"
os.makedirs(result_dir, exist_ok=True)

# Set the number of top retrieved items for evaluation
K = 10

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

# ---------------------------
# 3. Load Baseline TinyBERT
# ---------------------------
print("Evaluating Fine-Tuned Baseline TinyBERT...")
baseline_model = AutoModel.from_pretrained(baseline_model_dir)
baseline_model.eval()
baseline_model.to(device)

# ---------------------------
# 4. Load SB-MoE TinyBERT
# ---------------------------
print("Evaluating Fine-Tuned SB-MoE TinyBERT...")
# Here, we assume you saved your SB-MoE model using the TinyBERTWithMoE class.
sbmoe_model_path = os.path.join(sbmoe_model_dir, "model.safetensors")
sbmoe_state = safetensors.torch.load_file(sbmoe_model_path, device="cpu")
sbmoe_model = TinyBERTWithMoE(tinybert_name=base_model_name, num_experts=8)
sbmoe_model.load_state_dict(sbmoe_state)
sbmoe_model.eval()
sbmoe_model.to(device)

# ---------------------------
# 5. Load Regular MoE TinyBERT
# ---------------------------
print("Evaluating Fine-Tuned Regular MoE TinyBERT...")
moe_model_path = os.path.join(moe_model_dir, "model.safetensors")
moe_state = safetensors.torch.load_file(moe_model_path, device="cpu")
# Here we use our Regular MoE class (TinyBERTMoERegular)
moe_model = TinyBERTMoERegular(tinybert_name=base_model_name, num_experts=8, top_k=2, temperature=5.0)
moe_model.load_state_dict(moe_state)
moe_model.eval()
moe_model.to(device)

# ---------------------------
# 6. Encode the Corpus for Each Model and Build FAISS Indices
# ---------------------------
def encode_corpus(model, texts, tokenizer, batch_size, device):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            # For retrieval, we use the [CLS] token embedding
            embeddings.append(outputs.cpu().numpy())
    return np.vstack(embeddings)

print("Encoding corpus with Baseline TinyBERT...")
doc_embeddings_base = encode_corpus(baseline_model, doc_texts, tokenizer, batch_size=128, device=device)
dim = doc_embeddings_base.shape[1]
index_base = faiss.IndexFlatIP(dim)
index_base.add(doc_embeddings_base)

print("Encoding corpus with SB-MoE TinyBERT...")
doc_embeddings_sbmoe = encode_corpus(sbmoe_model, doc_texts, tokenizer, batch_size=128, device=device)
dim_sb = doc_embeddings_sbmoe.shape[1]
index_sbmoe = faiss.IndexFlatIP(dim_sb)
index_sbmoe.add(doc_embeddings_sbmoe)

print("Encoding corpus with Regular MoE TinyBERT...")
doc_embeddings_moe = encode_corpus(moe_model, doc_texts, tokenizer, batch_size=128, device=device)
dim_moe = doc_embeddings_moe.shape[1]
index_moe = faiss.IndexFlatIP(dim_moe)
index_moe.add(doc_embeddings_moe)

# ---------------------------
# 7. Evaluate Retrieval for Each Model
# ---------------------------
def evaluate_model(model, index, queries, qrels, tokenizer, device, K):
    recalls, ndcgs, mrrs = [], [], []
    with torch.no_grad():
        for qid, query_text in queries.items():
            enc = tokenizer(query_text, padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            outputs = model(**enc)
            q_emb = outputs.cpu().numpy()
            D, I = index.search(q_emb, K)
            retrieved_doc_ids = [doc_ids[idx] for idx in I[0]]
            gold_docs = set(qrels.get(qid, {}).keys())
            recalls.append(recall_at_k(retrieved_doc_ids, gold_docs, K))
            ndcgs.append(ndcg_at_k(retrieved_doc_ids, gold_docs, 10))
            mrrs.append(mean_reciprocal_rank(retrieved_doc_ids, gold_docs))
    return np.mean(recalls), np.mean(ndcgs), np.mean(mrrs)

print("Evaluating Baseline TinyBERT...")
baseline_recall, baseline_ndcg, baseline_mrr = evaluate_model(baseline_model, index_base, queries, qrels, tokenizer, device, K)

print("Evaluating SB-MoE TinyBERT...")
sbmoe_recall, sbmoe_ndcg, sbmoe_mrr = evaluate_model(sbmoe_model, index_sbmoe, queries, qrels, tokenizer, device, K)

print("Evaluating Regular MoE TinyBERT...")
moe_recall, moe_ndcg, moe_mrr = evaluate_model(moe_model, index_moe, queries, qrels, tokenizer, device, K)

# ---------------------------
# 8. Print and Save the Results
# ---------------------------
print("\n==== Final Comparison ====")
print(f"Baseline TinyBERT   - Recall@{K}: {baseline_recall:.4f}, NDCG@10: {baseline_ndcg:.4f}, MRR: {baseline_mrr:.4f}")
print(f"SB-MoE TinyBERT     - Recall@{K}: {sbmoe_recall:.4f}, NDCG@10: {sbmoe_ndcg:.4f}, MRR: {sbmoe_mrr:.4f}")
print(f"Regular MoE TinyBERT  - Recall@{K}: {moe_recall:.4f}, NDCG@10: {moe_ndcg:.4f}, MRR: {moe_mrr:.4f}")

comparison_path = os.path.join(result_dir, "comparison_metrics.csv")
with open(comparison_path, "w") as f:
    f.write("Model,Recall@K,NDCG@10,MRR\n")
    f.write(f"BaselineTinyBERT,{baseline_recall:.4f},{baseline_ndcg:.4f},{baseline_mrr:.4f}\n")
    f.write(f"SBMoETinyBERT,{sbmoe_recall:.4f},{sbmoe_ndcg:.4f},{sbmoe_mrr:.4f}\n")
    f.write(f"RegularMoETinyBERT,{moe_recall:.4f},{moe_ndcg:.4f},{moe_mrr:.4f}\n")

print(f"Metrics written to {comparison_path}")
