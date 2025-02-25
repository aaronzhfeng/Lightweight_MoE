# Dense Retrieval with TinyBERT + SB-MoE: An Informal Implementation Report

## Overview
This project builds a dense retrieval system using a lightweight transformer—**TinyBERT-6L**—enhanced with a **Single-Block Mixture-of-Experts (SB-MoE)** layer. Our goal is to improve retrieval performance without sacrificing efficiency. We use the **BEIR dataset** (e.g., “scifact”) as our evaluation benchmark and train the model with an **InfoNCE contrastive loss** using Hugging Face’s Trainer API. The system is optimized for multi-GPU training (up to 8 GPUs with 48GB each) and mixed-precision training to maximize speed and memory efficiency.

## Key Components

### 1. Base Model (TinyBERT-6L)
- **TinyBERT-6L** is selected for its efficiency and compact size.
- It is a distilled version of BERT with only 6 layers, making it ideal when resources are limited.

### 2. SB-MoE Layer
- A Mixture-of-Experts layer is added on top of the TinyBERT final layer.
- Contains 2–4 small feed-forward expert networks.
- A gating mechanism dynamically routes each input (the [CLS] token) to one expert (top-1 selection), ensuring minimal extra computation.

### 3. Dataset (BEIR)
- Utilizes the BEIR dataset (e.g., the “scifact” dataset) for training and evaluation.
- BEIR is a heterogeneous benchmark containing multiple retrieval tasks, making it an excellent testbed for our retriever.

### 4. Training Pipeline
- Fine-tuning is performed using Hugging Face’s Trainer API.
- The training objective is the **InfoNCE loss** with in-batch negatives, encouraging query embeddings to be close to their positive document embeddings and far from negatives.

### 5. Evaluation
- The entire corpus is encoded.
- A **FAISS index** is built for fast similarity search.
- Retrieval metrics such as **Recall@K, NDCG@10, and MRR** are measured.

### 6. Hardware & Efficiency
- Mixed precision training (FP16) is used.
- Multi-GPU distributed training (using DDP) ensures efficiency even on large datasets and high model capacity.

## Summary of Code Files

- **model.py:**  
  Defines the `MoELayer` and `TinyBERTWithMoE` classes, integrating the SB-MoE layer with TinyBERT.

- **data_prep.py:**  
  Loads the BEIR dataset and converts it into (query, positive, negative) triplets for training.

- **utils.py:**  
  Provides helper functions for computing the InfoNCE loss and key retrieval metrics such as Recall@K, NDCG@10, and MRR.

- **train.py:**  
  Implements the training pipeline using Hugging Face’s Trainer API, leveraging mixed precision and multi-GPU training.

- **eval.py:**  
  Encodes the corpus, builds a FAISS index for efficient similarity search, and evaluates the model's retrieval performance.

