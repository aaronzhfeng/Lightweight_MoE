import os
import random
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset
import matplotlib.pyplot as plt
from functools import partial

from data_prep import load_beir_dataset  # We assume this loads corpus, queries, qrels.
from model import TinyBERTWithMoE

# ---------------------------
# BM25-based Hard Negative Sampling
# ---------------------------
# We'll use rank_bm25 for BM25 retrieval.
from rank_bm25 import BM25Okapi

def generate_hard_triplets(corpus, queries, qrels, num_negatives=1):
    """
    For each query, use BM25 to retrieve hard negatives.
    """
    triplets = []
    # Prepare a list of documents (concatenate title and text if available)
    doc_ids = list(corpus.keys())
    doc_texts = []
    for doc_id in doc_ids:
        text = corpus[doc_id].get("text", "")
        title = corpus[doc_id].get("title")
        if title:
            text = title + " " + text
        doc_texts.append(text)
    # Build BM25 index
    bm25 = BM25Okapi([doc.split() for doc in doc_texts])
    
    for qid, query_text in queries.items():
        # Skip if no positive available
        if qid not in qrels or not qrels[qid]:
            continue
        pos_ids = [doc_id for doc_id, score in qrels[qid].items() if score > 0]
        for pos_id in pos_ids:
            pos_text = corpus[pos_id].get("text", "")
            title = corpus[pos_id].get("title")
            if title:
                pos_text = title + " " + pos_text
            # Get BM25 scores for this query
            tokenized_query = query_text.split()
            bm25_scores = bm25.get_scores(tokenized_query)
            # Get document indices sorted by score (descending)
            sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
            # Exclude all positive docs from BM25 hits
            hard_negs = [doc_ids[i] for i in sorted_indices if doc_ids[i] not in pos_ids]
            if not hard_negs:
                continue
            # Choose the top hard negative
            neg_id = hard_negs[0]
            neg_text = corpus[neg_id].get("text", "")
            title = corpus[neg_id].get("title")
            if title:
                neg_text = title + " " + neg_text
            triplets.append((query_text, pos_text, neg_text))
    return triplets

# ---------------------------
# Configuration
# ---------------------------
model_name = "huawei-noah/TinyBERT_General_6L_768D"
dataset_name = "scifact"          # Example BEIR dataset
data_dir = "datasets"             # Directory containing the dataset
num_experts = 8                   # Number of experts in MoE layer
max_length = 256                  # Maximum sequence length for tokenizer
batch_size = 64                   # Per-device batch size
learning_rate = 5e-5
num_epochs = 20
output_dir = "./retrieval_model"  # Directory for saving the MoE model

os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# 1. Load and prepare dataset with BM25 negatives
# ---------------------------
corpus, queries, qrels = load_beir_dataset(data_dir, dataset_name)
train_triplets = generate_hard_triplets(corpus, queries, qrels, num_negatives=1)

# Convert triplets to a Hugging Face Dataset
train_dataset = Dataset.from_dict({
    "query": [t[0] for t in train_triplets],
    "pos": [t[1] for t in train_triplets],
    "neg": [t[2] for t in train_triplets],
})

# ---------------------------
# 2. Load tokenizer and model
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TinyBERTWithMoE(tinybert_name=model_name, num_experts=num_experts, top_k=2, temperature=5.0)

# ---------------------------
# 3. Tokenize function for triplets
# ---------------------------
def tokenize_triplets(example):
    return {
        "query_input_ids": tokenizer(example["query"], truncation=True, max_length=max_length)["input_ids"],
        "query_attention_mask": tokenizer(example["query"], truncation=True, max_length=max_length)["attention_mask"],
        "pos_input_ids": tokenizer(example["pos"], truncation=True, max_length=max_length)["input_ids"],
        "pos_attention_mask": tokenizer(example["pos"], truncation=True, max_length=max_length)["attention_mask"],
        "neg_input_ids": tokenizer(example["neg"], truncation=True, max_length=max_length)["input_ids"],
        "neg_attention_mask": tokenizer(example["neg"], truncation=True, max_length=max_length)["attention_mask"],
    }

train_dataset = train_dataset.map(tokenize_triplets, batched=True, remove_columns=["query", "pos", "neg"])

# ---------------------------
# 4. Custom Data Collator for Triplet Inputs
# ---------------------------
def custom_collate_fn(batch, tokenizer):
    query_inputs = {"input_ids": [], "attention_mask": []}
    pos_inputs = {"input_ids": [], "attention_mask": []}
    neg_inputs = {"input_ids": [], "attention_mask": []}
    
    for example in batch:
        query_inputs["input_ids"].append(example["query_input_ids"])
        query_inputs["attention_mask"].append(example["query_attention_mask"])
        pos_inputs["input_ids"].append(example["pos_input_ids"])
        pos_inputs["attention_mask"].append(example["pos_attention_mask"])
        neg_inputs["input_ids"].append(example["neg_input_ids"])
        neg_inputs["attention_mask"].append(example["neg_attention_mask"])
    
    padded_query = tokenizer.pad(query_inputs, return_tensors="pt")
    padded_pos   = tokenizer.pad(pos_inputs, return_tensors="pt")
    padded_neg   = tokenizer.pad(neg_inputs, return_tensors="pt")
    
    return {
        "query_input_ids": padded_query["input_ids"],
        "query_attention_mask": padded_query["attention_mask"],
        "pos_input_ids": padded_pos["input_ids"],
        "pos_attention_mask": padded_pos["attention_mask"],
        "neg_input_ids": padded_neg["input_ids"],
        "neg_attention_mask": padded_neg["attention_mask"],
    }

data_collator = lambda batch: custom_collate_fn(batch, tokenizer)

# ---------------------------
# 5. Define custom Trainer with Triplet Loss, dynamic scheduling, and gradient clipping
# ---------------------------
class RetrievalTrainer(Trainer):
    def __init__(self, aux_weight_init=0.02, aux_weight_final=0.005, **kwargs):
        super().__init__(**kwargs)
        self.aux_weight = aux_weight_init  # initial weight for auxiliary loss
        self.aux_weight_init = aux_weight_init
        self.aux_weight_final = aux_weight_final

    def compute_loss(self, model, inputs, return_outputs=False):
        q_input_ids = inputs["query_input_ids"]
        q_att_mask = inputs["query_attention_mask"]
        p_input_ids = inputs["pos_input_ids"]
        p_att_mask = inputs["pos_attention_mask"]
        n_input_ids = inputs["neg_input_ids"]
        n_att_mask = inputs["neg_attention_mask"]
        
        # Concatenate positive and negative inputs along the batch dimension
        doc_input_ids = torch.cat([p_input_ids, n_input_ids], dim=0)
        doc_att_mask = torch.cat([p_att_mask, n_att_mask], dim=0)
        
        # Get embeddings from the model
        q_emb = model(input_ids=q_input_ids, attention_mask=q_att_mask)           # [B, hidden]
        doc_emb = model(input_ids=doc_input_ids, attention_mask=doc_att_mask)       # [2B, hidden]
        
        batch_size = q_emb.size(0)
        # Compute similarity matrix: each query vs. each document embedding
        scores = torch.matmul(q_emb, doc_emb.T)  # [B, 2B]
        labels = torch.arange(batch_size).to(scores.device)  # correct index: 0 ... B-1
        
        loss_fn = torch.nn.CrossEntropyLoss()
        main_loss = loss_fn(scores, labels)
        
        # Retrieve auxiliary load-balancing loss from the model (computed in the MoE layer)
        aux_loss = getattr(model, "aux_loss", torch.tensor(0.0).to(scores.device))
        total_loss = main_loss + self.aux_weight * aux_loss
        
        # For logging, compute accuracy (optional)
        preds = torch.argmax(scores, dim=1)
        accuracy = (preds == labels).float().mean()
        
        # Accumulate metrics for logging
        if not hasattr(self, "epoch_loss_sum"):
            self.epoch_loss_sum = 0.0
            self.epoch_total = 0
            self.epoch_correct = 0
        self.epoch_loss_sum += total_loss.item() * batch_size
        self.epoch_total += batch_size
        self.epoch_correct += (preds == labels).sum().item()
        
        if return_outputs:
            return (total_loss, {"loss": total_loss.item(), "accuracy": accuracy.item()})
        else:
            return total_loss

    def training_step(self, model, inputs, num_items):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        self.accelerator.backward(loss)
        # Gradient clipping on the gating network parameters
        torch.nn.utils.clip_grad_norm_(model.moe.gate.parameters(), max_norm=1.0)
        return loss.detach()

# ---------------------------
# 6. Callback: DynamicSchedulerCallback to anneal temperature and aux weight
# ---------------------------
class DynamicSchedulerCallback(TrainerCallback):
    def __init__(self, total_epochs, temp_init=5.0, temp_final=1.0, aux_init=0.02, aux_final=0.005):
        self.total_epochs = total_epochs
        self.temp_init = temp_init
        self.temp_final = temp_final
        self.aux_init = aux_init
        self.aux_final = aux_final

    def on_epoch_begin(self, args, state, control, **kwargs):
        # Compute progress (0 to 1)
        progress = state.epoch / self.total_epochs
        # Linear annealing for temperature and auxiliary loss weight
        new_temp = self.temp_init - progress * (self.temp_init - self.temp_final)
        new_aux = self.aux_init - progress * (self.aux_init - self.aux_final)
        # Update model temperature and trainer aux_weight
        model = kwargs.get("model", None)
        if model is not None and hasattr(model, "moe"):
            model.moe.temperature = new_temp
        trainer = kwargs.get("trainer", None)
        if trainer is not None:
            trainer.aux_weight = new_aux
        print(f"Epoch {state.epoch:.2f}: Setting temperature to {new_temp:.4f}, aux weight to {new_aux:.4f}")
        return control

# ---------------------------
# 7. Callback to log epoch metrics to CSV (also can be used for monitoring)
# ---------------------------
class EpochMetricsLogger(TrainerCallback):
    def __init__(self, output_dir):
        self.filepath = os.path.join(output_dir, "training_metrics.csv")
        # Write CSV header
        with open(self.filepath, "w") as f:
            f.write("epoch,train_loss,train_accuracy\n")
        self.trainer = None

    def on_train_begin(self, args, state, control, **kwargs):
        # Capture trainer instance at the start of training
        trainer = kwargs.get("trainer", None)
        if trainer is not None:
            self.trainer = trainer
        return control

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control
        epoch = state.epoch
        loss_sum = getattr(self.trainer, "epoch_loss_sum", 0.0)
        total = getattr(self.trainer, "epoch_total", 0)
        correct = getattr(self.trainer, "epoch_correct", 0)
        avg_loss = loss_sum / total if total > 0 else 0.0
        avg_acc = correct / total if total > 0 else 0.0

        with open(self.filepath, "a") as f:
            f.write(f"{epoch},{avg_loss:.4f},{avg_acc:.4f}\n")
        # Reset accumulators
        self.trainer.epoch_loss_sum = 0.0
        self.trainer.epoch_total = 0
        self.trainer.epoch_correct = 0
        return control


# ---------------------------
# 8. Set up TrainingArguments
# ---------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=False,
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,
    num_train_epochs=num_epochs,
    fp16=True,
    logging_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
    save_total_limit=1,
    remove_unused_columns=False,
)

# ---------------------------
# 9. Initialize Trainer and add callbacks
# ---------------------------
trainer = RetrievalTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.add_callback(EpochMetricsLogger(output_dir))
trainer.add_callback(DynamicSchedulerCallback(total_epochs=num_epochs, temp_init=5.0, temp_final=1.0,
                                               aux_init=0.02, aux_final=0.005))

# ---------------------------
# 10. Train the model and save it
# ---------------------------
train_result = trainer.train()
trainer.save_model(output_dir)
print(f"Fine-tuned TinyBERT+MoE model saved to: {output_dir}")
