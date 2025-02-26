import os
import random
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset
from functools import partial
import matplotlib.pyplot as plt

from data_prep import load_beir_dataset, generate_triplets
from model import TinyBERTWithMoE
#from utils import infonce_loss  # (Not used in this version; using CrossEntropy loss directly)

# ---------------------------
# Configuration
# ---------------------------
model_name = "huawei-noah/TinyBERT_General_6L_768D"
dataset_name = "scifact"          # Example BEIR dataset
data_dir = "datasets"             # Directory containing the dataset
num_experts = 4                   # Number of experts in MoE layer
max_length = 256                  # Max sequence length for tokenizer
batch_size = 64                   # Per-device batch size
learning_rate = 5e-5
num_epochs = 20
output_dir = "./retrieval_model"

# Ensure output directory exists for saving model and logs
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# 1. Load and prepare dataset
# ---------------------------
corpus, queries, qrels = load_beir_dataset(data_dir, dataset_name)
train_triplets = generate_triplets(corpus, queries, qrels, num_negatives=1)

# Convert triplets to a Hugging Face Dataset for convenience
train_dataset = Dataset.from_dict({
    "query": [t[0] for t in train_triplets],
    "pos": [t[1] for t in train_triplets],
    "neg": [t[2] for t in train_triplets],
})

# ---------------------------
# 2. Load tokenizer and model
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TinyBERTWithMoE(tinybert_name=model_name, num_experts=num_experts)

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
# 5. Define custom Trainer with Triplet Loss and Logging
# ---------------------------
class RetrievalTrainer(Trainer):
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
        loss = loss_fn(scores, labels)
        
        # Compute triplet accuracy: correct if positive (at index i) has highest score
        preds = torch.argmax(scores, dim=1)
        accuracy = (preds == labels).float().mean()
        
        # Accumulate loss and accuracy for epoch-level logging
        if not hasattr(self, "epoch_loss_sum"):
            self.epoch_loss_sum = 0.0
            self.epoch_total = 0
            self.epoch_correct = 0
        self.epoch_loss_sum += loss.item() * batch_size
        self.epoch_total += batch_size
        self.epoch_correct += (preds == labels).sum().item()
        
        if return_outputs:
            return (loss, {"loss": loss.item(), "accuracy": accuracy.item()})
        else:
            return loss

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        self.accelerator.backward(loss)
        return loss.detach()

# ---------------------------
# Callback to log epoch metrics to CSV
# ---------------------------


class EpochMetricsLogger(TrainerCallback):
    def __init__(self, output_dir):
        self.filepath = os.path.join(output_dir, "training_metrics.csv")
        # Write header to CSV file
        with open(self.filepath, "w") as f:
            f.write("epoch,train_loss,train_accuracy\n")
        self.trainer = None  # Will be set via set_trainer()

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None:
            return control
        epoch = state.epoch
        # Access custom accumulators from the trainer instance
        loss_sum = getattr(self.trainer, "epoch_loss_sum", 0.0)
        total = getattr(self.trainer, "epoch_total", 0)
        correct = getattr(self.trainer, "epoch_correct", 0)
        avg_loss = loss_sum / total if total > 0 else 0.0
        avg_acc = correct / total if total > 0 else 0.0

        # Write metrics for this epoch to CSV
        with open(self.filepath, "a") as f:
            f.write(f"{epoch},{avg_loss:.4f},{avg_acc:.4f}\n")
        # Reset accumulators for next epoch
        self.trainer.epoch_loss_sum = 0.0
        self.trainer.epoch_total = 0
        self.trainer.epoch_correct = 0
        return control


# ---------------------------
# 6. Set up TrainingArguments
# ---------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=False,                   # evaluation is done separately
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,
    num_train_epochs=num_epochs,
    fp16=True,                       # use mixed precision if available
    logging_strategy="epoch",        # log metrics at epoch end
    save_strategy="epoch",
    logging_steps=1,
    save_total_limit=1,
    remove_unused_columns=False,     # Ensure custom input columns are preserved
)

# ---------------------------
# 7. Initialize Trainer and add callback
# ---------------------------
trainer = RetrievalTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.add_callback(EpochMetricsLogger(output_dir))

# ---------------------------
# 8. Train the model and save
# ---------------------------
train_result = trainer.train()
trainer.save_model(output_dir)
