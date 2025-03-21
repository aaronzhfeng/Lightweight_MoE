import os
import random
import torch
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset
from functools import partial
import matplotlib.pyplot as plt

from data_prep import load_beir_dataset, generate_triplets

# ---------------------------
# Configuration
# ---------------------------
model_name = "huawei-noah/TinyBERT_General_6L_768D"
dataset_name = "scifact"          # Example BEIR dataset
data_dir = "datasets"             # Directory containing the dataset
max_length = 256                  # Max sequence length for tokenizer
batch_size = 64                   # Per-device batch size
learning_rate = 5e-5
num_epochs = 20
output_dir = "./retrieval_model_tinybert"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# ---------------------------
# 1. Load and prepare dataset
# ---------------------------
corpus, queries, qrels = load_beir_dataset(data_dir, dataset_name)
train_triplets = generate_triplets(corpus, queries, qrels, num_negatives=1)

# Convert triplets to a Hugging Face Dataset
train_dataset = Dataset.from_dict({
    "query": [t[0] for t in train_triplets],
    "pos": [t[1] for t in train_triplets],
    "neg": [t[2] for t in train_triplets],
})

# ---------------------------
# 2. Load tokenizer and baseline TinyBERT model
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load the baseline TinyBERT without MoE
model = AutoModel.from_pretrained(model_name)

# ---------------------------
# 3. Tokenize function for triplets
# ---------------------------
def tokenize_triplets(example):
    # We'll tokenize each field separately
    q = tokenizer(example["query"], truncation=True, max_length=max_length)
    p = tokenizer(example["pos"], truncation=True, max_length=max_length)
    n = tokenizer(example["neg"], truncation=True, max_length=max_length)
    return {
        "query_input_ids": q["input_ids"],
        "query_attention_mask": q["attention_mask"],
        "pos_input_ids": p["input_ids"],
        "pos_attention_mask": p["attention_mask"],
        "neg_input_ids": n["input_ids"],
        "neg_attention_mask": n["attention_mask"],
    }

train_dataset = train_dataset.map(tokenize_triplets, batched=True, remove_columns=["query", "pos", "neg"])

# ---------------------------
# 4. Custom Data Collator
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
# 5. Custom Trainer with Triplet (InfoNCE) Loss
# ---------------------------
class RetrievalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        q_input_ids = inputs["query_input_ids"]
        q_att_mask = inputs["query_attention_mask"]
        p_input_ids = inputs["pos_input_ids"]
        p_att_mask = inputs["pos_attention_mask"]
        n_input_ids = inputs["neg_input_ids"]
        n_att_mask = inputs["neg_attention_mask"]
        
        # Concatenate positive and negative inputs along batch dimension
        doc_input_ids = torch.cat([p_input_ids, n_input_ids], dim=0)
        doc_att_mask = torch.cat([p_att_mask, n_att_mask], dim=0)
        
        # Forward pass for queries
        q_outputs = model(input_ids=q_input_ids, attention_mask=q_att_mask)
        # [batch_size, seq_len, hidden_dim]
        q_emb = q_outputs.last_hidden_state[:, 0, :]  # Take [CLS] embedding
        
        # Forward pass for docs (pos+neg)
        doc_outputs = model(input_ids=doc_input_ids, attention_mask=doc_att_mask)
        doc_emb = doc_outputs.last_hidden_state[:, 0, :]
        
        batch_size = q_emb.size(0)
        
        # Compute similarity matrix: [B, 2B]
        scores = torch.matmul(q_emb, doc_emb.T)
        
        # Labels: each query's positive doc is at index i
        labels = torch.arange(batch_size).to(scores.device)
        
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(scores, labels)
        
        # Compute accuracy (optional, for logging)
        preds = torch.argmax(scores, dim=1)
        accuracy = (preds == labels).float().mean()
        
        # Accumulate for epoch-level logging
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

    def training_step(self, model, inputs, num_items=None):
        """
        Override the default training_step to match huggingface's call signature,
        which includes an optional `num_items` argument in recent versions.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        self.accelerator.backward(loss)
        return loss.detach()

# ---------------------------
# Callback to log epoch metrics
# ---------------------------
class EpochMetricsLogger(TrainerCallback):
    def __init__(self, output_dir):
        self.filepath = os.path.join(output_dir, "training_metrics.csv")
        # Write header
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

        # Write to CSV
        with open(self.filepath, "a") as f:
            f.write(f"{epoch},{avg_loss:.4f},{avg_acc:.4f}\n")
        # Reset accumulators
        self.trainer.epoch_loss_sum = 0.0
        self.trainer.epoch_total = 0
        self.trainer.epoch_correct = 0
        return control


# ---------------------------
# 6. TrainingArguments
# ---------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=False,
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,
    num_train_epochs=num_epochs,
    fp16=True,                       # mixed precision
    logging_strategy="epoch",
    save_strategy="epoch",
    logging_steps=1,
    save_total_limit=1,
    remove_unused_columns=False,
)

# ---------------------------
# 7. Initialize Trainer
# ---------------------------
trainer = RetrievalTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,  # can still pass the tokenizer for some HF internals
    data_collator=data_collator,
)
trainer.add_callback(EpochMetricsLogger(output_dir))

# ---------------------------
# 8. Train and save
# ---------------------------
train_result = trainer.train()
trainer.save_model(output_dir)
print(f"Baseline TinyBERT fine-tuned model saved to: {output_dir}")
