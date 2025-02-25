import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer
from model import TinyBERTWithMoE
from data_prep import load_beir_dataset, generate_triplets
from utils import infonce_loss
from datasets import Dataset
from functools import partial

# ---------------------------
# Configuration
# ---------------------------
model_name = "huawei-noah/TinyBERT_General_6L_768D"
dataset_name = "scifact"          # example BEIR dataset
data_dir = "datasets"             # directory containing the dataset
num_experts = 4                   # number of experts in MoE layer
max_length = 256                  # max sequence length for tokenizer (adjust as needed)
batch_size = 64                   # per-device batch size (optimize for 48GB GPU)
learning_rate = 5e-5
num_epochs = 20
output_dir = "./retrieval_model"

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
    # Tokenize query, positive, and negative separately
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
def custom_collate_fn(batch: list, tokenizer: AutoTokenizer):
    """
    Custom collate function to pad query, positive, and negative inputs separately.
    """
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
    
    # Pad each part separately
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

data_collator = partial(custom_collate_fn, tokenizer=tokenizer)

# ---------------------------
# 5. Define custom Trainer with Triplet Loss
# ---------------------------
class RetrievalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract inputs for queries, positives, negatives
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
        # Compute similarity matrix: each query vs. each document embedding (2B candidates)
        scores = torch.matmul(q_emb, doc_emb.T)  # [B, 2B]
        labels = torch.arange(batch_size).to(scores.device)  # correct doc index for each query
        
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(scores, labels)
        return (loss, (q_emb, doc_emb)) if return_outputs else loss

# ---------------------------
# 6. Set up TrainingArguments (with DDP and FP16 enabled)
# ---------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=False,                   # we will evaluate separately
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,
    num_train_epochs=num_epochs,
    fp16=True,                      # use mixed precision
    eval_strategy="no",       # disable built-in evaluation
    save_strategy="epoch",
    logging_steps=100,
    save_total_limit=1,
    remove_unused_columns=False,    # ensure all custom keys are passed
)

# ---------------------------
# 7. Initialize Trainer
# ---------------------------
trainer = RetrievalTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ---------------------------
# 8. Train the model
# ---------------------------
trainer.train()

# ---------------------------
# 9. Save the final model
# ---------------------------
trainer.save_model(output_dir)
