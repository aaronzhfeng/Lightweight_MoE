import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi
import safetensors.torch

from data_prep import load_beir_dataset
from model import TinyBERTMoEStudent, TinyBERTMoERegular  # student and teacher classes

# ---------------------------
# BM25 Hard Negative Sampling
# ---------------------------
def generate_hard_triplets(corpus, queries, qrels, num_negatives=1):
    triplets = []
    doc_ids = list(corpus.keys())
    doc_texts = []
    for doc_id in doc_ids:
        text = corpus[doc_id].get("text", "")
        title = corpus[doc_id].get("title")
        if title:
            text = title + " " + text
        doc_texts.append(text)
    bm25 = BM25Okapi([doc.split() for doc in doc_texts])
    
    for qid, query_text in queries.items():
        if qid not in qrels or not qrels[qid]:
            continue
        pos_ids = [doc_id for doc_id, score in qrels[qid].items() if score > 0]
        for pos_id in pos_ids:
            pos_text = corpus[pos_id].get("text", "")
            title = corpus[pos_id].get("title")
            if title:
                pos_text = title + " " + pos_text
            tokenized_query = query_text.split()
            bm25_scores = bm25.get_scores(tokenized_query)
            sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
            hard_negs = [doc_ids[i] for i in sorted_indices if doc_ids[i] not in pos_ids]
            if not hard_negs:
                continue
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
teacher_checkpoint = "./retrieval_model_regular"  # pre-trained teacher checkpoint (safetensors)
student_output_dir = "./retrieval_model_distilled"
model_name = "huawei-noah/TinyBERT_General_6L_768D"
dataset_name = "scifact"
data_dir = "datasets"
max_length = 256
batch_size = 64
learning_rate = 5e-5
num_epochs = 20
distill_temperature = 2.0  # distillation temperature
kd_weight = 0.5  # weight for the distillation (KL) loss

os.makedirs(student_output_dir, exist_ok=True)

# ---------------------------
# Data Preparation
# ---------------------------
corpus, queries, qrels = load_beir_dataset(data_dir, dataset_name)
train_triplets = generate_hard_triplets(corpus, queries, qrels, num_negatives=1)
train_dataset = Dataset.from_dict({
    "query": [t[0] for t in train_triplets],
    "pos": [t[1] for t in train_triplets],
    "neg": [t[2] for t in train_triplets],
})

# ---------------------------
# Initialize Tokenizer and Student Model
# ---------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
student_model = TinyBERTMoEStudent(
    tinybert_name=model_name, 
    num_experts=4, 
    top_k=2, 
    temperature=5.0,
    expert_hidden_multiplier=0.8
)

# ---------------------------
# Tokenization Function
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
# Custom Data Collator
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
    padded_pos = tokenizer.pad(pos_inputs, return_tensors="pt")
    padded_neg = tokenizer.pad(neg_inputs, return_tensors="pt")
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
# Knowledge Distillation Loss Function
# ---------------------------
def kd_loss(teacher_logits, student_logits, T):
    teacher_prob = F.log_softmax(teacher_logits / T, dim=-1)
    student_prob = F.log_softmax(student_logits / T, dim=-1)
    return F.kl_div(student_prob, teacher_prob, reduction="batchmean") * (T * T)

# ---------------------------
# Custom Trainer for Distillation
# ---------------------------
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, kd_weight, kd_temperature, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.teacher_model.eval()  # Freeze teacher
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        # Ensure teacher model is on the same device as training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_model.to(device)
        self.kd_weight = kd_weight
        self.kd_temperature = kd_temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        # Ensure inputs are on the correct device
        device = model.device
        q_input_ids = inputs["query_input_ids"].to(device)
        q_att_mask = inputs["query_attention_mask"].to(device)
        p_input_ids = inputs["pos_input_ids"].to(device)
        p_att_mask = inputs["pos_attention_mask"].to(device)
        n_input_ids = inputs["neg_input_ids"].to(device)
        n_att_mask = inputs["neg_attention_mask"].to(device)
        
        doc_input_ids = torch.cat([p_input_ids, n_input_ids], dim=0)
        doc_att_mask = torch.cat([p_att_mask, n_att_mask], dim=0)
        
        # Student forward pass
        student_q = model(input_ids=q_input_ids, attention_mask=q_att_mask)
        student_doc = model(input_ids=doc_input_ids, attention_mask=doc_att_mask)
        batch_size = student_q.size(0)
        student_scores = torch.matmul(student_q, student_doc.T)
        labels = torch.arange(batch_size).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        student_loss = loss_fn(student_scores, labels)
        
        # Teacher forward pass (with no grad)
        with torch.no_grad():
            teacher_q = self.teacher_model(input_ids=q_input_ids, attention_mask=q_att_mask)
            teacher_doc = self.teacher_model(input_ids=doc_input_ids, attention_mask=doc_att_mask)
            teacher_scores = torch.matmul(teacher_q, teacher_doc.T)
        
        # Debug prints (uncomment if needed)
        # if torch.isnan(student_scores).any():
        #     print("Student scores contain NaNs!")
        # if torch.isnan(teacher_scores).any():
        #     print("Teacher scores contain NaNs!")
        # print("Teacher scores mean:", teacher_scores.mean().item(), "Student scores mean:", student_scores.mean().item())
        
        loss_kd = kd_loss(teacher_scores, student_scores, self.kd_temperature)
        total_loss = student_loss + self.kd_weight * loss_kd
        
        preds = torch.argmax(student_scores, dim=1)
        accuracy = (preds == labels).float().mean()
        
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
        # Optionally, clip gradients if needed:
        # torch.nn.utils.clip_grad_norm_(model.some_submodule.parameters(), max_norm=1.0)
        return loss.detach()

# ---------------------------
# TrainingArguments and Trainer Initialization
# ---------------------------
training_args = TrainingArguments(
    output_dir=student_output_dir,
    overwrite_output_dir=True,
    do_train=True,
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
# Load Teacher Model from Checkpoint using safetensors
# ---------------------------
teacher_model = TinyBERTMoERegular(
    tinybert_name=model_name, 
    num_experts=8, 
    top_k=2, 
    temperature=5.0
)
safetensors_path = os.path.join(teacher_checkpoint, "model.safetensors")
state_dict = safetensors.torch.load_file(safetensors_path, device="cpu")
teacher_model.load_state_dict(state_dict, strict=True)
teacher_model.eval()

trainer = DistillationTrainer(
    teacher_model=teacher_model,
    kd_weight=kd_weight,
    kd_temperature=distill_temperature,
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ---------------------------
# (Optional) Metrics Logger Callback
# ---------------------------
class EpochMetricsLogger(TrainerCallback):
    def __init__(self, output_dir):
        self.filepath = os.path.join(output_dir, "training_metrics.csv")
        with open(self.filepath, "w") as f:
            f.write("epoch,train_loss,train_accuracy\n")
        self.trainer = None
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
        self.trainer.epoch_loss_sum = 0.0
        self.trainer.epoch_total = 0
        self.trainer.epoch_correct = 0
        return control

trainer.add_callback(EpochMetricsLogger(student_output_dir))

# ---------------------------
# Train and Save Student Model
# ---------------------------
train_result = trainer.train()
trainer.save_model(student_output_dir)
print(f"Distilled MoE TinyBERT model saved to: {student_output_dir}")
