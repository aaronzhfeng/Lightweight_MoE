import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, PreTrainedModel

class MoELayer(nn.Module):
    """
    Single-Block Mixture-of-Experts (MoE) layer with top-1 gating.
    It consists of multiple feed-forward expert networks and a gating network 
    that routes each token's representation to one expert.
    """
    def __init__(self, input_dim: int, num_experts: int = 2, hidden_dim: int = None, dropout_prob: float = 0.1):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        # If hidden_dim not provided, use a smaller intermediate size (e.g., 2x input) to keep experts lightweight
        if hidden_dim is None:
            hidden_dim = input_dim * 2  # e.g., 1536 for 768 input, instead of 3072 to keep it lightweight
        # Define expert feed-forward networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),              # Activation (using GELU as in BERT)
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, input_dim),
            ) 
            for _ in range(num_experts)
        ])
        # Gating network: projects input to num_experts logits
        self.gate = nn.Linear(input_dim, num_experts)
        # Optionally, a softmax temperature could be tuned; here we use default softmax.
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of MoE layer.
        x: tensor of shape [batch_size, seq_len, input_dim]
        Returns a tensor of the same shape after expert transformation.
        """
        # Compute gating scores for each token (logits for each expert)
        gating_logits = self.gate(x)  # shape: [batch, seq_len, num_experts]
        # Softmax to get gating probabilities (across experts)
        gating_probs = F.softmax(gating_logits, dim=-1)  # same shape as logits
        # Select top-1 expert index for each token (sparse routing)
        # indices: [batch, seq_len]
        _, top1_indices = torch.max(gating_probs, dim=-1)
        
        # Prepare an output tensor
        batch_size, seq_len, _ = x.shape
        output = torch.zeros_like(x)  # [batch, seq_len, input_dim]
        
        # Process tokens for each expert
        # We will gather all tokens assigned to each expert and apply that expert's FFN at once.
        for expert_idx, expert_net in enumerate(self.experts):
            # Create a mask for tokens assigned to this expert
            mask = (top1_indices == expert_idx)  # boolean mask [batch, seq_len]
            if mask.any():
                # Get all tokens (vectors) that should go through this expert
                # Reshape to [N, input_dim] for N tokens assigned to this expert
                expert_input = x[mask]  # mask will flatten batch and seq dims where True
                # Apply expert feed-forward network
                expert_output = expert_net(expert_input)  # shape [N, input_dim]
                # Multiply by the gating probability for this expert (to weight the expert's contribution)
                # Note: For top-1 routing, most tokens will have a near-1 probability for the selected expert.
                expert_gate_probs = gating_probs[mask][:, expert_idx].unsqueeze(-1)  # shape [N, 1]
                expert_output = expert_output * expert_gate_probs  # weight output by gate probability
                # Place the expert's output back to the corresponding token positions
                output[mask] = expert_output
        return output

class TinyBERTWithMoE(PreTrainedModel):
    """
    TinyBERT model with an added Single-Block MoE layer at the final encoder output.
    This model produces embeddings suitable for dense retrieval.
    """
    def __init__(self, tinybert_name: str = "huawei-noah/TinyBERT_General_6L_768D", num_experts: int = 2):
        # Load TinyBERT config and base model
        config = AutoConfig.from_pretrained(tinybert_name)
        super().__init__(config)
        self.bert = AutoModel.from_pretrained(tinybert_name)  # Base TinyBERT (6 layers)
        hidden_size = config.hidden_size  # typically 768 for TinyBERT 6L model
        # Add MoE layer after TinyBERT's final layer output
        self.moe = MoELayer(input_dim=hidden_size, num_experts=num_experts)
        # Note: MoE layer is randomly initialized. We will fine-tune all parameters (TinyBERT + MoE).
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Get BERT outputs. We want the sequence output (hidden states for each token).
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  # shape [batch, seq_len, hidden_size]
        # Apply MoE layer on the final sequence outputs
        moe_output = self.moe(sequence_output)  # shape [batch, seq_len, hidden_size]
        # For retrieval, we typically use a single vector per sequence (e.g., [CLS] token embedding)
        # Here we take the [CLS] embedding (first token) after MoE transformation as the representation.
        cls_output = moe_output[:, 0, :]  # [batch, hidden_size]
        # (Optionally, you could normalize the representation if using cosine similarity)
        return cls_output
