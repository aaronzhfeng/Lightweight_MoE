import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, PreTrainedModel

class MoELayer(nn.Module):
    """
    Single-Block Mixture-of-Experts (MoE) layer with top-k gating.
    Uses multiple feed-forward expert networks and a gating network that routes each token's
    representation to the top-k experts. It computes an auxiliary load-balancing loss.
    """
    def __init__(self, input_dim: int, num_experts: int = 2, hidden_dim: int = None, 
                 dropout_prob: float = 0.1, top_k: int = 2, temperature: float = 5.0,
                 output_dropout_prob: float = 0.1):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.input_dim = input_dim
        if hidden_dim is None:
            hidden_dim = input_dim * 2  # e.g. 1536 for 768 input
        # Define expert feed-forward networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, input_dim),
            )
            for _ in range(num_experts)
        ])
        # Gating network: projects input to num_experts logits
        self.gate = nn.Linear(input_dim, num_experts)
        # Dropout applied to the MoE output (before residual addition)
        self.output_dropout = nn.Dropout(output_dropout_prob)
        # To store auxiliary load-balancing loss (computed per forward pass)
        self.aux_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape [batch_size, seq_len, input_dim]
        Returns a tensor of the same shape after processing through the MoE layer.
        """
        # Compute gating logits and apply temperature-scaled softmax.
        gating_logits = self.gate(x)  # [B, L, num_experts]
        gating_probs = F.softmax(gating_logits / self.temperature, dim=-1)  # [B, L, num_experts]

        # Compute auxiliary load-balancing loss.
        avg_gate = gating_probs.mean(dim=(0, 1))  # shape: [num_experts]
        uniform_prob = 1.0 / self.num_experts
        self.aux_loss = ((avg_gate - uniform_prob) ** 2).mean()

        # Top-k routing: for each token, keep only the top_k gating probabilities.
        topk_values, topk_indices = torch.topk(gating_probs, k=self.top_k, dim=-1)
        gating_mask = torch.zeros_like(gating_probs)
        gating_mask.scatter_(-1, topk_indices, topk_values)
        # Renormalize so that probabilities sum to 1 for each token.
        gating_mask = gating_mask / (gating_mask.sum(dim=-1, keepdim=True) + 1e-9)
        effective_gating = gating_mask  # [B, L, num_experts]

        batch_size, seq_len, _ = x.shape
        output = torch.zeros_like(x)

        # Process each expert separately.
        for expert_idx, expert_net in enumerate(self.experts):
            # Mask for tokens assigned (nonzero gating) to this expert.
            expert_mask = effective_gating[..., expert_idx] > 0  # shape: [B, L]
            if expert_mask.any():
                expert_input = x[expert_mask]  # [N, input_dim]
                expert_output = expert_net(expert_input)  # [N, input_dim]
                # Weight the output by the corresponding gating probabilities.
                gate_probs = effective_gating[..., expert_idx][expert_mask].unsqueeze(-1)
                weighted_output = expert_output * gate_probs
                # Accumulate contributions from this expert.
                output[expert_mask] += weighted_output

        # Apply dropout on the combined MoE output.
        output = self.output_dropout(output)
        return output

class TinyBERTWithMoE(PreTrainedModel):
    """
    TinyBERT model with an added Single-Block MoE layer at the final encoder output.
    Applies layer normalization on the encoder output and uses a residual connection.
    """
    def __init__(self, tinybert_name: str = "huawei-noah/TinyBERT_General_6L_768D", 
                 num_experts: int = 2, top_k: int = 2, temperature: float = 5.0,
                 output_dropout_prob: float = 0.1):
        config = AutoConfig.from_pretrained(tinybert_name)
        super().__init__(config)
        self.bert = AutoModel.from_pretrained(tinybert_name)  # Base TinyBERT (6 layers)
        hidden_size = config.hidden_size  # typically 768 for TinyBERT 6L
        # Apply LayerNorm to stabilize inputs to MoE.
        self.moe_layernorm = nn.LayerNorm(hidden_size)
        # MoE layer with new features.
        self.moe = MoELayer(input_dim=hidden_size, num_experts=num_experts, 
                            top_k=top_k, temperature=temperature,
                            output_dropout_prob=output_dropout_prob)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, 
                            token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  # [B, L, hidden_size]
        # Normalize before feeding into MoE.
        normalized = self.moe_layernorm(sequence_output)
        moe_output = self.moe(normalized)
        # Residual connection: add original sequence output.
        combined = sequence_output + moe_output
        # Save auxiliary loss for training.
        self.aux_loss = self.moe.aux_loss
        # For retrieval, return the [CLS] token representation.
        cls_output = combined[:, 0, :]
        return cls_output

