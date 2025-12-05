import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CausalSelfAttention(nn.Module):
    """Grouped Query causal self-attention with 12 query heads and 4 key-value heads."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = 12
        self.num_kv_heads = 4
        self.hidden_dim = 768
        self.head_dim = 64
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_dim, 256, bias=True)
        self.v_proj = nn.Linear(self.hidden_dim, 256, bias=True)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None, position_ids=None):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat K and V for GQA/MQA
        if self.num_kv_heads < self.num_heads:
            repeat_kv = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_kv, dim=1)
            v = v.repeat_interleave(repeat_kv, dim=1)


        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        if mask is None:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            mask = causal_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)

        return output

class FeedForward(nn.Module):
    """Feed-forward network with gelu activation."""

    def __init__(self, config):
        super().__init__()
        self.intermediate_size = 3072
        self.hidden_dim = 768
        self.activation_fn = 'gelu'
        self.use_gated = False

        self.fc1 = nn.Linear(self.hidden_dim, self.intermediate_size, bias=True)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_dim, bias=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        if self.activation_fn == 'gelu':
            x = F.gelu(x)
        elif self.activation_fn == 'relu':
            x = F.relu(x)
        elif self.activation_fn == 'silu':
            x = F.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DecoderLayer(nn.Module):
    """Single decoder transformer layer."""

    def __init__(self, config):
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(768, eps=1e-5, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(768, eps=1e-5, elementwise_affine=True)
        self.residual_dropout = nn.Dropout(0.1)
        self.norm_placement = 'pre'

    def forward(self, x, mask=None, position_ids=None):
        # Self-attention with residual (pre-norm)
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask, position_ids)
        x = self.residual_dropout(x)
        x = x + residual

        # Feed-forward with residual (pre-norm)
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.residual_dropout(x)
        x = x + residual

        return x


class GQAModel(nn.Module):
    """Decoder-only transformer model: GQAModel"""

    def __init__(self, config=None):
        super().__init__()
        self.vocab_size = 50257
        self.max_position_embeddings = 2048
        self.hidden_dim = 768
        self.num_layers = 12

        # Embeddings
        self.token_embedding = nn.Embedding(
            50257,
            768,
            padding_idx=None,
        )
        self.position_embedding = nn.Embedding(
            2048,
            768
        )
        self.embedding_dropout = nn.Dropout(0.1)

        # Decoder layers
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(12)])

        # Output head
        self.lm_head = None  # Will use token_embedding weights

        self._init_weights()
        # Initialize weights

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        x = token_embeds + pos_embeds
        x = self.embedding_dropout(x)

        # Decoder layers
        for layer in self.layers:
            x = layer(x, mask=attention_mask)

        # Language modeling head
        if self.lm_head is None:
            logits = torch.matmul(x, self.token_embedding.weight.t())
        else:
            logits = self.lm_head(x)

        return logits

    def _init_weights(self):
        """Initialize model weights."""
        # GPT-2 style initialization
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.20412414523193154 * std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
            elif isinstance(module, nn.LayerNorm):
                if hasattr(module, 'weight') and module.weight is not None:
                    module.weight.data.fill_(1.0)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.zero_()
            # Handle RMSNorm if present
            elif hasattr(module, 'weight') and hasattr(module, 'eps'):
                # Likely RMSNorm
                if module.weight is not None:
                    module.weight.data.fill_(1.0)