import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CausalSelfAttention(nn.Module):
    """Multi-Head causal self-attention with 12 query heads and 12 key-value heads."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = 12
        self.num_kv_heads = 12
        self.hidden_dim = 768
        self.head_dim = 64
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_dim, 768, bias=True)
        self.v_proj = nn.Linear(self.hidden_dim, 768, bias=True)
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
    """Feed-forward network with geglu activation."""

    def __init__(self, config):
        super().__init__()
        self.intermediate_size = 3072
        self.hidden_dim = 768
        self.activation_fn = 'geglu'
        self.use_gated = True

        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_dim, bias=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = F.gelu(gate) * up
        x = self.dropout(x)
        x = self.down_proj(x)
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
        self.norm_placement = 'post'

    def forward(self, x, mask=None, position_ids=None):
        # Self-attention with residual (post-norm)
        residual = x
        x = self.attention(x, mask, position_ids)
        x = self.residual_dropout(x)
        x = x + residual
        x = self.ln1(x)

        # Feed-forward with residual (post-norm)
        residual = x
        x = self.ffn(x)
        x = self.residual_dropout(x)
        x = x + residual
        x = self.ln2(x)

        return x


class PostNormModel(nn.Module):
    """Decoder-only transformer model: PostNormModel"""

    def __init__(self, config=None):
        super().__init__()
        self.vocab_size = 50257
        self.max_position_embeddings = 1024
        self.hidden_dim = 768
        self.num_layers = 12

        # Embeddings
        self.token_embedding = nn.Embedding(
            50257,
            768,
            padding_idx=None,
        )
        self.position_embedding = nn.Embedding(
            1024,
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
        # Xavier uniform initialization (gain=1.0)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)