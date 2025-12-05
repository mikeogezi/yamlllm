import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return x / (norm + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""

    def __init__(self, dim, max_position_embeddings=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, position_ids):
        # x: [batch_size, num_heads, seq_len, head_dim]
        inv_freq_expanded = self.inv_freq[None, :, None].float()
        position_ids_expanded = position_ids[:, None, :, None].float()
        freqs = (inv_freq_expanded @ position_ids_expanded.transpose(2, 3)).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin):
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with 32 heads."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = 32
        self.hidden_dim = 4096
        self.head_dim = 128
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(0.0)
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=2048,
            base=10000.0
        )

    def forward(self, x, mask=None, position_ids=None):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        cos, sin = self.rotary_emb(q, position_ids)
        q, k = self.rotary_emb.apply_rotary_pos_emb(q, k, cos, sin)

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
    """Feed-forward network with swiglu activation."""

    def __init__(self, config):
        super().__init__()
        self.intermediate_size = 11008
        self.hidden_dim = 4096
        self.activation_fn = 'swiglu'
        self.use_gated = True

        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = F.silu(gate) * up
        x = self.dropout(x)
        x = self.down_proj(x)
        return x

class DecoderLayer(nn.Module):
    """Single decoder transformer layer."""

    def __init__(self, config):
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = RMSNorm(4096, eps=1e-6)
        self.ln2 = RMSNorm(4096, eps=1e-6)
        self.residual_dropout = nn.Dropout(0.0)

    def forward(self, x, mask=None, position_ids=None):
        # Self-attention with residual
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask, position_ids)
        x = self.residual_dropout(x)
        x = x + residual

        # Feed-forward with residual
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.residual_dropout(x)
        x = x + residual

        return x


class LLaMAStyle(nn.Module):
    """Decoder-only transformer model: LLaMAStyle"""

    def __init__(self, config=None):
        super().__init__()
        self.vocab_size = 32000
        self.max_position_embeddings = 2048
        self.hidden_dim = 4096
        self.num_layers = 32

        # Embeddings
        self.token_embedding = nn.Embedding(
            32000,
            4096,
            padding_idx=None,
        )
        self.embedding_dropout = nn.Dropout(0.0)

        # Decoder layers
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(32)])

        # Final layer norm
        self.final_ln = RMSNorm(4096, eps=1e-6)

        # Output head
        self.lm_head = nn.Linear(4096, 32000, bias=False)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        # Create position ids
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        x = token_embeds
        x = self.embedding_dropout(x)

        # Decoder layers
        for layer in self.layers:
            x = layer(x, mask=attention_mask, position_ids=position_ids)

        # Final layer norm
        x = self.final_ln(x)

        # Language modeling head
        logits = self.lm_head(x)

        return logits