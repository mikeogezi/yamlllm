import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with 4 heads."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = 4
        self.hidden_dim = 256
        self.head_dim = 64
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

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
    """Feed-forward network with relu activation."""

    def __init__(self, config):
        super().__init__()
        self.intermediate_size = 512
        self.hidden_dim = 256
        self.activation_fn = 'relu'

        self.fc1 = nn.Linear(self.hidden_dim, self.intermediate_size, bias=False)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_dim, bias=False)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        x = self.fc1(x)
        if self.activation_fn == 'gelu':
            x = F.gelu(x)
        elif self.activation_fn == 'relu':
            x = F.relu(x)
        elif self.activation_fn == 'silu':
            x = F.silu(x)
        elif self.activation_fn == 'swiglu':
            # SwiGLU: Swish(xW + b) * (xV + c)
            # For simplicity, using single gate here
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
        self.ln1 = nn.LayerNorm(256, eps=1e-6, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(256, eps=1e-6, elementwise_affine=True)
        self.residual_dropout = nn.Dropout(0.0)

    def forward(self, x, mask=None):
        # Self-attention with residual
        residual = x
        x = self.ln1(x)
        x = self.attention(x, mask)
        x = self.residual_dropout(x)
        x = x + residual

        # Feed-forward with residual
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.residual_dropout(x)
        x = x + residual

        return x


class TinyModel(nn.Module):
    """Decoder-only transformer model: TinyModel"""

    def __init__(self, config=None):
        super().__init__()
        self.vocab_size = 1000
        self.max_position_embeddings = 128
        self.hidden_dim = 256
        self.num_layers = 2

        # Embeddings
        self.token_embedding = nn.Embedding(
            1000,
            256,
            padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            128,
            256
        )
        self.embedding_dropout = nn.Dropout(0.0)

        # Decoder layers
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(2)])


        # Output head
        self.lm_head = nn.Linear(256, 1000, bias=False)

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
        logits = self.lm_head(x)

        return logits