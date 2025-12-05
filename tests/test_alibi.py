import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with 12 heads."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = 12
        self.hidden_dim = 768
        self.head_dim = 64
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.dropout = nn.Dropout(0.1)
        # ALiBi slopes
        alibi_slopes = torch.tensor([0.6299605249, 0.3968502630, 0.2500000000, 0.1574901312, 0.0992125657, 0.0625000000, 0.0393725328, 0.0248031414, 0.0156250000, 0.0098431332, 0.0062007854, 0.0039062500])
        self.register_buffer('alibi_slopes', alibi_slopes)

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

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply ALiBi biases
        alibi_bias = self.alibi_slopes.unsqueeze(0).unsqueeze(-1) * torch.arange(seq_len, device=x.device).float()
        alibi_bias = alibi_bias.unsqueeze(0).unsqueeze(-1) - alibi_bias.unsqueeze(0).unsqueeze(-2)
        scores = scores + alibi_bias.unsqueeze(0)

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


class ALiBiModel(nn.Module):
    """Decoder-only transformer model: ALiBiModel"""

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
        self.embedding_dropout = nn.Dropout(0.1)

        # Decoder layers
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(12)])

        # Final layer norm
        self.final_ln = nn.LayerNorm(768, eps=1e-5, elementwise_affine=True)

        # Output head
        self.lm_head = None  # Will use token_embedding weights

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
        if self.lm_head is None:
            logits = torch.matmul(x, self.token_embedding.weight.t())
        else:
            logits = self.lm_head(x)

        return logits