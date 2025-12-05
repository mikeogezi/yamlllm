import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class TinyModel(nn.Module):
    """TinyModel: Decoder-only transformer model."""
    vocab_size: int = 53
    hidden_dim: int = 256
    num_layers: int = 2

    @nn.compact
    def __call__(self, input_ids, deterministic: bool = True):
        batch_size, seq_len = input_ids.shape

        # Token embedding
        x = nn.Embed(
            num_embeddings=53,
            features=256,
            name='token_embedding'
        )(input_ids)

        # Position embedding
        position_ids = jnp.arange(seq_len)[None, :]
        pos_emb = nn.Embed(
            num_embeddings=128,
            features=256,
            name='position_embedding'
        )(position_ids)
        x = x + pos_emb

        # Decoder layers
        # Layer 0 - Attention
        residual = x
        x = nn.LayerNorm(name='ln1_0')(x)
        q = nn.Dense(256, name='q_proj_0')(x)
        k = nn.Dense(256, name='k_proj_0')(x)
        v = nn.Dense(256, name='v_proj_0')(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, 4, 64).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, 4, 64).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, 4, 64).transpose(0, 2, 1, 3)

        # Standard attention
        scale = 64 ** -0.5
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]
        scores = jnp.where(causal_mask, scores, -1e10)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = attn_weights @ v

        # Reshape and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, 256)
        x = nn.Dense(256, name='out_proj_0')(attn_output)

        x = residual + x

        # Layer 0 - FFN
        residual = x
        x = nn.LayerNorm(name='ln2_0')(x)
        x = nn.Dense(512, name='fc1_0')(x)
        x = jax.nn.relu(x)
        x = nn.Dense(256, name='fc2_0')(x)

        x = residual + x

        # Layer 1 - Attention
        residual = x
        x = nn.LayerNorm(name='ln1_1')(x)
        q = nn.Dense(256, name='q_proj_1')(x)
        k = nn.Dense(256, name='k_proj_1')(x)
        v = nn.Dense(256, name='v_proj_1')(x)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, 4, 64).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, 4, 64).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, 4, 64).transpose(0, 2, 1, 3)

        # Standard attention
        scale = 64 ** -0.5
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]
        scores = jnp.where(causal_mask, scores, -1e10)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        attn_output = attn_weights @ v

        # Reshape and project
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, 256)
        x = nn.Dense(256, name='out_proj_1')(attn_output)

        x = residual + x

        # Layer 1 - FFN
        residual = x
        x = nn.LayerNorm(name='ln2_1')(x)
        x = nn.Dense(512, name='fc1_1')(x)
        x = jax.nn.relu(x)
        x = nn.Dense(256, name='fc2_1')(x)

        x = residual + x

        # LM head
        logits = nn.Dense(53, use_bias=False, name='lm_head')(x)

        return logits