import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional


class TinyModel(nn.Module):
    """TinyModel: Decoder-only transformer model."""
    vocab_size: int = 1000
    hidden_dim: int = 256
    num_layers: int = 2

    @nn.compact
    def __call__(self, input_ids, deterministic: bool = True):
        batch_size, seq_len = input_ids.shape

        # Token embedding
        x = nn.Embed(
            num_embeddings=1000,
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
        x = residual + x

        # Layer 0 - FFN
        residual = x
        x = nn.LayerNorm(name='ln2_0')(x)
        x = residual + x

        # Layer 1 - Attention
        residual = x
        x = nn.LayerNorm(name='ln1_1')(x)
        x = residual + x

        # Layer 1 - FFN
        residual = x
        x = nn.LayerNorm(name='ln2_1')(x)
        x = residual + x

        # LM head
        logits = nn.Dense(1000, use_bias=False, name='lm_head')(x)

        return logits