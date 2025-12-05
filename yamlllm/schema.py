"""YAML schema definition and validation for decoder-only transformer architectures."""

from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class AttentionConfig:
    """Multi-head self-attention configuration."""
    num_heads: int
    head_dim: Optional[int] = None  # If None, computed as hidden_dim // num_heads
    dropout: float = 0.1
    bias: bool = True
    use_flash_attention: bool = False
    mechanism: Literal["standard", "alibi", "mla"] = "standard"
    # ALiBi-specific parameters
    alibi_max_positions: Optional[int] = None  # Max positions for ALiBi slopes
    # GQA/MQA: number of key-value heads (if None, uses num_heads for standard MHA)
    num_kv_heads: Optional[int] = None  # For GQA/MQA: if 1, MQA; if < num_heads, GQA
    # MLA-specific parameters
    mla_latent_dim: Optional[int] = None  # Latent dimension for MLA (default: hidden_dim // 4)
    mla_rank: Optional[int] = None  # Rank for MLA compression (default: 8)


@dataclass
class FFNConfig:
    """Feed-forward network configuration."""
    intermediate_size: int
    activation: Literal["gelu", "relu", "silu", "swiglu", "geglu", "reglu"] = "gelu"
    dropout: float = 0.1
    bias: bool = True
    # For SwiGLU: intermediate_size is the size after gating (typically 2/3 of total)
    use_gated_activation: bool = False  # True for SwiGLU/GeGLU/ReGLU-style gating


@dataclass
class PositionalEncodingConfig:
    """Positional encoding configuration."""
    type: Literal["learned", "rope", "alibi"] = "learned"
    max_position_embeddings: int = 1024
    # RoPE-specific parameters
    rope_theta: float = 10000.0  # Base frequency for RoPE
    rope_scaling: Optional[Literal["linear", "dynamic"]] = None
    rope_scaling_factor: Optional[float] = None
    # ALiBi is configured in attention config


@dataclass
class EmbeddingConfig:
    """Token and positional embedding configuration."""
    vocab_size: int
    max_position_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int] = None
    dropout: float = 0.1
    positional_encoding: Optional[PositionalEncodingConfig] = None


@dataclass
class LayerNormConfig:
    """Layer normalization configuration."""
    type: Literal["layernorm", "rmsnorm"] = "layernorm"
    eps: float = 1e-5
    elementwise_affine: bool = True


@dataclass
class DecoderLayerConfig:
    """Single decoder layer configuration."""
    hidden_dim: int
    attention: AttentionConfig
    ffn: FFNConfig
    layer_norm: LayerNormConfig
    residual_dropout: float = 0.1
    # Architecture variant: pre-norm (norm before attention/ffn) or post-norm (norm after)
    norm_placement: Literal["pre", "post"] = "pre"  # pre-norm is more common in modern models


@dataclass
class InitializationConfig:
    """Weight initialization configuration."""
    type: Literal["default", "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal", "gpt2"] = "default"
    gain: float = 1.0  # Gain factor for initialization
    # GPT-2 style: std = 0.02 for embeddings, std = residual / sqrt(2 * num_layers) for others
    gpt2_residual_scale: Optional[float] = None  # Override residual scaling for GPT-2 init


@dataclass
class ModelConfig:
    """Complete decoder-only transformer model configuration."""
    name: str
    embedding: EmbeddingConfig
    num_layers: int
    layer: DecoderLayerConfig
    final_layer_norm: Optional[LayerNormConfig] = None
    tie_word_embeddings: bool = True  # Tie input and output embeddings
    initialization: Optional[InitializationConfig] = None


def validate_config(config: ModelConfig) -> list[str]:
    """Validate model configuration and return list of errors (empty if valid)."""
    errors = []
    
    # Check embedding dimension matches hidden dimension
    if config.embedding.embedding_dim != config.layer.hidden_dim:
        errors.append(
            f"Embedding dimension ({config.embedding.embedding_dim}) must match "
            f"hidden dimension ({config.layer.hidden_dim})"
        )
    
    # Check attention head dimension
    if config.layer.attention.head_dim is not None:
        if config.layer.hidden_dim % config.layer.attention.head_dim != 0:
            errors.append(
                f"Hidden dimension ({config.layer.hidden_dim}) must be divisible "
                f"by attention head_dim ({config.layer.attention.head_dim})"
            )
    else:
        if config.layer.hidden_dim % config.layer.attention.num_heads != 0:
            errors.append(
                f"Hidden dimension ({config.layer.hidden_dim}) must be divisible "
                f"by number of attention heads ({config.layer.attention.num_heads})"
            )
    
    # Check FFN intermediate size is reasonable
    if config.layer.ffn.intermediate_size < config.layer.hidden_dim:
        errors.append(
            f"FFN intermediate_size ({config.layer.ffn.intermediate_size}) should "
            f"typically be >= hidden_dim ({config.layer.hidden_dim})"
        )
    
    # Validate positional encoding compatibility
    pos_enc = config.embedding.positional_encoding
    if pos_enc:
        if pos_enc.type == "alibi" and config.layer.attention.mechanism != "alibi":
            errors.append(
                "ALiBi positional encoding requires attention mechanism to be 'alibi'"
            )
        if pos_enc.type == "rope" and config.layer.attention.head_dim is None:
            # RoPE works best with explicit head_dim
            pass  # Not an error, just a note
    
    # Validate attention mechanism
    if config.layer.attention.mechanism == "alibi":
        if pos_enc and pos_enc.type != "alibi":
            errors.append(
                "ALiBi attention mechanism requires ALiBi positional encoding"
            )
    
    # Validate MLA
    if config.layer.attention.mechanism == "mla":
        if config.layer.attention.mla_latent_dim is not None:
            if config.layer.attention.mla_latent_dim > config.layer.hidden_dim:
                errors.append("MLA latent_dim cannot exceed hidden_dim")
        if config.layer.attention.mla_rank is not None:
            if config.layer.attention.mla_rank < 1:
                errors.append("MLA rank must be >= 1")
    
    # Validate GQA/MQA
    if config.layer.attention.num_kv_heads is not None:
        if config.layer.attention.num_kv_heads < 1:
            errors.append("num_kv_heads must be >= 1")
        if config.layer.attention.num_kv_heads > config.layer.attention.num_heads:
            errors.append("num_kv_heads cannot exceed num_heads")
        if config.layer.attention.num_heads % config.layer.attention.num_kv_heads != 0:
            errors.append("num_heads must be divisible by num_kv_heads for GQA")
    
    # Validate gated activations
    if config.layer.ffn.activation in ["swiglu", "geglu", "reglu"]:
        if not config.layer.ffn.use_gated_activation:
            errors.append(
                f"{config.layer.ffn.activation} requires use_gated_activation: true"
            )
    
    return errors

