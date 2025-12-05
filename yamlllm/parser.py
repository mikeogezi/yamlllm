"""YAML parser for decoder-only transformer configurations."""

import yaml
from pathlib import Path
from typing import Dict, Any

from .schema import (
    ModelConfig,
    EmbeddingConfig,
    PositionalEncodingConfig,
    DecoderLayerConfig,
    AttentionConfig,
    FFNConfig,
    LayerNormConfig,
    InitializationConfig,
    validate_config,
)


def parse_yaml_config(yaml_path: str | Path) -> ModelConfig:
    """Parse a YAML file and return a validated ModelConfig."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    
    return parse_dict_config(data)


def parse_dict_config(data: Dict[str, Any]) -> ModelConfig:
    """Parse a dictionary and return a validated ModelConfig."""
    # Parse embedding config
    emb_data = data["embedding"]
    
    # Parse positional encoding if present
    pos_enc_data = emb_data.get("positional_encoding", {})
    positional_encoding = None
    if pos_enc_data:
        positional_encoding = PositionalEncodingConfig(
            type=pos_enc_data.get("type", "learned"),
            max_position_embeddings=pos_enc_data.get("max_position_embeddings", emb_data.get("max_position_embeddings", 1024)),
            rope_theta=pos_enc_data.get("rope_theta", 10000.0),
            rope_scaling=pos_enc_data.get("rope_scaling"),
            rope_scaling_factor=pos_enc_data.get("rope_scaling_factor"),
        )
    
    embedding = EmbeddingConfig(
        vocab_size=emb_data["vocab_size"],
        max_position_embeddings=emb_data["max_position_embeddings"],
        embedding_dim=emb_data["embedding_dim"],
        padding_idx=emb_data.get("padding_idx"),
        dropout=emb_data.get("dropout", 0.1),
        positional_encoding=positional_encoding,
    )
    
    # Parse layer config
    layer_data = data["layer"]
    
    # Parse attention
    attn_data = layer_data["attention"]
    attention = AttentionConfig(
        num_heads=attn_data["num_heads"],
        head_dim=attn_data.get("head_dim"),
        dropout=attn_data.get("dropout", 0.1),
        bias=attn_data.get("bias", True),
        use_flash_attention=attn_data.get("use_flash_attention", False),
        mechanism=attn_data.get("mechanism", "standard"),
        alibi_max_positions=attn_data.get("alibi_max_positions"),
        num_kv_heads=attn_data.get("num_kv_heads"),
        mla_latent_dim=attn_data.get("mla_latent_dim"),
        mla_rank=attn_data.get("mla_rank"),
    )
    
    # Parse FFN
    ffn_data = layer_data["ffn"]
    activation = ffn_data.get("activation", "gelu")
    use_gated = activation in ["swiglu", "geglu", "reglu"] or ffn_data.get("use_gated_activation", False)
    ffn = FFNConfig(
        intermediate_size=ffn_data["intermediate_size"],
        activation=activation,
        dropout=ffn_data.get("dropout", 0.1),
        bias=ffn_data.get("bias", True),
        use_gated_activation=use_gated,
    )
    
    # Parse layer norm
    ln_data = layer_data.get("layer_norm", {})
    layer_norm = LayerNormConfig(
        type=ln_data.get("type", "layernorm"),
        eps=ln_data.get("eps", 1e-5),
        elementwise_affine=ln_data.get("elementwise_affine", True),
    )
    
    # Parse decoder layer
    decoder_layer = DecoderLayerConfig(
        hidden_dim=layer_data["hidden_dim"],
        attention=attention,
        ffn=ffn,
        layer_norm=layer_norm,
        residual_dropout=layer_data.get("residual_dropout", 0.1),
        norm_placement=layer_data.get("norm_placement", "pre"),
    )
    
    # Parse final layer norm if present
    final_ln_data = data.get("final_layer_norm", {})
    final_layer_norm = None
    if final_ln_data:
        final_layer_norm = LayerNormConfig(
            type=final_ln_data.get("type", "layernorm"),
            eps=final_ln_data.get("eps", 1e-5),
            elementwise_affine=final_ln_data.get("elementwise_affine", True),
        )
    
    # Parse initialization if present
    init_data = data.get("initialization", {})
    initialization = None
    if init_data:
        initialization = InitializationConfig(
            type=init_data.get("type", "default"),
            gain=init_data.get("gain", 1.0),
            gpt2_residual_scale=init_data.get("gpt2_residual_scale"),
        )
    
    # Create model config
    config = ModelConfig(
        name=data["name"],
        embedding=embedding,
        num_layers=data["num_layers"],
        layer=decoder_layer,
        final_layer_norm=final_layer_norm,
        tie_word_embeddings=data.get("tie_word_embeddings", True),
        initialization=initialization,
    )
    
    # Validate
    errors = validate_config(config)
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return config

