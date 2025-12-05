# YamlLLM Advanced Features

This document describes the advanced features supported by YamlLLM for defining decoder-only transformer architectures.

## Positional Encodings

### Learned Embeddings (Default)
Standard learned positional embeddings added to token embeddings.

```yaml
embedding:
  positional_encoding:
    type: learned  # or omit for default
```

### RoPE (Rotary Positional Embeddings)
Rotary positional embeddings that encode relative position information directly in attention.

```yaml
embedding:
  positional_encoding:
    type: rope
    rope_theta: 10000.0  # Base frequency (default: 10000.0)
    max_position_embeddings: 2048
```

### ALiBi (Attention with Linear Biases)
Positional encoding via linear biases in attention scores. No learned embeddings needed.

```yaml
embedding:
  positional_encoding:
    type: alibi
layer:
  attention:
    mechanism: alibi
    alibi_max_positions: 2048  # Optional
```

## Attention Mechanisms

### Standard Attention (Default)
Standard scaled dot-product attention with causal masking.

```yaml
layer:
  attention:
    mechanism: standard  # or omit for default
```

### ALiBi Attention
Attention with linear biases for position encoding. Automatically computes per-head slopes.

```yaml
layer:
  attention:
    mechanism: alibi
```

### Multi-Latent Attention (MLA)
Compresses key-value cache into a low-rank latent space for memory efficiency. Reduces KV cache size significantly while maintaining performance.

```yaml
layer:
  attention:
    mechanism: mla
    mla_latent_dim: 192  # Latent dimension (typically hidden_dim / 4)
    mla_rank: 8  # Rank for compression (optional)
```

## Normalization

### LayerNorm (Default)
Standard layer normalization.

```yaml
layer:
  layer_norm:
    type: layernorm  # or omit for default
    eps: 1e-5
    elementwise_affine: true
```

### RMSNorm
Root Mean Square Layer Normalization (used in LLaMA, PaLM).

```yaml
layer:
  layer_norm:
    type: rmsnorm
    eps: 1e-6
```

## Feed-Forward Networks

### Standard Activations
- `gelu`: Gaussian Error Linear Unit (default)
- `relu`: Rectified Linear Unit
- `silu`: Sigmoid Linear Unit (Swish)

```yaml
layer:
  ffn:
    activation: gelu
    intermediate_size: 3072
```

### SwiGLU (Gated Activation)
SwiGLU uses a gated architecture: `SwiGLU(x) = Swish(xW_g + b_g) ⊙ (xW_u + b_u)`

```yaml
layer:
  ffn:
    activation: swiglu
    use_gated_activation: true  # Required for SwiGLU
    intermediate_size: 11008  # Size after gating
```

## Example Configurations

### LLaMA-Style Model
```yaml
name: LLaMAStyle
embedding:
  vocab_size: 32000
  max_position_embeddings: 2048
  embedding_dim: 4096
  positional_encoding:
    type: rope
    rope_theta: 10000.0
num_layers: 32
layer:
  hidden_dim: 4096
  attention:
    num_heads: 32
    head_dim: 128
    bias: false
  ffn:
    intermediate_size: 11008
    activation: swiglu
    use_gated_activation: true
    bias: false
  layer_norm:
    type: rmsnorm
    eps: 1e-6
final_layer_norm:
  type: rmsnorm
  eps: 1e-6
```

### ALiBi Model
```yaml
name: ALiBiModel
embedding:
  vocab_size: 50257
  max_position_embeddings: 2048
  embedding_dim: 768
  positional_encoding:
    type: alibi
num_layers: 12
layer:
  hidden_dim: 768
  attention:
    num_heads: 12
    mechanism: alibi
  ffn:
    intermediate_size: 3072
    activation: gelu
```

## Attention Variants

### Standard Multi-Head Attention (MHA)
Standard attention with equal number of query, key, and value heads.

```yaml
layer:
  attention:
    num_heads: 12
    # num_kv_heads defaults to num_heads for standard MHA
```

### Multi-Query Attention (MQA)
Single key-value head shared across all query heads. More memory efficient.

```yaml
layer:
  attention:
    num_heads: 12
    num_kv_heads: 1  # MQA: 1 key-value head
```

### Grouped Query Attention (GQA)
Multiple query heads share fewer key-value heads. Balance between MHA and MQA.

```yaml
layer:
  attention:
    num_heads: 12
    num_kv_heads: 4  # GQA: 12 query heads share 4 key-value heads
```

## Architecture Variants

### Pre-Norm (Default)
Layer normalization applied before attention and feed-forward. More stable training.

```yaml
layer:
  norm_placement: pre  # or omit for default
```

### Post-Norm
Layer normalization applied after attention and feed-forward. Original Transformer architecture.

```yaml
layer:
  norm_placement: post
```

## Weight Initialization

### Default
PyTorch default initialization.

### GPT-2 Style
```yaml
initialization:
  type: gpt2
  gpt2_residual_scale: null  # Auto-computed as 1/sqrt(2*num_layers)
```

### Xavier Uniform
```yaml
initialization:
  type: xavier_uniform
  gain: 1.0
```

### Xavier Normal
```yaml
initialization:
  type: xavier_normal
  gain: 1.0
```

### Kaiming (He) Initialization
```yaml
initialization:
  type: kaiming_uniform  # or kaiming_normal
```

## Additional Gated Activations

### GeGLU
Gated activation with GELU: `GeGLU(x) = GELU(xW_g + b_g) ⊙ (xW_u + b_u)`

```yaml
layer:
  ffn:
    activation: geglu
    use_gated_activation: true
```

### ReGLU
Gated activation with ReLU: `ReGLU(x) = ReLU(xW_g + b_g) ⊙ (xW_u + b_u)`

```yaml
layer:
  ffn:
    activation: reglu
    use_gated_activation: true
```

## Feature Compatibility

- **RoPE** works with standard attention, GQA, MQA, and MLA
- **ALiBi** requires `positional_encoding.type: alibi` and `attention.mechanism: alibi`
- **MLA** compresses KV cache and works with all positional encodings (except ALiBi)
- **SwiGLU/GeGLU/ReGLU** require `use_gated_activation: true`
- **RMSNorm** can be used for both layer norms and final layer norm
- **GQA/MQA** work with all positional encodings and attention mechanisms
- **Pre-norm vs Post-norm** are architectural choices independent of other features

