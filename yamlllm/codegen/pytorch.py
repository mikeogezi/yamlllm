"""PyTorch code generator for decoder-only transformer models."""

import math
from ..schema import ModelConfig


def _generate_rmsnorm_class(config: ModelConfig) -> list[str]:
    """Generate RMSNorm class if needed."""
    if config.layer.layer_norm.type == "rmsnorm" or (
        config.final_layer_norm and config.final_layer_norm.type == "rmsnorm"
    ):
        return [
            "",
            "class RMSNorm(nn.Module):",
            '    """Root Mean Square Layer Normalization."""',
            "",
            "    def __init__(self, dim, eps=1e-6):",
            "        super().__init__()",
            "        self.eps = eps",
            "        self.weight = nn.Parameter(torch.ones(dim))",
            "",
            "    def forward(self, x):",
            "        norm = x.norm(dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)",
            "        return x / (norm + self.eps) * self.weight",
            "",
        ]
    return []


def _generate_rope_class(config: ModelConfig) -> list[str]:
    """Generate RoPE (Rotary Positional Embedding) class if needed."""
    pos_enc = config.embedding.positional_encoding
    if pos_enc and pos_enc.type == "rope":
        theta = pos_enc.rope_theta
        scaling_type = pos_enc.rope_scaling
        scaling_factor = pos_enc.rope_scaling_factor or 1.0
        
        lines = [
            "",
            "class RotaryEmbedding(nn.Module):",
            '    """Rotary Positional Embedding (RoPE) with optional scaling."""',
            "",
            f"    def __init__(self, dim, max_position_embeddings=2048, base={theta}, scaling_type={repr(scaling_type)}, scaling_factor={scaling_factor}):",
            "        super().__init__()",
            "        self.dim = dim",
            "        self.max_position_embeddings = max_position_embeddings",
            "        self.base = base",
            "        self.scaling_type = scaling_type",
            "        self.scaling_factor = scaling_factor",
            "",
        ]
        
        if scaling_type == "dynamic":
            # NTK-aware scaling: adjust base frequency
            lines.extend([
                "        # Dynamic NTK-aware scaling: adjust base frequency for extrapolation",
                "        if scaling_type == 'dynamic' and scaling_factor > 1.0:",
                "            base = base * ((scaling_factor * max_position_embeddings / max_position_embeddings) - (scaling_factor - 1)) ** (dim / (dim - 2))",
                "",
            ])
        
        lines.extend([
            "        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))",
            "        self.register_buffer('inv_freq', inv_freq)",
            "",
            "    def forward(self, x, position_ids):",
            "        # x: [batch_size, num_heads, seq_len, head_dim]",
        ])
        
        if scaling_type == "linear":
            lines.extend([
                "        # Linear scaling: scale position IDs",
                "        if self.scaling_factor > 1.0:",
                "            position_ids = position_ids.float() / self.scaling_factor",
                "",
            ])
        
        lines.extend([
            "        inv_freq_expanded = self.inv_freq[None, :, None].float()",
            "        position_ids_expanded = position_ids[:, None, :, None].float()",
            "        freqs = (inv_freq_expanded @ position_ids_expanded.transpose(2, 3)).transpose(2, 3)",
            "        emb = torch.cat((freqs, freqs), dim=-1)",
            "        cos = emb.cos()",
            "        sin = emb.sin()",
            "        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)",
            "",
            "    def rotate_half(self, x):",
            "        x1, x2 = x.chunk(2, dim=-1)",
            "        return torch.cat((-x2, x1), dim=-1)",
            "",
            "    def apply_rotary_pos_emb(self, q, k, cos, sin):",
            "        q_embed = (q * cos) + (self.rotate_half(q) * sin)",
            "        k_embed = (k * cos) + (self.rotate_half(k) * sin)",
            "        return q_embed, k_embed",
            "",
        ])
        return lines
    return []


def _generate_attention_class(config: ModelConfig) -> list[str]:
    """Generate attention class with support for standard, ALiBi, GQA, MQA, and MLA."""
    head_dim = (
        config.layer.attention.head_dim
        if config.layer.attention.head_dim is not None
        else config.layer.hidden_dim // config.layer.attention.num_heads
    )
    
    use_rope = (
        config.embedding.positional_encoding
        and config.embedding.positional_encoding.type == "rope"
    )
    use_alibi = config.layer.attention.mechanism == "alibi"
    use_mla = config.layer.attention.mechanism == "mla"
    use_flash = config.layer.attention.use_flash_attention and not use_alibi and not use_mla
    
    # GQA/MQA support
    num_kv_heads = config.layer.attention.num_kv_heads or config.layer.attention.num_heads
    use_gqa = num_kv_heads < config.layer.attention.num_heads
    kv_dim = num_kv_heads * head_dim
    
    # MLA parameters
    if use_mla:
        mla_latent_dim = config.layer.attention.mla_latent_dim or (config.layer.hidden_dim // 4)
        mla_rank = config.layer.attention.mla_rank or 8
    else:
        mla_latent_dim = None
        mla_rank = None
    
    attn_type = "Multi-Latent" if use_mla else ("Multi-Query" if num_kv_heads == 1 else ("Grouped Query" if use_gqa else "Multi-Head"))
    
    lines = [
        f"class CausalSelfAttention(nn.Module):",
        f'    """{attn_type} causal self-attention with {config.layer.attention.num_heads} query heads and {num_kv_heads} key-value heads."""',
        "",
        "    def __init__(self, config):",
        f"        super().__init__()",
        f"        self.num_heads = {config.layer.attention.num_heads}",
        f"        self.num_kv_heads = {num_kv_heads}",
        f"        self.hidden_dim = {config.layer.hidden_dim}",
        f"        self.head_dim = {head_dim}",
        f"        self.scale = self.head_dim ** -0.5",
        f"        self.use_flash_attention = {use_flash}",
        "",
        f"        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias={config.layer.attention.bias})",
    ]
    
    if use_mla:
        # MLA: down-project K/V to latent space, then up-project
        lines.extend([
            f"        # MLA: Key/Value projections to latent space",
            f"        self.k_down_proj = nn.Linear(self.hidden_dim, {mla_latent_dim}, bias={config.layer.attention.bias})",
            f"        self.v_down_proj = nn.Linear(self.hidden_dim, {mla_latent_dim}, bias={config.layer.attention.bias})",
            f"        self.k_up_proj = nn.Linear({mla_latent_dim}, {kv_dim}, bias={config.layer.attention.bias})",
            f"        self.v_up_proj = nn.Linear({mla_latent_dim}, {kv_dim}, bias={config.layer.attention.bias})",
            f"        self.mla_latent_dim = {mla_latent_dim}",
            f"        self.mla_rank = {mla_rank}",
        ])
    else:
        lines.extend([
            f"        self.k_proj = nn.Linear(self.hidden_dim, {kv_dim}, bias={config.layer.attention.bias})",
            f"        self.v_proj = nn.Linear(self.hidden_dim, {kv_dim}, bias={config.layer.attention.bias})",
        ])
    
    lines.extend([
        f"        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias={config.layer.attention.bias})",
        f"        self.dropout = nn.Dropout({config.layer.attention.dropout})",
    ])
    
    if use_rope:
        lines.extend([
            f"        self.rotary_emb = RotaryEmbedding(",
            f"            self.head_dim,",
            f"            max_position_embeddings={config.embedding.max_position_embeddings},",
            f"            base={config.embedding.positional_encoding.rope_theta}",
            f"        )",
        ])
    
    if use_alibi:
        max_pos = (
            config.layer.attention.alibi_max_positions
            or config.embedding.max_position_embeddings
        )
        # ALiBi slopes: negative slopes for each head
        slopes = []
        for i in range(config.layer.attention.num_heads):
            slope = 2 ** (-(2 ** -(math.log2(config.layer.attention.num_heads) - 3)) * (i + 1))
            slopes.append(slope)
        slopes_str = "[" + ", ".join(f"{s:.10f}" for s in slopes) + "]"
        lines.extend([
            f"        # ALiBi slopes",
            f"        alibi_slopes = torch.tensor({slopes_str})",
            f"        self.register_buffer('alibi_slopes', alibi_slopes)",
        ])
    
    lines.extend([
        "",
        "    def forward(self, x, mask=None, position_ids=None):",
        "        batch_size, seq_len, _ = x.shape",
        "",
        "        # Project to Q",
        "        q = self.q_proj(x)",
    ])
    
    if use_mla:
        # For MLA, we need to project Q to match latent dimension or use a different approach
        # Standard MLA: compute attention in latent space by projecting Q to latent dim per head
        lines.extend([
            "",
            "        # MLA: Down-project K and V to latent space",
            "        k_latent = self.k_down_proj(x)",
            "        v_latent = self.v_down_proj(x)",
            "",
            "        # Reshape Q to full dimension",
            "        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)",
            f"        # Reshape K, V to latent space (latent_dim must be divisible by num_heads)",
            f"        latent_head_dim = self.mla_latent_dim // self.num_heads",
            f"        k_latent = k_latent.view(batch_size, seq_len, self.num_heads, latent_head_dim).transpose(1, 2)",
            f"        v_latent = v_latent.view(batch_size, seq_len, self.num_heads, latent_head_dim).transpose(1, 2)",
            "",
            "        # For MLA: project Q to latent dimension for attention computation",
            "        # We'll use a simple projection: take first latent_head_dim elements or average pool",
            "        # Standard approach: compute QK^T where K is in latent space",
            "        # We need Q to match latent_head_dim, so we project Q per head",
            "        q_latent = q[..., :latent_head_dim]  # Truncate Q to match latent dimension",
            "",
            "        # Compute attention in latent space",
            f"        scale = latent_head_dim ** -0.5",
            "        scores = torch.matmul(q_latent, k_latent.transpose(-2, -1)) * scale",
        ])
    else:
        lines.extend([
            "        # Project to K, V",
            "        k = self.k_proj(x)",
            "        v = self.v_proj(x)",
            "",
            "        # Reshape for attention",
            "        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)",
            "        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)",
            "        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)",
            "",
        ])
        
        if use_gqa:
            lines.extend([
                "        # Repeat K and V for GQA/MQA",
                "        if self.num_kv_heads < self.num_heads:",
                "            repeat_kv = self.num_heads // self.num_kv_heads",
                "            k = k.repeat_interleave(repeat_kv, dim=1)",
                "            v = v.repeat_interleave(repeat_kv, dim=1)",
                "",
            ])
        
        if use_rope:
            lines.extend([
                "",
                "        # Apply RoPE",
                "        if position_ids is None:",
                "            position_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)",
                "        cos, sin = self.rotary_emb(q, position_ids)",
                "        q, k = self.rotary_emb.apply_rotary_pos_emb(q, k, cos, sin)",
            ])
        
        if use_flash:
            # Flash Attention path using PyTorch's efficient SDPA
            lines.extend([
                "",
                "        # Use Flash Attention (scaled_dot_product_attention)",
                "        # This is much faster on compatible hardware (CUDA with PyTorch 2.0+)",
                f"        dropout_p = {config.layer.attention.dropout} if self.training else 0.0",
                "        attn_output = F.scaled_dot_product_attention(",
                "            q, k, v,",
                "            attn_mask=None,",
                "            dropout_p=dropout_p,",
                "            is_causal=True,",
                "        )",
            ])
        else:
            lines.extend([
                "",
                "        # Scaled dot-product attention",
                "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale",
            ])
            
            if use_alibi:
                lines.extend([
                    "",
                    "        # Apply ALiBi biases",
                    "        alibi_bias = self.alibi_slopes.unsqueeze(0).unsqueeze(-1) * torch.arange(seq_len, device=x.device).float()",
                    "        alibi_bias = alibi_bias.unsqueeze(0).unsqueeze(-1) - alibi_bias.unsqueeze(0).unsqueeze(-2)",
                    "        scores = scores + alibi_bias.unsqueeze(0)",
                ])
    
    # Apply mask and compute attention (common for both paths)
    if use_mla:
        # MLA path: apply mask, compute attention, up-project
        lines.extend([
            "",
            "        # Apply causal mask",
            "        if mask is None:",
            "            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()",
            "            mask = causal_mask.unsqueeze(0).unsqueeze(0)",
            "        scores = scores.masked_fill(mask, float('-inf'))",
            "",
            "        attn_weights = F.softmax(scores, dim=-1)",
            "        attn_weights = self.dropout(attn_weights)",
            "",
            "        # Apply attention to latent values",
            "        attn_latent = torch.matmul(attn_weights, v_latent)",
            "",
            "        # Up-project from latent space back to full dimension",
            "        attn_latent = attn_latent.transpose(1, 2).contiguous()",
            f"        attn_latent = attn_latent.view(batch_size, seq_len, self.mla_latent_dim)",
            "        attn_output = self.v_up_proj(attn_latent)",
            "        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)",
        ])
    elif not use_flash:
        # Standard manual attention path (not Flash)
        lines.extend([
            "",
            "        # Apply causal mask",
            "        if mask is None:",
            "            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()",
            "            mask = causal_mask.unsqueeze(0).unsqueeze(0)",
            "        scores = scores.masked_fill(mask, float('-inf'))",
            "",
            "        attn_weights = F.softmax(scores, dim=-1)",
            "        attn_weights = self.dropout(attn_weights)",
            "        attn_output = torch.matmul(attn_weights, v)",
        ])
    
    lines.extend([
        "",
        "        # Reshape and project output",
        "        attn_output = attn_output.transpose(1, 2).contiguous()",
        "        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)",
        "        output = self.out_proj(attn_output)",
        "",
        "        return output",
    ])
    
    return lines


def _generate_ffn_class(config: ModelConfig) -> list[str]:
    """Generate feed-forward network with support for SwiGLU, GeGLU, ReGLU."""
    use_gated = config.layer.ffn.use_gated_activation
    activation = config.layer.ffn.activation
    
    lines = [
        "class FeedForward(nn.Module):",
        f'    """Feed-forward network with {activation} activation."""',
        "",
        "    def __init__(self, config):",
        f"        super().__init__()",
        f"        self.intermediate_size = {config.layer.ffn.intermediate_size}",
        f"        self.hidden_dim = {config.layer.hidden_dim}",
        f"        self.activation_fn = '{activation}'",
        f"        self.use_gated = {use_gated}",
        "",
    ]
    
    if use_gated:
        # Gated activations: gate_proj and up_proj
        lines.extend([
            f"        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias={config.layer.ffn.bias})",
            f"        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias={config.layer.ffn.bias})",
            f"        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_dim, bias={config.layer.ffn.bias})",
        ])
    else:
        lines.extend([
            f"        self.fc1 = nn.Linear(self.hidden_dim, self.intermediate_size, bias={config.layer.ffn.bias})",
            f"        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_dim, bias={config.layer.ffn.bias})",
        ])
    
    lines.extend([
        f"        self.dropout = nn.Dropout({config.layer.ffn.dropout})",
        "",
        "    def forward(self, x):",
    ])
    
    if use_gated:
        if activation == "swiglu":
            lines.extend([
                "        gate = self.gate_proj(x)",
                "        up = self.up_proj(x)",
                "        x = F.silu(gate) * up",
            ])
        elif activation == "geglu":
            lines.extend([
                "        gate = self.gate_proj(x)",
                "        up = self.up_proj(x)",
                "        x = F.gelu(gate) * up",
            ])
        elif activation == "reglu":
            lines.extend([
                "        gate = self.gate_proj(x)",
                "        up = self.up_proj(x)",
                "        x = F.relu(gate) * up",
            ])
        else:
            lines.extend([
                "        gate = self.gate_proj(x)",
                "        up = self.up_proj(x)",
                f"        # Using {activation} for gating",
                "        if self.activation_fn == 'gelu':",
                "            x = F.gelu(gate) * up",
                "        elif self.activation_fn == 'relu':",
                "            x = F.relu(gate) * up",
                "        elif self.activation_fn == 'silu':",
                "            x = F.silu(gate) * up",
            ])
        lines.extend([
            "        x = self.dropout(x)",
            "        x = self.down_proj(x)",
            "        return x",
        ])
    else:
        lines.extend([
            "        x = self.fc1(x)",
            f"        if self.activation_fn == 'gelu':",
            "            x = F.gelu(x)",
            f"        elif self.activation_fn == 'relu':",
            "            x = F.relu(x)",
            f"        elif self.activation_fn == 'silu':",
            "            x = F.silu(x)",
            "        x = self.dropout(x)",
            "        x = self.fc2(x)",
            "        return x",
        ])
    
    return lines


def _get_norm_class(config: ModelConfig, norm_config) -> str:
    """Get the normalization class name based on config."""
    if norm_config.type == "rmsnorm":
        return "RMSNorm"
    return "nn.LayerNorm"


def _get_norm_init(norm_config, hidden_dim: int) -> str:
    """Get normalization initialization code."""
    if norm_config.type == "rmsnorm":
        return f"RMSNorm({hidden_dim}, eps={norm_config.eps})"
    return f"nn.LayerNorm({hidden_dim}, eps={norm_config.eps}, elementwise_affine={norm_config.elementwise_affine})"


def generate_pytorch_code(config: ModelConfig) -> str:
    """Generate PyTorch nn.Module code from a ModelConfig."""
    
    lines = [
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "import math",
        "from typing import Optional",
        "",
    ]
    
    # Add helper classes
    lines.extend(_generate_rmsnorm_class(config))
    lines.extend(_generate_rope_class(config))
    lines.extend([""])
    lines.extend(_generate_attention_class(config))
    lines.extend([""])
    lines.extend(_generate_ffn_class(config))
    lines.extend([
        "",
        "class DecoderLayer(nn.Module):",
        '    """Single decoder transformer layer."""',
        "",
        "    def __init__(self, config):",
        f"        super().__init__()",
        f"        self.attention = CausalSelfAttention(config)",
        f"        self.ffn = FeedForward(config)",
        f"        self.ln1 = {_get_norm_init(config.layer.layer_norm, config.layer.hidden_dim)}",
        f"        self.ln2 = {_get_norm_init(config.layer.layer_norm, config.layer.hidden_dim)}",
        f"        self.residual_dropout = nn.Dropout({config.layer.residual_dropout})",
        f"        self.norm_placement = '{config.layer.norm_placement}'",
        "",
        "    def forward(self, x, mask=None, position_ids=None):",
    ])
    
    if config.layer.norm_placement == "pre":
        # Pre-norm: normalize before attention/ffn
        lines.extend([
            "        # Self-attention with residual (pre-norm)",
            "        residual = x",
            "        x = self.ln1(x)",
            "        x = self.attention(x, mask, position_ids)",
            "        x = self.residual_dropout(x)",
            "        x = x + residual",
            "",
            "        # Feed-forward with residual (pre-norm)",
            "        residual = x",
            "        x = self.ln2(x)",
            "        x = self.ffn(x)",
            "        x = self.residual_dropout(x)",
            "        x = x + residual",
        ])
    else:
        # Post-norm: normalize after attention/ffn
        lines.extend([
            "        # Self-attention with residual (post-norm)",
            "        residual = x",
            "        x = self.attention(x, mask, position_ids)",
            "        x = self.residual_dropout(x)",
            "        x = x + residual",
            "        x = self.ln1(x)",
            "",
            "        # Feed-forward with residual (post-norm)",
            "        residual = x",
            "        x = self.ffn(x)",
            "        x = self.residual_dropout(x)",
            "        x = x + residual",
            "        x = self.ln2(x)",
        ])
    
    lines.extend([
        "",
        "        return x",
        "",
        "",
        f"class {config.name}(nn.Module):",
        f'    """Decoder-only transformer model: {config.name}"""',
        "",
        "    def __init__(self, config=None):",
        f"        super().__init__()",
        f"        self.vocab_size = {config.embedding.vocab_size}",
        f"        self.max_position_embeddings = {config.embedding.max_position_embeddings}",
        f"        self.hidden_dim = {config.layer.hidden_dim}",
        f"        self.num_layers = {config.num_layers}",
        "",
    ])
    
    # Positional encoding setup
    pos_enc = config.embedding.positional_encoding
    use_rope = pos_enc and pos_enc.type == "rope"
    use_alibi = pos_enc and pos_enc.type == "alibi"
    use_learned = not pos_enc or pos_enc.type == "learned"
    
    lines.append(f"        # Embeddings")
    lines.append(f"        self.token_embedding = nn.Embedding(")
    lines.append(f"            {config.embedding.vocab_size},")
    lines.append(f"            {config.embedding.embedding_dim},")
    if config.embedding.padding_idx is not None:
        lines.append(f"            padding_idx={config.embedding.padding_idx},")
    else:
        lines.append(f"            padding_idx=None,")
    lines.append(f"        )")
    
    if use_learned:
        lines.append(f"        self.position_embedding = nn.Embedding(")
        lines.append(f"            {config.embedding.max_position_embeddings},")
        lines.append(f"            {config.embedding.embedding_dim}")
        lines.append(f"        )")
    
    lines.append(f"        self.embedding_dropout = nn.Dropout({config.embedding.dropout})")
    lines.append("")
    lines.append(f"        # Decoder layers")
    lines.append(f"        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range({config.num_layers})])")
    lines.append("")
    
    if config.final_layer_norm:
        lines.append(f"        # Final layer norm")
        lines.append(f"        self.final_ln = {_get_norm_init(config.final_layer_norm, config.layer.hidden_dim)}")
        lines.append("")
    
    lines.append(f"        # Output head")
    
    if config.tie_word_embeddings:
        lines.append(f"        self.lm_head = None  # Will use token_embedding weights")
    else:
        lines.append(f"        self.lm_head = nn.Linear({config.embedding.embedding_dim}, {config.embedding.vocab_size}, bias=False)")
    
    lines.extend([
        "",
        "    def forward(self, input_ids, attention_mask=None):",
        "        batch_size, seq_len = input_ids.shape",
        "",
        "        # Create position ids",
        "        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)",
        "",
        "        # Embeddings",
        "        token_embeds = self.token_embedding(input_ids)",
    ])
    
    if use_learned:
        lines.extend([
            "        pos_embeds = self.position_embedding(position_ids)",
            "        x = token_embeds + pos_embeds",
        ])
    else:
        lines.append("        x = token_embeds")
    
    lines.extend([
        "        x = self.embedding_dropout(x)",
        "",
        "        # Decoder layers",
        "        for layer in self.layers:",
    ])
    
    if use_rope or use_alibi:
        lines.append("            x = layer(x, mask=attention_mask, position_ids=position_ids)")
    else:
        lines.append("            x = layer(x, mask=attention_mask)")
    
    lines.append("")
    
    if config.final_layer_norm:
        lines.append(f"        # Final layer norm")
        lines.append(f"        x = self.final_ln(x)")
        lines.append("")
    
    lines.append("        # Language modeling head")
    
    if config.tie_word_embeddings:
        lines.extend([
            "        if self.lm_head is None:",
            "            logits = torch.matmul(x, self.token_embedding.weight.t())",
            "        else:",
            "            logits = self.lm_head(x)",
        ])
    else:
        lines.append("        logits = self.lm_head(x)")
    
    lines.extend([
        "",
        "        return logits",
    ])
    
    # Add initialization method and call if specified
    if config.initialization and config.initialization.type != "default":
        # Add call at end of __init__
        # Find the line before "def forward" and insert there
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("def forward"):
                lines.insert(i, "")
                lines.insert(i, "        # Initialize weights")
                lines.insert(i, "        self._init_weights()")
                break
        
        # Add initialization method
        lines.extend([
            "",
            "    def _init_weights(self):",
            '        """Initialize model weights."""',
        ])
        
        init_type = config.initialization.type
        if init_type == "gpt2":
            residual_scale = config.initialization.gpt2_residual_scale or (1.0 / math.sqrt(2 * config.num_layers))
            lines.extend([
                "        # GPT-2 style initialization",
                "        std = 0.02",
                "        for module in self.modules():",
                "            if isinstance(module, nn.Linear):",
                f"                module.weight.data.normal_(mean=0.0, std={residual_scale} * std)",
                "                if module.bias is not None:",
                "                    module.bias.data.zero_()",
                "            elif isinstance(module, nn.Embedding):",
                "                module.weight.data.normal_(mean=0.0, std=std)",
                "            elif isinstance(module, nn.LayerNorm):",
                "                if hasattr(module, 'weight') and module.weight is not None:",
                "                    module.weight.data.fill_(1.0)",
                "                if hasattr(module, 'bias') and module.bias is not None:",
                "                    module.bias.data.zero_()",
                "            # Handle RMSNorm if present",
                "            elif hasattr(module, 'weight') and hasattr(module, 'eps'):",
                "                # Likely RMSNorm",
                "                if module.weight is not None:",
                "                    module.weight.data.fill_(1.0)",
            ])
        elif init_type == "xavier_uniform":
            gain = config.initialization.gain
            lines.extend([
                f"        # Xavier uniform initialization (gain={gain})",
                "        for module in self.modules():",
                "            if isinstance(module, nn.Linear):",
                f"                nn.init.xavier_uniform_(module.weight, gain={gain})",
                "                if module.bias is not None:",
                "                    nn.init.constant_(module.bias, 0.0)",
            ])
        elif init_type == "xavier_normal":
            gain = config.initialization.gain
            lines.extend([
                f"        # Xavier normal initialization (gain={gain})",
                "        for module in self.modules():",
                "            if isinstance(module, nn.Linear):",
                f"                nn.init.xavier_normal_(module.weight, gain={gain})",
                "                if module.bias is not None:",
                "                    nn.init.constant_(module.bias, 0.0)",
            ])
        elif init_type == "kaiming_uniform":
            lines.extend([
                "        # Kaiming uniform initialization",
                "        for module in self.modules():",
                "            if isinstance(module, nn.Linear):",
                "                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')",
                "                if module.bias is not None:",
                "                    nn.init.constant_(module.bias, 0.0)",
            ])
        elif init_type == "kaiming_normal":
            lines.extend([
                "        # Kaiming normal initialization",
                "        for module in self.modules():",
                "            if isinstance(module, nn.Linear):",
                "                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')",
                "                if module.bias is not None:",
                "                    nn.init.constant_(module.bias, 0.0)",
            ])
    
    return "\n".join(lines)
