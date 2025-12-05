"""PyTorch code generator for decoder-only transformer models.

This module generates PyTorch nn.Module code from ModelIR (intermediate representation).
"""

from typing import List, Optional
from ..ir import (
    ModelIR, ModuleNode, LinearNode, EmbeddingNode, AttentionNode, 
    FFNNode, NormNode, SequenceNode, NodeType
)


class PyTorchRenderer:
    """Renders ModelIR to PyTorch nn.Module code."""
    
    def __init__(self):
        self.indent_level = 0
    
    def render(self, ir: ModelIR) -> str:
        """Generate PyTorch code from IR."""
        lines = self._generate_imports()
        lines.append("")
        
        # Generate helper classes if needed
        lines.extend(self._generate_rmsnorm_class(ir))
        lines.extend(self._generate_rope_class(ir))
        lines.append("")
        
        # Generate attention and FFN classes
        lines.extend(self._generate_attention_class(ir))
        lines.append("")
        lines.extend(self._generate_ffn_class(ir))
        lines.append("")
        
        # Generate decoder layer
        lines.extend(self._generate_decoder_layer(ir))
        lines.append("")
        
        # Generate main model class
        lines.extend(self._generate_model_class(ir))
        
        return "\n".join(lines)
    
    def _generate_imports(self) -> List[str]:
        """Generate import statements."""
        return [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "import math",
            "from typing import Optional",
        ]
    
    def _generate_rmsnorm_class(self, ir: ModelIR) -> List[str]:
        """Generate RMSNorm class if needed."""
        # Check if any layer uses RMSNorm
        needs_rmsnorm = any(
            module.node_type == NodeType.RMS_NORM
            for layer in ir.decoder_layers
            for module in layer.modules
        )
        
        if needs_rmsnorm:
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
    
    def _generate_rope_class(self, ir: ModelIR) -> List[str]:
        """Generate RoPE class if any attention uses RoPE."""
        # Check if any attention layer uses RoPE
        attention_node = self._get_first_attention_node(ir)
        if not attention_node or not attention_node.use_rope:
            return []
        
        rope_theta = attention_node.params.get('rope_theta', 10000.0)
        rope_scaling = attention_node.params.get('rope_scaling', None)
        rope_scaling_factor = attention_node.params.get('rope_scaling_factor', 1.0)
        
        lines = [
            "",
            "class RotaryEmbedding(nn.Module):",
            '    """Rotary Positional Embedding (RoPE) with optional scaling."""',
            "",
            f"    def __init__(self, dim, max_position_embeddings=2048, base={rope_theta}, scaling_type={repr(rope_scaling)}, scaling_factor={rope_scaling_factor}):",
            "        super().__init__()",
            "        self.dim = dim",
            "        self.max_position_embeddings = max_position_embeddings",
            "        self.base = base",
            "        self.scaling_type = scaling_type",
            "        self.scaling_factor = scaling_factor",
            "",
        ]
        
        if rope_scaling == "dynamic":
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
        
        if rope_scaling == "linear":
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
    
    def _get_first_attention_node(self, ir: ModelIR) -> Optional[AttentionNode]:
        """Get first attention node from IR (they should all be identical)."""
        for layer in ir.decoder_layers:
            for module in layer.modules:
                if isinstance(module, AttentionNode):
                    return module
        return None
    
    def _get_first_ffn_node(self, ir: ModelIR) -> Optional[FFNNode]:
        """Get first FFN node from IR."""
        for layer in ir.decoder_layers:
            for module in layer.modules:
                if isinstance(module, FFNNode):
                    return module
        return None
    
    def _get_norm_type(self, ir: ModelIR) -> str:
        """Get normalization type from first norm node."""
        for layer in ir.decoder_layers:
            for module in layer.modules:
                if isinstance(module, NormNode):
                    return module.norm_type
        return "layernorm"
    
    def _generate_attention_class(self, ir: ModelIR) -> List[str]:
        """Generate attention class from IR."""
        attention = self._get_first_attention_node(ir)
        if not attention:
            return []
        
        num_heads = attention.num_heads
        num_kv_heads = attention.num_kv_heads or num_heads
        head_dim = attention.head_dim
        hidden_dim = attention.hidden_dim
        kv_dim = num_kv_heads * head_dim
        use_flash = attention.use_flash
        use_rope = attention.use_rope
        use_alibi = attention.use_alibi
        use_gqa = attention.use_gqa
        dropout = attention.dropout
        
        # Check for MLA
        use_mla = attention.params.get('mechanism') == 'mla'
        mla_latent_dim = attention.params.get('mla_latent_dim')
        
        # Get bias setting from first linear child
        use_bias = False
        for child in attention.children:
            if isinstance(child, LinearNode):
                use_bias = child.use_bias
                break
        
        attn_type = "Multi-Latent" if use_mla else ("Multi-Query" if num_kv_heads == 1 else ("Grouped Query" if use_gqa else "Multi-Head"))
        
        lines = [
            f"class CausalSelfAttention(nn.Module):",
            f'    """{attn_type} causal self-attention with {num_heads} query heads and {num_kv_heads} key-value heads."""',
            "",
            "    def __init__(self, config):",
            f"        super().__init__()",
            f"        self.num_heads = {num_heads}",
            f"        self.num_kv_heads = {num_kv_heads}",
            f"        self.hidden_dim = {hidden_dim}",
            f"        self.head_dim = {head_dim}",
            f"        self.scale = self.head_dim ** -0.5",
            f"        self.use_flash_attention = {use_flash}",
            "",
            f"        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias={use_bias})",
        ]
        
        if use_mla:
            lines.extend([
                f"        # MLA: Key/Value projections to latent space",
                f"        self.k_down_proj = nn.Linear(self.hidden_dim, {mla_latent_dim}, bias={use_bias})",
                f"        self.v_down_proj = nn.Linear(self.hidden_dim, {mla_latent_dim}, bias={use_bias})",
                f"        self.k_up_proj = nn.Linear({mla_latent_dim}, {kv_dim}, bias={use_bias})",
                f"        self.v_up_proj = nn.Linear({mla_latent_dim}, {kv_dim}, bias={use_bias})",
                f"        self.mla_latent_dim = {mla_latent_dim}",
                f"        self.mla_rank = {attention.params.get('mla_rank', 8)}",
            ])
        else:
            lines.extend([
                f"        self.k_proj = nn.Linear(self.hidden_dim, {kv_dim}, bias={use_bias})",
                f"        self.v_proj = nn.Linear(self.hidden_dim, {kv_dim}, bias={use_bias})",
            ])
        
        lines.extend([
            f"        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias={use_bias})",
            f"        self.dropout = nn.Dropout({dropout})",
        ])
        
        if use_rope:
            rope_theta = attention.params.get('rope_theta', 10000.0)
            lines.extend([
                f"        self.rotary_emb = RotaryEmbedding(",
                f"            self.head_dim,",
                f"            max_position_embeddings={ir.max_position_embeddings},",
                f"            base={rope_theta}",
                f"        )",
            ])
        
        if use_alibi:
            import math
            alibi_max_pos = attention.params.get('alibi_max_positions', ir.max_position_embeddings)
            slopes = []
            for i in range(num_heads):
                slope = 2 ** (-(2 ** -(math.log2(num_heads) - 3)) * (i + 1))
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
                lines.extend([
                    "",
                    "        # Use Flash Attention (scaled_dot_product_attention)",
                    "        # This is much faster on compatible hardware (CUDA with PyTorch 2.0+)",
                    f"        dropout_p = {dropout} if self.training else 0.0",
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
    
    def _generate_ffn_class(self, ir: ModelIR) -> List[str]:
        """Generate FFN class from IR."""
        ffn = self._get_first_ffn_node(ir)
        if not ffn:
            return []
        
        use_gated = ffn.use_gated
        activation = ffn.activation
        hidden_dim = ffn.hidden_dim
        intermediate_size = ffn.intermediate_size
        dropout = ffn.dropout
        use_bias = ffn.use_bias
        
        lines = [
            "class FeedForward(nn.Module):",
            f'    """Feed-forward network with {activation} activation."""',
            "",
            "    def __init__(self, config):",
            f"        super().__init__()",
            f"        self.intermediate_size = {intermediate_size}",
            f"        self.hidden_dim = {hidden_dim}",
            f"        self.activation_fn = '{activation}'",
            f"        self.use_gated = {use_gated}",
            "",
        ]
        
        if use_gated:
            lines.extend([
                f"        self.gate_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias={use_bias})",
                f"        self.up_proj = nn.Linear(self.hidden_dim, self.intermediate_size, bias={use_bias})",
                f"        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_dim, bias={use_bias})",
            ])
        else:
            lines.extend([
                f"        self.fc1 = nn.Linear(self.hidden_dim, self.intermediate_size, bias={use_bias})",
                f"        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_dim, bias={use_bias})",
            ])
        
        lines.extend([
            f"        self.dropout = nn.Dropout({dropout})",
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
    
    def _get_norm_init(self, ir: ModelIR) -> str:
        """Get norm initialization code."""
        for layer in ir.decoder_layers:
            for module in layer.modules:
                if isinstance(module, NormNode):
                    if module.norm_type == "rmsnorm":
                        return f"RMSNorm({ir.hidden_dim}, eps={module.eps})"
                    return f"nn.LayerNorm({ir.hidden_dim}, eps={module.eps}, elementwise_affine={module.elementwise_affine})"
        return f"nn.LayerNorm({ir.hidden_dim})"
    
    def _get_norm_placement(self, ir: ModelIR) -> str:
        """Get norm placement from first decoder layer."""
        if ir.decoder_layers:
            return ir.decoder_layers[0].params.get('norm_placement', 'pre')
        return 'pre'
    
    def _get_residual_dropout(self, ir: ModelIR) -> float:
        """Get residual dropout from first decoder layer."""
        if ir.decoder_layers:
            for module in ir.decoder_layers[0].modules:
                if module.node_type == NodeType.DROPOUT and module.name == "residual_dropout":
                    return module.params.get('p', 0.0)
        return 0.0
    
    def _generate_decoder_layer(self, ir: ModelIR) -> List[str]:
        """Generate decoder layer class."""
        norm_placement = self._get_norm_placement(ir)
        norm_init = self._get_norm_init(ir)
        residual_dropout = self._get_residual_dropout(ir)
        
        lines = [
            "class DecoderLayer(nn.Module):",
            '    """Single decoder transformer layer."""',
            "",
            "    def __init__(self, config):",
            f"        super().__init__()",
            f"        self.attention = CausalSelfAttention(config)",
            f"        self.ffn = FeedForward(config)",
            f"        self.ln1 = {norm_init}",
            f"        self.ln2 = {norm_init}",
            f"        self.residual_dropout = nn.Dropout({residual_dropout})",
            f"        self.norm_placement = '{norm_placement}'",
            "",
            "    def forward(self, x, mask=None, position_ids=None):",
        ]
        
        if norm_placement == "pre":
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
        ])
        
        return lines
    
    def _has_learned_pos_embedding(self, ir: ModelIR) -> bool:
        """Check if model uses learned positional embeddings."""
        for module in ir.embedding_modules:
            if isinstance(module, EmbeddingNode) and module.name == "position_embedding":
                return True
        return False
    
    def _get_token_embedding_node(self, ir: ModelIR) -> Optional[EmbeddingNode]:
        """Get token embedding node."""
        for module in ir.embedding_modules:
            if isinstance(module, EmbeddingNode) and module.name == "token_embedding":
                return module
        return None
    
    def _get_embedding_dropout(self, ir: ModelIR) -> float:
        """Get embedding dropout."""
        for module in ir.embedding_modules:
            if module.node_type == NodeType.DROPOUT:
                return module.params.get('p', 0.0)
        return 0.0
    
    def _generate_model_class(self, ir: ModelIR) -> List[str]:
        """Generate main model class."""
        attention = self._get_first_attention_node(ir)
        use_rope = attention and attention.use_rope
        use_alibi = attention and attention.use_alibi
        use_learned = self._has_learned_pos_embedding(ir)
        
        token_emb = self._get_token_embedding_node(ir)
        padding_idx = token_emb.padding_idx if token_emb else None
        embedding_dropout = self._get_embedding_dropout(ir)
        
        lines = [
            f"class {ir.name}(nn.Module):",
            f'    """Decoder-only transformer model: {ir.name}"""',
            "",
            "    def __init__(self, config=None):",
            f"        super().__init__()",
            f"        self.vocab_size = {ir.vocab_size}",
            f"        self.max_position_embeddings = {ir.max_position_embeddings}",
            f"        self.hidden_dim = {ir.hidden_dim}",
            f"        self.num_layers = {ir.num_layers}",
            "",
            f"        # Embeddings",
            f"        self.token_embedding = nn.Embedding(",
            f"            {ir.vocab_size},",
            f"            {ir.hidden_dim},",
        ]
        
        if padding_idx is not None:
            lines.append(f"            padding_idx={padding_idx},")
        else:
            lines.append(f"            padding_idx=None,")
        lines.append(f"        )")
        
        if use_learned:
            lines.extend([
                f"        self.position_embedding = nn.Embedding(",
                f"            {ir.max_position_embeddings},",
                f"            {ir.hidden_dim}",
                f"        )",
            ])
        
        lines.extend([
            f"        self.embedding_dropout = nn.Dropout({embedding_dropout})",
            "",
            f"        # Decoder layers",
            f"        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range({ir.num_layers})])",
            "",
        ])
        
        # Final layer norm
        if ir.use_final_norm:
            # Find final norm in output modules
            final_norm_init = self._get_norm_init(ir)
            lines.extend([
                f"        # Final layer norm",
                f"        self.final_ln = {final_norm_init}",
                "",
            ])
        
        lines.append(f"        # Output head")
        
        if ir.tie_embeddings:
            lines.append(f"        self.lm_head = None  # Will use token_embedding weights")
        else:
            lines.append(f"        self.lm_head = nn.Linear({ir.hidden_dim}, {ir.vocab_size}, bias=False)")
        
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
        
        if ir.use_final_norm:
            lines.extend([
                f"        # Final layer norm",
                f"        x = self.final_ln(x)",
                "",
            ])
        
        lines.append("        # Language modeling head")
        
        if ir.tie_embeddings:
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
        
        return lines


def generate_pytorch_code(ir: ModelIR) -> str:
    """Generate PyTorch nn.Module code from ModelIR."""
    renderer = PyTorchRenderer()
    return renderer.render(ir)
