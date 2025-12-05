"""JAX/Flax code generator for YamlLLM models."""

from typing import List
from ..ir import ModelIR, ModuleNode, LinearNode, EmbeddingNode, AttentionNode, FFNNode, NormNode, SequenceNode, NodeType


class FlaxRenderer:
    """Renders ModelIR to JAX/Flax code."""
    
    def __init__(self):
        self.indent_level = 0
    
    def render(self, ir: ModelIR) -> str:
        """Generate Flax code from IR."""
        lines = self._generate_imports()
        lines.append("")
        
        # Generate helper classes if needed
        lines.extend(self._generate_helpers(ir))
        
        # Generate main model
        lines.extend(self._generate_model_class(ir))
        
        return "\n".join(lines)
    
    def _generate_imports(self) -> List[str]:
        """Generate import statements."""
        return [
            "import jax",
            "import jax.numpy as jnp",
            "from flax import linen as nn",
            "from typing import Optional",
        ]
    
    def _generate_helpers(self, ir: ModelIR) -> List[str]:
        """Generate helper classes (RMSNorm, RoPE, etc.)."""
        lines = []
        
        # Check if we need RMSNorm
        needs_rmsnorm = any(
            module.node_type == NodeType.RMS_NORM
            for layer in ir.decoder_layers
            for module in layer.modules
        )
        
        if needs_rmsnorm:
            lines.extend([
                "",
                "class RMSNorm(nn.Module):",
                '    """Root Mean Square Layer Normalization."""',
                "    eps: float = 1e-6",
                "",
                "    @nn.compact",
                "    def __call__(self, x):",
                "        scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))",
                "        norm = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)",
                "        return x / norm * scale",
                "",
            ])
        
        return lines
    
    def _generate_model_class(self, ir: ModelIR) -> List[str]:
        """Generate main model class."""
        lines = [
            "",
            f"class {ir.name}(nn.Module):",
            f'    """{ir.name}: Decoder-only transformer model."""',
            f"    vocab_size: int = {ir.vocab_size}",
            f"    hidden_dim: int = {ir.hidden_dim}",
            f"    num_layers: int = {ir.num_layers}",
            "",
            "    @nn.compact",
            "    def __call__(self, input_ids, deterministic: bool = True):",
            "        batch_size, seq_len = input_ids.shape",
            "",
        ]
        
        # Embeddings
        lines.extend(self._generate_embeddings(ir, indent=2))
        
        # Decoder layers
        lines.extend([
            "        # Decoder layers",
        ])
        
        for i, layer in enumerate(ir.decoder_layers):
            lines.extend(self._generate_decoder_layer(layer, i, indent=2))
        
        # Output
        lines.extend(self._generate_output(ir, indent=2))
        
        return lines
    
    def _generate_embeddings(self, ir: ModelIR, indent: int) -> List[str]:
        """Generate embedding code."""
        ind = "    " * indent
        lines = []
        
        # Token embedding
        lines.extend([
            f"{ind}# Token embedding",
            f"{ind}x = nn.Embed(",
            f"{ind}    num_embeddings={ir.vocab_size},",
            f"{ind}    features={ir.hidden_dim},",
            f"{ind}    name='token_embedding'",
            f"{ind})(input_ids)",
        ])
        
        # Position embedding (if learned)
        has_pos_emb = any(m.name == "position_embedding" for m in ir.embedding_modules)
        if has_pos_emb:
            lines.extend([
                "",
                f"{ind}# Position embedding",
                f"{ind}position_ids = jnp.arange(seq_len)[None, :]",
                f"{ind}pos_emb = nn.Embed(",
                f"{ind}    num_embeddings={ir.max_position_embeddings},",
                f"{ind}    features={ir.hidden_dim},",
                f"{ind}    name='position_embedding'",
                f"{ind})(position_ids)",
                f"{ind}x = x + pos_emb",
            ])
        
        lines.append("")
        return lines
    
    def _generate_decoder_layer(self, layer: SequenceNode, layer_idx: int, indent: int) -> List[str]:
        """Generate a single decoder layer."""
        ind = "    " * indent
        lines = []
        
        norm_placement = layer.params.get('norm_placement', 'pre')
        
        # Find attention and FFN modules
        attention = None
        ffn = None
        for module in layer.modules:
            if module.node_type == NodeType.ATTENTION:
                attention = module
            elif module.node_type == NodeType.FFN:
                ffn = module
        
        if norm_placement == "pre":
            # Pre-norm
            lines.extend([
                f"{ind}# Layer {layer_idx} - Attention",
                f"{ind}residual = x",
                f"{ind}x = nn.LayerNorm(name='ln1_{layer_idx}')(x)",
            ])
            
            if attention:
                lines.extend(self._generate_attention_call(attention, layer_idx, indent))
            
            lines.extend([
                f"{ind}x = residual + x",
                "",
                f"{ind}# Layer {layer_idx} - FFN",
                f"{ind}residual = x",
                f"{ind}x = nn.LayerNorm(name='ln2_{layer_idx}')(x)",
            ])
            
            if ffn:
                lines.extend(self._generate_ffn_call(ffn, layer_idx, indent))
            
            lines.extend([
                f"{ind}x = residual + x",
                "",
            ])
        
        return lines
    
    def _generate_attention_call(self, attention: AttentionNode, layer_idx: int, indent: int) -> List[str]:
        """Generate attention mechanism code."""
        ind = "    " * indent
        lines = []
        
        num_heads = attention.params['num_heads']
        head_dim = attention.params['head_dim']
        hidden_dim = attention.params['hidden_dim']
        use_flash = attention.params.get('use_flash', False)
        
        # Project to Q, K, V
        lines.extend([
            f"{ind}q = nn.Dense({hidden_dim}, name='q_proj_{layer_idx}')(x)",
            f"{ind}k = nn.Dense({hidden_dim}, name='k_proj_{layer_idx}')(x)",
            f"{ind}v = nn.Dense({hidden_dim}, name='v_proj_{layer_idx}')(x)",
            "",
            f"{ind}# Reshape for multi-head attention",
            f"{ind}q = q.reshape(batch_size, seq_len, {num_heads}, {head_dim}).transpose(0, 2, 1, 3)",
            f"{ind}k = k.reshape(batch_size, seq_len, {num_heads}, {head_dim}).transpose(0, 2, 1, 3)",
            f"{ind}v = v.reshape(batch_size, seq_len, {num_heads}, {head_dim}).transpose(0, 2, 1, 3)",
            "",
        ])
        
        if use_flash:
            lines.extend([
                f"{ind}# Flash Attention (using JAX's scaled_dot_product_attention)",
                f"{ind}attn_output = nn.dot_product_attention(",
                f"{ind}    q, k, v,",
                f"{ind}    mask=nn.make_causal_mask(input_ids[:, :, None]),",
                f"{ind}    deterministic=deterministic",
                f"{ind})",
            ])
        else:
            lines.extend([
                f"{ind}# Standard attention",
                f"{ind}scale = {head_dim} ** -0.5",
                f"{ind}scores = (q @ k.transpose(0, 1, 3, 2)) * scale",
                f"{ind}causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))[None, None, :, :]",
                f"{ind}scores = jnp.where(causal_mask, scores, -1e10)",
                f"{ind}attn_weights = jax.nn.softmax(scores, axis=-1)",
                f"{ind}attn_output = attn_weights @ v",
            ])
        
        lines.extend([
            "",
            f"{ind}# Reshape and project",
            f"{ind}attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, {hidden_dim})",
            f"{ind}x = nn.Dense({hidden_dim}, name='out_proj_{layer_idx}')(attn_output)",
            "",
        ])
        
        return lines
    
    def _generate_ffn_call(self, ffn: FFNNode, layer_idx: int, indent: int) -> List[str]:
        """Generate FFN code."""
        ind = "    " * indent
        lines = []
        
        intermediate_size = ffn.params['intermediate_size']
        hidden_dim = ffn.params['hidden_dim']
        activation = ffn.params['activation']
        use_gated = ffn.params.get('use_gated', False)
        
        if use_gated:
            # SwiGLU-style
            lines.extend([
                f"{ind}gate = nn.Dense({intermediate_size}, name='gate_proj_{layer_idx}')(x)",
                f"{ind}up = nn.Dense({intermediate_size}, name='up_proj_{layer_idx}')(x)",
            ])
            
            if activation == "swiglu":
                lines.append(f"{ind}x = jax.nn.silu(gate) * up")
            elif activation == "geglu":
                lines.append(f"{ind}x = jax.nn.gelu(gate) * up")
            else:
                lines.append(f"{ind}x = jax.nn.relu(gate) * up")
            
            lines.append(f"{ind}x = nn.Dense({hidden_dim}, name='down_proj_{layer_idx}')(x)")
        else:
            # Standard FFN
            lines.append(f"{ind}x = nn.Dense({intermediate_size}, name='fc1_{layer_idx}')(x)")
            
            if activation == "gelu":
                lines.append(f"{ind}x = jax.nn.gelu(x)")
            elif activation == "silu":
                lines.append(f"{ind}x = jax.nn.silu(x)")
            else:
                lines.append(f"{ind}x = jax.nn.relu(x)")
            
            lines.append(f"{ind}x = nn.Dense({hidden_dim}, name='fc2_{layer_idx}')(x)")
        
        lines.append("")
        return lines
    
    def _generate_output(self, ir: ModelIR, indent: int) -> List[str]:
        """Generate output projection."""
        ind = "    " * indent
        lines = []
        
        # Final norm (if configured)
        if ir.use_final_norm:
            lines.extend([
                f"{ind}# Final layer norm",
                f"{ind}x = nn.LayerNorm(name='final_ln')(x)",
                "",
            ])
        
        # LM head
        if ir.tie_embeddings:
            lines.extend([
                f"{ind}# Tied embeddings (share with input)",
                f"{ind}# Note: Flax doesn't support direct weight tying, use manual embedding",
                f"{ind}logits = nn.Dense({ir.vocab_size}, name='lm_head')(x)",
            ])
        else:
            lines.extend([
                f"{ind}# LM head",
                f"{ind}logits = nn.Dense({ir.vocab_size}, use_bias=False, name='lm_head')(x)",
            ])
        
        lines.extend([
            "",
            f"{ind}return logits",
        ])
        
        return lines


def generate_jax_code(ir: ModelIR) -> str:
    """Generate JAX/Flax code from ModelIR."""
    renderer = FlaxRenderer()
    return renderer.render(ir)
