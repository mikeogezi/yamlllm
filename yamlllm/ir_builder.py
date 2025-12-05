"""IR Builder - converts ModelConfig (schema) to ModelIR (intermediate representation)."""

from typing import List
from .schema import ModelConfig, AttentionConfig, FFNConfig, LayerNormConfig
from .ir import (
    ModelIR, ModuleNode, LinearNode, EmbeddingNode, AttentionNode,
    FFNNode, NormNode, SequenceNode, NodeType
)


class IRBuilder:
    """Builds IR from a validated ModelConfig."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def build(self) -> ModelIR:
        """Build complete model IR from config."""
        ir = ModelIR(
            name=self.config.name,
            vocab_size=self.config.embedding.vocab_size,
            hidden_dim=self.config.layer.hidden_dim,
            num_layers=self.config.num_layers,
            max_position_embeddings=self.config.embedding.max_position_embeddings,
            tie_embeddings=self.config.tie_word_embeddings,
            use_final_norm=self.config.final_layer_norm is not None
        )
        
        # Build embedding modules
        ir.embedding_modules = self._build_embedding_modules()
        
        # Build decoder layers
        for i in range(self.config.num_layers):
            layer = self._build_decoder_layer(i)
            ir.add_decoder_layer(layer)
        
        # Build output modules
        ir.output_modules = self._build_output_modules()
        
        return ir
    
    def _build_embedding_modules(self) -> List[ModuleNode]:
        """Build embedding-related modules."""
        modules = []
        
        # Token embedding
        token_emb = EmbeddingNode(
            name="token_embedding",
            num_embeddings=self.config.embedding.vocab_size,
            embedding_dim=self.config.embedding.embedding_dim,
            padding_idx=self.config.embedding.padding_idx
        )
        modules.append(token_emb)
        
        # Positional embedding (if learned)
        pos_enc = self.config.embedding.positional_encoding
        if not pos_enc or pos_enc.type == "learned":
            pos_emb = EmbeddingNode(
                name="position_embedding",
                num_embeddings=self.config.embedding.max_position_embeddings,
                embedding_dim=self.config.embedding.embedding_dim,
                padding_idx=None
            )
            modules.append(pos_emb)
        
        # Embedding dropout
        if self.config.embedding.dropout > 0:
            dropout = ModuleNode(
                name="embedding_dropout",
                node_type=NodeType.DROPOUT
            )
            dropout.add_param('p', self.config.embedding.dropout)
            modules.append(dropout)
        
        return modules
    
    def _build_decoder_layer(self, layer_idx: int) -> SequenceNode:
        """Build a single decoder layer as a sequence of modules."""
        layer_seq = SequenceNode(name=f"layer_{layer_idx}")
        
        layer_config = self.config.layer
        
        # Attention module
        attention = self._build_attention(layer_config.attention)
        layer_seq.add_module(attention)
        
        # Layer norm for attention
        ln1 = self._build_norm(layer_config.layer_norm, "ln1")
        layer_seq.add_module(ln1)
        
        # FFN module
        ffn = self._build_ffn(layer_config.ffn)
        layer_seq.add_module(ffn)
        
        # Layer norm for FFN
        ln2 = self._build_norm(layer_config.layer_norm, "ln2")
        layer_seq.add_module(ln2)
        
        # Residual dropout
        if layer_config.residual_dropout > 0:
            residual_dropout = ModuleNode(
                name="residual_dropout",
                node_type=NodeType.DROPOUT
            )
            residual_dropout.add_param('p', layer_config.residual_dropout)
            layer_seq.add_module(residual_dropout)
        
        # Store norm placement in sequence metadata
        layer_seq.params = {'norm_placement': layer_config.norm_placement}
        
        return layer_seq
    
    def _build_attention(self, attn_config: AttentionConfig) -> AttentionNode:
        """Build attention module IR."""
        # Compute head_dim
        head_dim = attn_config.head_dim
        if head_dim is None:
            head_dim = self.config.layer.hidden_dim // attn_config.num_heads
        
        # Compute num_kv_heads
        num_kv_heads = attn_config.num_kv_heads or attn_config.num_heads
        use_gqa = num_kv_heads < attn_config.num_heads
        
        # Check for RoPE
        pos_enc = self.config.embedding.positional_encoding
        use_rope = pos_enc and pos_enc.type == "rope"
        
        # Check for ALiBi
        use_alibi = attn_config.mechanism == "alibi"
        
        # Check for Flash Attention
        use_flash = attn_config.use_flash_attention and not use_alibi and attn_config.mechanism != "mla"
        
        attention = AttentionNode(
            name="attention",
            num_heads=attn_config.num_heads,
            head_dim=head_dim,
            hidden_dim=self.config.layer.hidden_dim,
            num_kv_heads=num_kv_heads,
            use_flash=use_flash,
            use_rope=use_rope,
            use_alibi=use_alibi,
            use_gqa=use_gqa,
            dropout=attn_config.dropout
        )
        
        # Add projection layers as children
        hidden_dim = self.config.layer.hidden_dim
        kv_dim = num_kv_heads * head_dim
        
        q_proj = LinearNode(
            name="q_proj",
            in_features=hidden_dim,
            out_features=hidden_dim,
            use_bias=attn_config.bias
        )
        attention.add_child(q_proj)
        
        if attn_config.mechanism == "mla":
            # MLA: special K/V projections
            mla_latent_dim = attn_config.mla_latent_dim or (hidden_dim // 4)
            
            k_down = LinearNode(name="k_down_proj", in_features=hidden_dim, out_features=mla_latent_dim, use_bias=attn_config.bias)
            v_down = LinearNode(name="v_down_proj", in_features=hidden_dim, out_features=mla_latent_dim, use_bias=attn_config.bias)
            k_up = LinearNode(name="k_up_proj", in_features=mla_latent_dim, out_features=kv_dim, use_bias=attn_config.bias)
            v_up = LinearNode(name="v_up_proj", in_features=mla_latent_dim, out_features=kv_dim, use_bias=attn_config.bias)
            
            attention.add_child(k_down)
            attention.add_child(v_down)
            attention.add_child(k_up)
            attention.add_child(v_up)
            attention.add_param('mla_latent_dim', mla_latent_dim)
            attention.add_param('mechanism', 'mla')
        else:
            k_proj = LinearNode(name="k_proj", in_features=hidden_dim, out_features=kv_dim, use_bias=attn_config.bias)
            v_proj = LinearNode(name="v_proj", in_features=hidden_dim, out_features=kv_dim, use_bias=attn_config.bias)
            attention.add_child(k_proj)
            attention.add_child(v_proj)
        
        out_proj = LinearNode(name="out_proj", in_features=hidden_dim, out_features=hidden_dim, use_bias=attn_config.bias)
        attention.add_child(out_proj)
        
        # Add RoPE if needed
        if use_rope:
            attention.add_param('rope_theta', pos_enc.rope_theta)
            attention.add_param('rope_scaling', pos_enc.rope_scaling)
            attention.add_param('rope_scaling_factor', pos_enc.rope_scaling_factor)
        
        # Add ALiBi if needed
        if use_alibi:
            attention.add_param('alibi_max_positions', attn_config.alibi_max_positions or self.config.embedding.max_position_embeddings)
        
        return attention
    
    def _build_ffn(self, ffn_config: FFNConfig) -> FFNNode:
        """Build FFN module IR."""
        ffn = FFNNode(
            name="ffn",
            hidden_dim=self.config.layer.hidden_dim,
            intermediate_size=ffn_config.intermediate_size,
            activation=ffn_config.activation,
            use_gated=ffn_config.use_gated_activation,
            dropout=ffn_config.dropout,
            use_bias=ffn_config.bias
        )
        
        # Add linear layers as children
        if ffn_config.use_gated_activation:
            gate_proj = LinearNode(name="gate_proj", in_features=self.config.layer.hidden_dim, 
                                  out_features=ffn_config.intermediate_size, use_bias=ffn_config.bias)
            up_proj = LinearNode(name="up_proj", in_features=self.config.layer.hidden_dim,
                                out_features=ffn_config.intermediate_size, use_bias=ffn_config.bias)
            down_proj = LinearNode(name="down_proj", in_features=ffn_config.intermediate_size,
                                  out_features=self.config.layer.hidden_dim, use_bias=ffn_config.bias)
            ffn.add_child(gate_proj)
            ffn.add_child(up_proj)
            ffn.add_child(down_proj)
        else:
            fc1 = LinearNode(name="fc1", in_features=self.config.layer.hidden_dim,
                           out_features=ffn_config.intermediate_size, use_bias=ffn_config.bias)
            fc2 = LinearNode(name="fc2", in_features=ffn_config.intermediate_size,
                           out_features=self.config.layer.hidden_dim, use_bias=ffn_config.bias)
            ffn.add_child(fc1)
            ffn.add_child(fc2)
        
        return ffn
    
    def _build_norm(self, norm_config: LayerNormConfig, name: str) -> NormNode:
        """Build normalization layer IR."""
        norm = NormNode(
            name=name,
            normalized_shape=self.config.layer.hidden_dim,
            eps=norm_config.eps,
            elementwise_affine=norm_config.elementwise_affine,
            norm_type=norm_config.type
        )
        return norm
    
    def _build_output_modules(self) -> List[ModuleNode]:
        """Build output-related modules."""
        modules = []
        
        # Final layer norm (if configured)
        if self.config.final_layer_norm:
            final_norm = self._build_norm(self.config.final_layer_norm, "final_ln")
            modules.append(final_norm)
        
        # LM head (if not tied)
        if not self.config.tie_word_embeddings:
            lm_head = LinearNode(
                name="lm_head",
                in_features=self.config.embedding.embedding_dim,
                out_features=self.config.embedding.vocab_size,
                use_bias=False
            )
            modules.append(lm_head)
        
        return modules
