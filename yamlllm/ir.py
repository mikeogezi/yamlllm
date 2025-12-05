"""Intermediate Representation (IR) for YamlLLM models.

This module defines an IR layer between the high-level schema and backend code generators.
The IR provides a backend-agnostic representation of the model structure.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Literal
from enum import Enum


class NodeType(Enum):
    """Types of IR nodes."""
    MODULE = "module"
    LINEAR = "linear"
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    FFN = "ffn"
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"
    ROPE = "rope"
    DROPOUT = "dropout"
    SEQUENCE = "sequence"


@dataclass
class IRNode:
    """Base class for all IR nodes."""
    name: str
    node_type: NodeType = NodeType.MODULE  # Default, overridden by subclasses
    location: Optional[str] = None  # Source location for error messages
    
    def __post_init__(self):
        """Validate node after initialization."""
        if not self.name:
            raise ValueError("Node name cannot be empty")


@dataclass
class ParamSpec:
    """Parameter specification for a module."""
    name: str
    value: Any
    dtype: Optional[str] = None  # For type hints in generated code


@dataclass
class ModuleNode(IRNode):
    """Represents a neural network module (e.g., Layer, Attention, FFN)."""
    params: Dict[str, Any] = field(default_factory=dict)
    children: List[IRNode] = field(default_factory=list)
    init_stmts: List[str] = field(default_factory=list)  # Initialization statements
    forward_deps: List[str] = field(default_factory=list)  # Dependencies in forward pass
    
    def add_child(self, child: IRNode):
        """Add a child node."""
        self.children.append(child)
    
    def add_param(self, name: str, value: Any):
        """Add a parameter."""
        self.params[name] = value
    
    def get_param(self, name: str, default: Any = None) -> Any:
        """Get a parameter value."""
        return self.params.get(name, default)


@dataclass
class LinearNode(ModuleNode):
    """Linear/Dense layer node."""
    in_features: int = 0
    out_features: int = 0
    use_bias: bool = True
    
    def __post_init__(self):
        self.node_type = NodeType.LINEAR
        super().__post_init__()
        self.params.update({
            'in_features': self.in_features,
            'out_features': self.out_features,
            'bias': self.use_bias
        })


@dataclass
class EmbeddingNode(ModuleNode):
    """Embedding layer node."""
    num_embeddings: int = 0
    embedding_dim: int = 0
    padding_idx: Optional[int] = None
    
    def __post_init__(self):
        self.node_type = NodeType.EMBEDDING
        super().__post_init__()
        self.params.update({
            'num_embeddings': self.num_embeddings,
            'embedding_dim': self.embedding_dim,
            'padding_idx': self.padding_idx
        })


@dataclass
class AttentionNode(ModuleNode):
    """Multi-head attention node."""
    num_heads: int = 0
    head_dim: int = 0
    hidden_dim: int = 0
    num_kv_heads: int = 0
    use_flash: bool = False
    use_rope: bool = False
    use_alibi: bool = False
    use_gqa: bool = False
    dropout: float = 0.0
    
    def __post_init__(self):
        self.node_type = NodeType.ATTENTION
        super().__post_init__()
        self.params.update({
            'num_heads': self.num_heads,
            'head_dim': self.head_dim,
            'hidden_dim': self.hidden_dim,
            'num_kv_heads': self.num_kv_heads,
            'use_flash': self.use_flash,
            'use_rope': self.use_rope,
            'use_alibi': self.use_alibi,
            'dropout': self.dropout
        })


@dataclass
class FFNNode(ModuleNode):
    """Feed-forward network node."""
    hidden_dim: int = 0
    intermediate_size: int = 0
    activation: str = "gelu"
    use_gated: bool = False
    dropout: float = 0.0
    use_bias: bool = True
    
    def __post_init__(self):
        self.node_type = NodeType.FFN
        super().__post_init__()
        self.params.update({
            'hidden_dim': self.hidden_dim,
            'intermediate_size': self.intermediate_size,
            'activation': self.activation,
            'use_gated': self.use_gated,
            'dropout': self.dropout,
            'use_bias': self.use_bias
        })


@dataclass
class NormNode(ModuleNode):
    """Normalization layer node (LayerNorm or RMSNorm)."""
    normalized_shape: int = 0
    eps: float = 1e-5
    elementwise_affine: bool = True
    norm_type: Literal["layernorm", "rmsnorm"] = "layernorm"
    
    def __post_init__(self):
        self.node_type = NodeType.RMS_NORM if self.norm_type == "rmsnorm" else NodeType.LAYER_NORM
        super().__post_init__()
        self.params.update({
            'normalized_shape': self.normalized_shape,
            'eps': self.eps,
            'elementwise_affine': self.elementwise_affine
        })


@dataclass
class SequenceNode(IRNode):
    """Sequential composition of modules."""
    modules: List[IRNode] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.node_type = NodeType.SEQUENCE
        super().__post_init__()
    
    def add_module(self, module: IRNode):
        """Add a module to the sequence."""
        self.modules.append(module)


@dataclass
class ModelIR:
    """Complete model intermediate representation."""
    name: str
    vocab_size: int
    hidden_dim: int
    num_layers: int
    
    # Top-level modules
    embedding_modules: List[IRNode] = field(default_factory=list)
    decoder_layers: List[SequenceNode] = field(default_factory=list)
    output_modules: List[IRNode] = field(default_factory=list)
    
    # Metadata
    max_position_embeddings: int = 1024
    tie_embeddings: bool = True
    use_final_norm: bool = False
    
    # Forward pass dependency graph
    forward_graph: Dict[str, List[str]] = field(default_factory=dict)
    
    def add_embedding_module(self, module: IRNode):
        """Add an embedding module."""
        self.embedding_modules.append(module)
    
    def add_decoder_layer(self, layer: SequenceNode):
        """Add a decoder layer."""
        self.decoder_layers.append(layer)
    
    def add_output_module(self, module: IRNode):
        """Add an output module."""
        self.output_modules.append(module)
    
    def validate(self) -> List[str]:
        """Validate the IR and return list of errors."""
        errors = []
        
        if not self.name:
            errors.append("Model name cannot be empty")
        
        if self.vocab_size <= 0:
            errors.append(f"Invalid vocab_size: {self.vocab_size}")
        
        if self.hidden_dim <= 0:
            errors.append(f"Invalid hidden_dim: {self.hidden_dim}")
        
        if self.num_layers <= 0:
            errors.append(f"Invalid num_layers: {self.num_layers}")
        
        if len(self.decoder_layers) != self.num_layers:
            errors.append(f"Expected {self.num_layers} decoder layers, got {len(self.decoder_layers)}")
        
        return errors
