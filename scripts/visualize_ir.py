#!/usr/bin/env python3
"""Visualize the IR structure of a YamlLLM model."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yamlllm.parser import parse_yaml_config
from yamlllm.ir_builder import IRBuilder
from yamlllm.ir import IRNode, ModuleNode, SequenceNode, NodeType


class IRVisualizer:
    """Visualize IR as ASCII tree or Mermaid diagram."""
    
    def __init__(self, ir):
        self.ir = ir
    
    def to_ascii(self) -> str:
        """Generate ASCII tree representation."""
        lines = []
        lines.append(f"Model: {self.ir.name}")
        lines.append(f"├─ Vocab Size: {self.ir.vocab_size}")
        lines.append(f"├─ Hidden Dim: {self.ir.hidden_dim}")
        lines.append(f"└─ Num Layers: {self.ir.num_layers}")
        lines.append("")
        
        # Embedding modules
        lines.append("Embedding Modules:")
        for i, module in enumerate(self.ir.embedding_modules):
            is_last = i == len(self.ir.embedding_modules) - 1
            prefix = "└──" if is_last else "├──"
            lines.extend(self._render_node(module, prefix, is_last))
        lines.append("")
        
        # Decoder layers
        lines.append(f"Decoder Layers ({len(self.ir.decoder_layers)}):")
        for i, layer in enumerate(self.ir.decoder_layers):
            is_last = i == len(self.ir.decoder_layers) - 1
            prefix = "└──" if is_last else "├──"
            lines.append(f"{prefix} Layer {i}")
            
            indent = "    " if is_last else "│   "
            for j, module in enumerate(layer.modules):
                is_module_last = j == len(layer.modules) - 1
                module_prefix = indent + ("└──" if is_module_last else "├──")
                lines.extend(self._render_node(module, module_prefix, is_module_last, indent=indent))
        lines.append("")
        
        # Output modules
        if self.ir.output_modules:
            lines.append("Output Modules:")
            for i, module in enumerate(self.ir.output_modules):
                is_last = i == len(self.ir.output_modules) - 1
                prefix = "└──" if is_last else "├──"
                lines.extend(self._render_node(module, prefix, is_last))
        
        return "\n".join(lines)
    
    def _render_node(self, node: IRNode, prefix: str, is_last: bool, indent: str = "") -> list[str]:
        """Render a single IR node."""
        lines = []
        
        # Node header
        node_info = f"{node.name} ({node.node_type.value})"
        if hasattr(node, 'params') and node.params:
            # Show key params
            param_str = ", ".join(f"{k}={v}" for k, v in list(node.params.items())[:3])
            if len(node.params) > 3:
                param_str += ", ..."
            node_info += f" [{param_str}]"
        lines.append(f"{prefix} {node_info}")
        
        # Children
        if isinstance(node, ModuleNode) and node.children:
            child_indent = indent + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                is_child_last = i == len(node.children) - 1
                child_prefix = child_indent + ("└──" if is_child_last else "├──")
                lines.extend(self._render_node(child, child_prefix, is_child_last, child_indent))
        
        return lines
    
    def to_mermaid(self) -> str:
        """Generate Mermaid flowchart showing detailed model structure."""
        lines = ["flowchart TD"]
        lines.append("")
        
        # Start
        lines.append("    START([Input IDs])")
        lines.append("    START --> TOKEN_EMB")
        
        # Embeddings
        lines.append("    TOKEN_EMB[Token Embedding]")
        lines.append("    POS_EMB[Position Embedding]")
        lines.append("    TOKEN_EMB --> ADD_EMB")
        lines.append("    POS_EMB --> ADD_EMB")
        lines.append("    ADD_EMB((+)) --> EMB_OUT[Embeddings]")
        lines.append("")
        
        # Decoder layers (show first 2 in detail)
        prev = "EMB_OUT"
        for i, layer in enumerate(self.ir.decoder_layers[:2]):
            layer_prefix = f"L{i}"
            
            # Layer input
            lines.append(f"    {prev} --> {layer_prefix}_IN")
            lines.append(f"    subgraph Layer{i}[Decoder Layer {i}]")
            lines.append(f"        {layer_prefix}_IN[Input]")
            
            # Attention block
            lines.append(f"        {layer_prefix}_IN --> {layer_prefix}_LN1[LayerNorm]")
            lines.append(f"        {layer_prefix}_LN1 --> {layer_prefix}_ATT[Multi-Head Attention]")
            
            # Check for flash attention
            for module in layer.modules:
                if module.node_type == NodeType.ATTENTION:
                    if module.params.get('use_flash', False):
                        lines.append(f"        {layer_prefix}_ATT -.Flash Attention.-> {layer_prefix}_ATT")
            
            lines.append(f"        {layer_prefix}_ATT --> {layer_prefix}_ADD1((+))")
            lines.append(f"        {layer_prefix}_IN --> {layer_prefix}_ADD1")
            
            # FFN block
            lines.append(f"        {layer_prefix}_ADD1 --> {layer_prefix}_LN2[LayerNorm]")
            lines.append(f"        {layer_prefix}_LN2 --> {layer_prefix}_FFN1[Linear]")
            
            # Get activation type
            activation = "GELU"
            for module in layer.modules:
                if module.node_type == NodeType.FFN:
                    activation = module.params.get('activation', 'gelu').upper()
                    break
            
            lines.append(f"        {layer_prefix}_FFN1 --> {layer_prefix}_ACT[{activation}]")
            lines.append(f"        {layer_prefix}_ACT --> {layer_prefix}_FFN2[Linear]")
            lines.append(f"        {layer_prefix}_FFN2 --> {layer_prefix}_ADD2((+))")
            lines.append(f"        {layer_prefix}_ADD1 --> {layer_prefix}_ADD2")
            lines.append(f"        {layer_prefix}_ADD2 --> {layer_prefix}_OUT[Output]")
            lines.append(f"    end")
            lines.append("")
            
            prev = f"{layer_prefix}_OUT"
        
        # Show remaining layers as collapsed
        if len(self.ir.decoder_layers) > 2:
            lines.append(f"    {prev} --> MORE[... {len(self.ir.decoder_layers) - 2} more layers ...]")
            prev = "MORE"
        
        # Output
        if self.ir.use_final_norm:
            lines.append(f"    {prev} --> FINAL_NORM[Final LayerNorm]")
            prev = "FINAL_NORM"
        
        lines.append(f"    {prev} --> LM_HEAD[LM Head: Linear]")
        lines.append(f"    LM_HEAD --> END([Logits])")
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Visualize YamlLLM IR structure")
    parser.add_argument("config", type=str, help="Path to YAML config file")
    parser.add_argument("--format", type=str, 
                       choices=["ascii", "mermaid"],
                       default="ascii",
                       help="Output format (default: ascii)")
    parser.add_argument("-o", "--output", type=str, help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    try:
        # Build IR
        config = parse_yaml_config(args.config)
        builder = IRBuilder(config)
        ir = builder.build()
        
        # Validate
        errors = ir.validate()
        if errors:
            print(f"IR Validation errors:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)
        
        # Visualize
        visualizer = IRVisualizer(ir)
        
        if args.format == "ascii":
            output = visualizer.to_ascii()
        else:
            output = visualizer.to_mermaid()
        
        # Write output
        if args.output:
            Path(args.output).write_text(output)
            print(f"Visualization written to {args.output}", file=sys.stderr)
        else:
            print(output)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
