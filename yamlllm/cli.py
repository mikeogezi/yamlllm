"""Command-line interface for YamlLLM compiler."""

import argparse
import sys
from pathlib import Path

from .parser import parse_yaml_config
from .codegen.pytorch import generate_pytorch_code


def main():
    parser = argparse.ArgumentParser(
        description="YamlLLM: Compile YAML transformer definitions to PyTorch code"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file path (default: print to stdout)",
    )
    parser.add_argument(
        "-b", "--backend",
        type=str,
        choices=["pytorch", "jax"],
        default="pytorch",
        help="Target backend (default: pytorch)",
    )
    
    args = parser.parse_args()
    
    try:
        # Parse YAML
        config = parse_yaml_config(args.input)
        
        # Generate code
        if args.backend == "pytorch":
            code = generate_pytorch_code(config)
        elif args.backend == "jax":
            from .ir_builder import IRBuilder
            from .codegen.jax import generate_jax_code
            
            # Build IR and generate JAX code
            ir = IRBuilder(config).build()
            code = generate_jax_code(ir)
        else:
            print(f"Error: {args.backend} backend not supported", file=sys.stderr)
            sys.exit(1)
        
        # Output
        if args.output:
            output_path = Path(args.output)
            output_path.write_text(code)
            print(f"Generated code written to {output_path}", file=sys.stderr)
        else:
            print(code)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

