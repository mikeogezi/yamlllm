"""Integration tests for the training script.

These tests validate that the train.py script correctly uses the IR pipeline
and can successfully generate and instantiate models from config files.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_train_pytorch_uses_ir_builder():
    """Ensure train_pytorch correctly builds IR before code generation."""
    import torch
    from yamlllm.parser import parse_yaml_config
    from yamlllm.ir_builder import IRBuilder
    from yamlllm.codegen.pytorch import generate_pytorch_code
    
    # Load tiny.yaml config
    config_path = Path(__file__).parent.parent / "examples" / "tiny.yaml"
    config = parse_yaml_config(str(config_path))
    
    # This is what train.py should do - build IR first
    builder = IRBuilder(config)
    ir = builder.build()
    
    # Verify IR has required attributes that code generator expects
    assert hasattr(ir, 'decoder_layers'), "IR must have decoder_layers attribute"
    assert hasattr(ir, 'embedding_modules'), "IR must have embedding_modules attribute"
    assert len(ir.decoder_layers) > 0, "IR should have at least one decoder layer"
    
    # Generate and execute code
    code = generate_pytorch_code(ir)
    namespace = {}
    exec(code, namespace)
    
    # Verify model can be instantiated
    assert config.name in namespace, f"Model class {config.name} not found in generated code"
    model = namespace[config.name]()
    assert isinstance(model, torch.nn.Module)
    
    # Run a forward pass
    input_ids = torch.randint(0, 100, (1, 16))
    output = model(input_ids)
    assert output.shape[0] == 1
    assert output.shape[1] == 16


def test_config_to_ir_conversion_preserves_model_name():
    """Verify model name is preserved through config -> IR conversion."""
    from yamlllm.parser import parse_yaml_config
    from yamlllm.ir_builder import IRBuilder
    
    config_path = Path(__file__).parent.parent / "examples" / "tiny.yaml"
    config = parse_yaml_config(str(config_path))
    
    ir = IRBuilder(config).build()
    
    assert ir.name == config.name, "Model name should be preserved in IR"


def test_all_example_configs_generate_valid_models():
    """Test that all example YAML configs can generate valid PyTorch models."""
    import torch
    from yamlllm.parser import parse_yaml_config
    from yamlllm.ir_builder import IRBuilder
    from yamlllm.codegen.pytorch import generate_pytorch_code
    
    examples_dir = Path(__file__).parent.parent / "examples"
    yaml_files = list(examples_dir.glob("*.yaml"))
    
    assert len(yaml_files) > 0, "Should have at least one example YAML file"
    
    for yaml_file in yaml_files:
        # Parse config
        config = parse_yaml_config(str(yaml_file))
        
        # Build IR (this is the critical step that was missing in train.py)
        ir = IRBuilder(config).build()
        
        # Generate code
        code = generate_pytorch_code(ir)
        
        # Execute and instantiate
        namespace = {}
        exec(code, namespace)
        
        model = namespace[config.name]()
        assert isinstance(model, torch.nn.Module), f"Failed for {yaml_file.name}"
        
        # Quick forward pass sanity check
        input_ids = torch.randint(0, 100, (1, 8))
        output = model(input_ids)
        assert output.dim() == 3, f"Expected 3D output for {yaml_file.name}"
