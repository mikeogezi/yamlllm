import pytest
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from yamlllm.parser import parse_dict_config, parse_yaml_config
from yamlllm.codegen.pytorch import generate_pytorch_code

# Sample valid config
VALID_CONFIG = {
    "name": "TestModel",
    "embedding": {
        "vocab_size": 1000,
        "max_position_embeddings": 128,
        "embedding_dim": 64,
        "positional_encoding": {"type": "learned"}
    },
    "num_layers": 2,
    "layer": {
        "hidden_dim": 64,
        "attention": {"num_heads": 4},
        "ffn": {"intermediate_size": 256}
    }
}

def test_schema_validation():
    """Test that invalid configs raise errors."""
    # Invalid: embedding dim mismatch
    invalid_config = VALID_CONFIG.copy()
    invalid_config["embedding"] = VALID_CONFIG["embedding"].copy()
    invalid_config["embedding"]["embedding_dim"] = 32  # Mismatch with hidden_dim 64
    
    with pytest.raises(ValueError, match="Configuration errors"):
        parse_dict_config(invalid_config)

def test_rope_scaling_config():
    """Test parsing of RoPE scaling parameters."""
    config_data = VALID_CONFIG.copy()
    config_data["embedding"] = VALID_CONFIG["embedding"].copy()
    config_data["embedding"]["positional_encoding"] = {
        "type": "rope",
        "rope_scaling": "linear",
        "rope_scaling_factor": 2.0
    }
    
    config = parse_dict_config(config_data)
    assert config.embedding.positional_encoding.rope_scaling == "linear"
    assert config.embedding.positional_encoding.rope_scaling_factor == 2.0

def test_flash_attention_config():
    """Test parsing of Flash Attention flag."""
    config_data = VALID_CONFIG.copy()
    config_data["layer"] = VALID_CONFIG["layer"].copy()
    config_data["layer"]["attention"] = VALID_CONFIG["layer"]["attention"].copy()
    config_data["layer"]["attention"]["use_flash_attention"] = True
    
    config = parse_dict_config(config_data)
    assert config.layer.attention.use_flash_attention is True

def test_codegen_execution():
    """Test that generated code can be executed and creates a valid model."""
    config = parse_dict_config(VALID_CONFIG)
    code = generate_pytorch_code(config)
    
    # Execute generated code in a separate namespace
    namespace = {}
    exec(code, namespace)
    
    # Check if model class exists
    assert "TestModel" in namespace
    ModelClass = namespace["TestModel"]
    
    # Instantiate model
    model = ModelClass()
    assert isinstance(model, nn.Module)
    
    # Run forward pass
    input_ids = torch.randint(0, 1000, (1, 16))
    output = model(input_ids)
    assert output.shape == (1, 16, 1000)

def test_rope_scaling_codegen():
    """Test that RoPE scaling code is generated correctly."""
    config_data = VALID_CONFIG.copy()
    config_data["embedding"] = VALID_CONFIG["embedding"].copy()
    config_data["embedding"]["positional_encoding"] = {
        "type": "rope",
        "rope_scaling": "linear",
        "rope_scaling_factor": 2.0
    }
    config = parse_dict_config(config_data)
    code = generate_pytorch_code(config)
    
    assert "scaling_type='linear'" in code
    assert "scaling_factor=2.0" in code
    assert "position_ids = position_ids.float() / self.scaling_factor" in code

def test_flash_attention_codegen():
    """Test that Flash Attention code is generated correctly."""
    config_data = VALID_CONFIG.copy()
    config_data["layer"] = VALID_CONFIG["layer"].copy()
    config_data["layer"]["attention"] = VALID_CONFIG["layer"]["attention"].copy()
    config_data["layer"]["attention"]["use_flash_attention"] = True
    
    config = parse_dict_config(config_data)
    code = generate_pytorch_code(config)
    
    assert "F.scaled_dot_product_attention" in code


def test_ir_generation():
    """Test that IR can be built from config."""
    from yamlllm.ir_builder import IRBuilder
    
    config = parse_dict_config(VALID_CONFIG)
    builder = IRBuilder(config)
    ir = builder.build()
    
    # Validate IR
    assert ir.name == "TestModel"
    assert ir.vocab_size == 1000
    assert ir.hidden_dim == 64
    assert ir.num_layers == 2
    assert len(ir.decoder_layers) == 2
    
    # Check for errors
    errors = ir.validate()
    assert len(errors) == 0


def test_jax_codegen():
    """Test that JAX/Flax code can be generated."""
    from yamlllm.ir_builder import IRBuilder
    from yamlllm.codegen.jax import generate_jax_code
    
    config = parse_dict_config(VALID_CONFIG)
    builder = IRBuilder(config)
    ir = builder.build()
    
    # Generate JAX code
    code = generate_jax_code(ir)
    
    # Check for Flax imports
    assert "import jax" in code
    assert "from flax import linen as nn" in code
    
    # Check class definition
    assert "class TestModel(nn.Module)" in code
    assert "@nn.compact" in code
    
    # Check forward pass
    assert "def __call__" in code
    assert "nn.Embed" in code
    assert "nn.Dense" in code
