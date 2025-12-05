# YamlLLM 🚀

**A declarative DSL for transformer models with multi-backend code generation**

YamlLLM demonstrates key programming language principles through a domain-specific language (DSL) for defining transformer architectures. The system uses an intermediate representation (IR) to decouple the high-level YAML specification from backend-specific code generation.

```yaml
# Declarative model specification
name: TinyModel
embedding:
  vocab_size: 1000
  max_position_embeddings: 128
  embedding_dim: 256
  padding_idx: 0
  dropout: 0.0

num_layers: 2

layer:
  hidden_dim: 256
  attention:
    num_heads: 4
    dropout: 0.0
    bias: false
  ffn:
    intermediate_size: 512
    activation: relu
    dropout: 0.0
    bias: false
  layer_norm:
    eps: 1e-6
  residual_dropout: 0.0

tie_word_embeddings: false
```

**Key PL Concepts Demonstrated**:
- 📝 **Domain-Specific Language**: High-level declarative syntax for transformers
- 🔄 **Multi-Backend Compilation**: Single source → PyTorch + JAX/Flax
- 🏗️ **Intermediate Representation**: Decouples frontend from backend
- ✅ **Type Safety**: Schema validation with semantic checks
- 🎯 **Code Generation**: Template-based + IR traversal

> **Academic Details**: See [`docs/REPORT.md`](docs/REPORT.md) for full PL analysis, IR design, and implementation.

## 🏗️ Architecture

YamlLLM uses a three-layer architecture inspired by traditional compiler design:

**Pipeline**: YAML Schema → Validated Config → **IR** → Backend Renderer → Generated Code

### In more detail

```
┌──────────┐     ┌────────┐     ┌──────────┐     ┌────────────┐     ┌──────┐
│   YAML   │────▶│ Parser │────▶│  Schema  │────▶│ IR Builder │────▶│  IR  │
└──────────┘     └────────┘     └──────────┘     └────────────┘     └──────┘
                                                                        │
                                                        ┌───────────────┴──────────────┐
                                                        ▼                              ▼
                                                  ┌──────────┐                  ┌──────────┐
                                                  │ PyTorch  │                  │   JAX    │
                                                  │ Renderer │                  │ Renderer │
                                                  └──────────┘                  └──────────┘
```

The IR layer provides:
- **Abstraction**: Backend-agnostic model representation
- **Extensibility**: Easy to add new backends (e.g., ONNX, TensorFlow)
- **Optimization**: IR transformations before code generation

## 🚀 Quick Start

### 1. Define Your Model

Create a YAML config ([example](examples/tiny.yaml)):

```yaml
name: TinyModel
embedding:
  vocab_size: 1000
  max_position_embeddings: 128
  embedding_dim: 256
  positional_encoding: {type: learned}
layer:
  hidden_dim: 256
  attention: {num_heads: 4}
  ffn: {intermediate_size: 512, activation: relu}
  layer_norm: {type: layernorm, eps: 1e-6}
num_layers: 2
tie_word_embeddings: false
```

### 2. Generate Code

**PyTorch** ([example output](outputs/tiny_pytorch.py)):
```bash
python -m yamlllm.cli examples/tiny.yaml -o model.py
```

**JAX/Flax** ([example output](outputs/tiny_jax.py)):
```bash
python -m yamlllm.cli examples/tiny.yaml -b jax -o model_jax.py
```

**Both generate from the same YAML** - demonstrating multi-backend compilation!

### 3. Use Generated Models

**PyTorch**:
```python
from model import TinyModel
import torch

model = TinyModel()
input_ids = torch.randint(0, 1000, (1, 128))
logits = model(input_ids)  # (1, 128, 1000)
```

**JAX/Flax**:
```python
import jax
import jax.numpy as jnp
from model_jax import TinyModel

model = TinyModel()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 128), dtype=jnp.int32))
logits = model.apply(params, jnp.ones((1, 128), dtype=jnp.int32))  # (1, 128, 1000)
```

4. **Serialized model architecture**
```python
TinyModel(
  (token_embedding): Embedding(1000, 256, padding_idx=0)
  (position_embedding): Embedding(128, 256)
  (embedding_dropout): Dropout(p=0.0, inplace=False)
  (layers): ModuleList(
    (0-1): 2 x DecoderLayer(
      (attention): CausalSelfAttention(
        (q_proj): Linear(in_features=256, out_features=256, bias=False)
        (k_proj): Linear(in_features=256, out_features=256, bias=False)
        (v_proj): Linear(in_features=256, out_features=256, bias=False)
        (out_proj): Linear(in_features=256, out_features=256, bias=False)
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (ffn): FeedForward(
        (fc1): Linear(in_features=256, out_features=512, bias=False)
        (fc2): Linear(in_features=512, out_features=256, bias=False)
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (ln1): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
      (ln2): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
      (residual_dropout): Dropout(p=0.0, inplace=False)
    )
  )
  (lm_head): Linear(in_features=256, out_features=1000, bias=False)
))
```

5. **Model Diagram**

![Model Diagram](assets/diagram.png)

## 🛠️ CLI Usage

```bash
python -m yamlllm.cli [-h] [-o OUTPUT] [--backend {pytorch,jax}] input

positional arguments:
  input                 Path to YAML configuration file

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file path (default: print to stdout)
  --backend {pytorch,jax}
                        Target backend (default: pytorch)
```

## 📚 Supported Architectures

YamlLLM supports defining architectures similar to:

| Architecture | YAML Example |
|--------------|--------------|
| **LLaMA** | `examples/llama-style.yaml` |
| **GPT-2** | `examples/gpt-mini.yaml` |
| **Mistral** | `examples/gqa-model.yaml` |
| **DeepSeek** | `examples/mla-model.yaml` |
| **Bloom** | `examples/alibi-model.yaml` |

See [docs/FEATURES.md](docs/FEATURES.md) for full configuration options.

## 🔍 IR Visualization

Visualize the intermediate representation structure (see a Mermaid diagram [example](outputs/diagram.mmd)):

```bash
# ASCII tree view
python scripts/visualize_ir.py examples/tiny.yaml

# Mermaid diagram
python scripts/visualize_ir.py examples/tiny.yaml --format mermaid -o diagram.mmd
```

Example output for ASCII tree view:
```
Model: TinyModel
├─ Vocab Size: 1000
├─ Hidden Dim: 256
└─ Num Layers: 2

Embedding Modules:
├── token_embedding (module) [num_embeddings=1000, embedding_dim=256, padding_idx=0]
└── position_embedding (module) [num_embeddings=128, embedding_dim=256, padding_idx=None]

Decoder Layers (2):
├── Layer 0
│   ├── attention (module) [num_heads=4, head_dim=64, hidden_dim=256, ...]
│   │   ├── q_proj (module) [in_features=256, out_features=256, bias=False]
│   │   ├── k_proj (module) [in_features=256, out_features=256, bias=False]
│   │   ├── v_proj (module) [in_features=256, out_features=256, bias=False]
│   │   └── out_proj (module) [in_features=256, out_features=256, bias=False]
│   ├── ln1 (module) [normalized_shape=256, eps=1e-6, elementwise_affine=True]
│   ├── ffn (module) [hidden_dim=256, intermediate_size=512, activation=relu, ...]
│   │   ├── fc1 (module) [in_features=256, out_features=512, bias=False]
│   │   └── fc2 (module) [in_features=512, out_features=256, bias=False]
│   └── ln2 (module) [normalized_shape=256, eps=1e-6, elementwise_affine=True]
└── Layer 1
    ├── attention (module) [num_heads=4, head_dim=64, hidden_dim=256, ...]
    │   ├── q_proj (module) [in_features=256, out_features=256, bias=False]
    │   ├── k_proj (module) [in_features=256, out_features=256, bias=False]
    │   ├── v_proj (module) [in_features=256, out_features=256, bias=False]
    │   └── out_proj (module) [in_features=256, out_features=256, bias=False]
    ├── ln1 (module) [normalized_shape=256, eps=1e-6, elementwise_affine=True]
    ├── ffn (module) [hidden_dim=256, intermediate_size=512, activation=relu, ...]
    │   ├── fc1 (module) [in_features=256, out_features=512, bias=False]
    │   └── fc2 (module) [in_features=512, out_features=256, bias=False]
    └── ln2 (module) [normalized_shape=256, eps=1e-6, elementwise_affine=True]

Output Modules:
└── lm_head (module) [in_features=256, out_features=1000, bias=False]
```

## 🎭 Train a Character-Level Shakespeare Model

See your model learn to write like Shakespeare! ([full training log](outputs/train_pytorch.log))

```bash
python scripts/train.py --config examples/tiny.yaml --steps 1000
```

**Sample output after training (for 3000 steps)**:
```
=== Final Generation ===
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No m
```

The model learns character-level patterns from Shakespeare and generates new text! 🎭

## 🧪 Testing

Run the test suite to ensure everything is working:

```bash
python -m pytest tests/
```

Project carried out with ❤️ and the help of Cursor and Antigravity.