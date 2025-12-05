#!/usr/bin/env python3
"""Train a simple character-level language model on Shakespeare."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from tqdm import tqdm
from yamlllm.parser import parse_yaml_config
from yamlllm.codegen.pytorch import generate_pytorch_code


def load_data(data_path="data/shakespeare.txt"):
    """Load and prepare Shakespeare dataset."""
    text = Path(data_path).read_text()
    
    # Character-level tokenization
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    
    # Create mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Encode text
    data = torch.tensor([char_to_idx[ch] for ch in text], dtype=torch.long)
    
    return data, vocab_size, char_to_idx, idx_to_char


def get_batch(data, batch_size, seq_len):
    """Get a random batch of sequences."""
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y


def generate_text(model, idx_to_char, char_to_idx, seed_text="To be", max_length=100):
    """Generate text from the model."""
    device = next(model.parameters()).device  # Get model's device
    model.eval()
    
    # Encode seed text
    input_ids = torch.tensor([[char_to_idx[ch] for ch in seed_text]], dtype=torch.long).to(device)
    
    generated = seed_text
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)
            probs = torch.softmax(logits[0, -1], dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            next_char = idx_to_char[next_idx]
            generated += next_char
            
            # Append to input
            input_ids = torch.cat([input_ids, torch.tensor([[next_idx]], device=device)], dim=1)
            
            # Keep only last seq_len tokens
            if input_ids.shape[1] > 64:
                input_ids = input_ids[:, -64:]
    
    model.train()  # Switch back to training mode
    return generated


def main():
    parser = argparse.ArgumentParser(description="Train a simple character-level model")
    parser.add_argument("--config", type=str, default="examples/tiny.yaml", help="Model config")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--data-path", type=str, default="data/shakespeare.txt", help="Path to dataset")
    parser.add_argument("-b", "--backend", type=str, choices=["pytorch", "jax"], default="pytorch",
                       help="Backend to use (pytorch or jax)")

    args = parser.parse_args()
    
    print(f"Backend: {args.backend.upper()}")
    
    # Load data
    print("\nLoading Shakespeare dataset...")
    data, vocab_size, char_to_idx, idx_to_char = load_data(args.data_path)
    print(f"Vocabulary size: {vocab_size} characters")
    print(f"Dataset size: {len(data)} characters")
    
    # Parse config and adjust vocab size
    config = parse_yaml_config(args.config)
    config.embedding.vocab_size = vocab_size
    
    if args.backend == "pytorch":
        train_pytorch(config, data, vocab_size, char_to_idx, idx_to_char, args)
    else:
        train_jax(config, data, vocab_size, char_to_idx, idx_to_char, args)


def train_pytorch(config, data, vocab_size, char_to_idx, idx_to_char, args):
    """Train using PyTorch backend."""
    import torch
    import torch.nn as nn
    from yamlllm.codegen.pytorch import generate_pytorch_code
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected, falling back to CPU")
    
    # Generate and execute model code
    print(f"\nGenerating model: {config.name}")
    code = generate_pytorch_code(config)
    exec(code, globals())
    
    # Instantiate model and move to device
    model = globals()[config.name]()
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop with progress bar
    print(f"\nTraining for {args.steps} steps...")
    model.train()
    
    pbar = tqdm(range(args.steps), desc="Training", ncols=100)
    for step in pbar:
        # Get batch and move to device
        x, y = get_batch(data, args.batch_size, args.seq_len)
        x, y = x.to(device), y.to(device)
        
        # Forward pass
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Generate sample every 100 steps
        if (step + 1) % 100 == 0:
            pbar.write(f"\n{'='*60}")
            pbar.write(f"Step {step+1}/{args.steps} | Loss: {loss.item():.4f}")
            sample = generate_text(model, idx_to_char, char_to_idx, max_length=100)
            pbar.write(f"Sample: {sample}")
            pbar.write(f"{'='*60}\n")
    
    pbar.close()
    
    print(f"\n✅ Training complete! Final loss: {loss.item():.4f}")
    print("\n=== Final Generation ===")
    final_sample = generate_text(model, idx_to_char, char_to_idx, seed_text="To be, or not to be", max_length=200)
    print(final_sample)


def train_jax(config, data, vocab_size, char_to_idx, idx_to_char, args):
    """Train using JAX/Flax backend."""
    try:
        import jax
        import jax.numpy as jnp
        import optax
        from yamlllm.ir_builder import IRBuilder
        from yamlllm.codegen.jax import generate_jax_code
    except ImportError as e:
        print(f"❌ Error: JAX/Flax not installed. Install with: pip install jax[cuda12] flax optax")
        print(f"   {e}")
        sys.exit(1)
    
    print(f"Using device: {jax.devices()[0]}")
    
    # Generate and execute model code
    print(f"\nGenerating model: {config.name}")
    builder = IRBuilder(config)
    ir = builder.build()
    code = generate_jax_code(ir)
    exec(code, globals())
    
    # Instantiate model
    model = globals()[config.name]()
    
    # Initialize parameters
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, args.seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy_input)
    
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {param_count:,}")
    
    # Training setup
    optimizer = optax.adamw(args.lr)
    opt_state = optimizer.init(params)
    
    # Convert data to JAX
    data_jax = jnp.array(data.numpy())
    
    # JAX text generation function (fixed-length for fast JIT compilation)
    @jax.jit
    def generate_step(params, input_ids):
        """Single generation step (JIT-compiled)."""
        logits = model.apply(params, input_ids, deterministic=True)
        next_token = jnp.argmax(logits[:, -1, :], axis=-1)
        return next_token
    
    def generate_text_jax(params, seed_text, max_length=100):
        """Generate text using JAX model with fixed-length generation."""
        # Encode seed (pad to fixed length)
        seed_ids = [char_to_idx[ch] for ch in seed_text]
        seq_len = 64  # Fixed context window
        
        # Pad/truncate to seq_len
        if len(seed_ids) < seq_len:
            input_ids = jnp.array([[0] * (seq_len - len(seed_ids)) + seed_ids])
        else:
            input_ids = jnp.array([seed_ids[-seq_len:]])
        
        generated = seed_text
        for _ in range(max_length):
            # Generate next token (JIT-compiled, fast!)
            next_idx = int(generate_step(params, input_ids)[0])
            next_char = idx_to_char[next_idx]
            generated += next_char
            
            # Shift and append (keep fixed length)
            input_ids = jnp.concatenate([
                input_ids[:, 1:],
                jnp.array([[next_idx]])
            ], axis=1)
        
        return generated
    
    # Training loop
    print(f"\nTraining for {args.steps} steps...")
    
    @jax.jit
    def train_step(params, opt_state, x, y):
        def loss_fn(params):
            logits = model.apply(params, x)
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits.reshape(-1, vocab_size), y.reshape(-1)
            ).mean()
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    pbar = tqdm(range(args.steps), desc="Training", ncols=100)
    for step in pbar:
        # Get batch
        x, y = get_batch(data, args.batch_size, args.seq_len)
        x_jax = jnp.array(x.numpy())
        y_jax = jnp.array(y.numpy())
        
        # Training step
        params, opt_state, loss = train_step(params, opt_state, x_jax, y_jax)
        
        # Update progress bar
        pbar.set_postfix({"loss": f"{float(loss):.4f}"})
        
        # Generate sample every 100 steps
        if (step + 1) % 100 == 0:
            pbar.write(f"\n{'='*60}")
            pbar.write(f"Step {step+1}/{args.steps} | Loss: {float(loss):.4f}")
            sample = generate_text_jax(params, "To be", max_length=100)
            pbar.write(f"Sample: {sample}")
            pbar.write(f"{'='*60}\n")
    
    pbar.close()
    
    print(f"\n✅ Training complete! Final loss: {float(loss):.4f}")
    print("\n=== Final Generation ===")
    final_sample = generate_text_jax(params, "To be, or not to be", max_length=200)
    print(final_sample)


if __name__ == "__main__":
    main()
