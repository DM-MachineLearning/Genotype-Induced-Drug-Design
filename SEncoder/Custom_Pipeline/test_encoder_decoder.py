"""
Simple test script for SMILES Encoder and Decoder

This script demonstrates a complete forward pass through the encoder and decoder
with sample input tokens and shows the output shapes and values.
"""

import torch
import torch.nn.functional as F
from smiles_encoder import SMILESEncoder
from smiles_decoder import SMILESDecoder


def main():
    print("=" * 60)
    print("SMILES Encoder-Decoder Test Script")
    print("=" * 60)
    
    # ==================== Configuration ====================
    vocab_size = 45          # SMILES vocabulary size
    embedding_dim = 512      # Model dimension
    latent_size = 512        # Latent space dimension
    n_layers = 8            # Number of encoder/decoder layers
    max_len = 122           # Maximum sequence length
    batch_size = 4          # Number of samples in batch
    seq_length = 30         # Sequence length for test
    
    print(f"\nConfiguration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Latent size: {latent_size}")
    print(f"  Number of layers: {n_layers}")
    print(f"  Max sequence length: {max_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Test sequence length: {seq_length}")
    
    # ==================== Create Models ====================
    print(f"\n{'='*60}")
    print("Creating Encoder and Decoder...")
    print("=" * 60)
    
    encoder = SMILESEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        latent_size=latent_size,
        max_len=max_len,
        dropout=0.0
    )
    
    decoder = SMILESDecoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        latent_size=latent_size,
        max_len=max_len,
        dropout=0.0
    )
    
    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    total_params = encoder_params + decoder_params
    
    print(f"\nModel Parameters:")
    print(f"  Encoder: {encoder_params:,} parameters")
    print(f"  Decoder: {decoder_params:,} parameters")
    print(f"  Total: {total_params:,} parameters")
    
    # ==================== Create Input Tokens ====================
    print(f"\n{'='*60}")
    print("Creating Input Tokens...")
    print("=" * 60)
    
    # Create sample input tokens (simulating tokenized SMILES)
    # Token 0 = padding, Token 1 = start, Token 2 = end
    # For this test, we'll use tokens 3-45 (actual SMILES tokens)
    input_tokens = torch.randint(3, vocab_size, (batch_size, seq_length))
    
    # Add start and end tokens to make it more realistic
    input_tokens = torch.cat([
        torch.ones(batch_size, 1, dtype=torch.long),  # Start token (1)
        input_tokens,
        torch.full((batch_size, 1), 2, dtype=torch.long)  # End token (2)
    ], dim=1)
    
    # Pad sequences to max_len for consistency
    if input_tokens.shape[1] < max_len:
        padding = torch.zeros(batch_size, max_len - input_tokens.shape[1], dtype=torch.long)
        input_tokens = torch.cat([input_tokens, padding], dim=1)
    else:
        input_tokens = input_tokens[:, :max_len]
    
    print(f"\nInput Tokens:")
    print(f"  Shape: {input_tokens.shape}")
    print(f"  First sample (first 20 tokens): {input_tokens[0, :20].tolist()}")
    print(f"  Token value range: [{input_tokens.min().item()}, {input_tokens.max().item()}]")
    
    # ==================== Encode ====================
    print(f"\n{'='*60}")
    print("Encoding Input Tokens...")
    print("=" * 60)
    
    encoder.eval()
    with torch.no_grad():
        mu, var = encoder(input_tokens)
        
    print(f"\nEncoder Output:")
    print(f"  Mu (mean) shape: {mu.shape}")
    print(f"  Var (variance) shape: {var.shape}")
    print(f"  Mu stats:")
    print(f"    Mean: {mu.mean().item():.4f}")
    print(f"    Std: {mu.std().item():.4f}")
    print(f"    Min: {mu.min().item():.4f}")
    print(f"    Max: {mu.max().item():.4f}")
    print(f"  Var stats:")
    print(f"    Mean: {var.mean().item():.4f}")
    print(f"    Std: {var.std().item():.4f}")
    print(f"    Min: {var.min().item():.4f}")
    print(f"    Max: {var.max().item():.4f}")
    
    # ==================== Sample from Latent ====================
    print(f"\n{'='*60}")
    print("Sampling from Latent Distribution...")
    print("=" * 60)
    
    # Reparameterization trick: z = mu + std * epsilon
    epsilon = torch.randn_like(mu)
    latent = mu + torch.sqrt(var) * epsilon
    
    print(f"\nLatent Sample:")
    print(f"  Shape: {latent.shape}")
    print(f"  Mean: {latent.mean().item():.4f}")
    print(f"  Std: {latent.std().item():.4f}")
    
    # ==================== Decode (Teacher Forcing) ====================
    print(f"\n{'='*60}")
    print("Decoding with Teacher Forcing...")
    print("=" * 60)
    
    # Create target tokens (for teacher forcing)
    # Shift input_tokens by one position
    decoder_input = input_tokens[:, :-1]  # Remove last token
    decoder_target = input_tokens[:, 1:]  # Remove first token
    
    decoder.eval()
    with torch.no_grad():
        logits = decoder(latent, decoder_input, mode='forced')
    
    print(f"\nDecoder Output (Teacher Forcing):")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected: [batch_size={batch_size}, seq_len-1={decoder_input.shape[1]}, vocab_size={vocab_size}]")
    print(f"  Logits stats:")
    print(f"    Mean: {logits.mean().item():.4f}")
    print(f"    Std: {logits.std().item():.4f}")
    print(f"    Min: {logits.min().item():.4f}")
    print(f"    Max: {logits.max().item():.4f}")
    
    # Convert logits to probabilities and predictions
    probs = F.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)
    
    print(f"\nPredictions:")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  First sample predictions (first 15): {predictions[0, :15].tolist()}")
    print(f"  First sample targets (first 15): {decoder_target[0, :15].tolist()}")
    
    # Calculate accuracy (excluding padding tokens)
    mask = (decoder_target != 0)  # Non-padding tokens
    correct = (predictions == decoder_target) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    print(f"  Accuracy (excluding padding): {accuracy.item()*100:.2f}%")
    
    # ==================== Autoregressive Generation ====================
    print(f"\n{'='*60}")
    print("Autoregressive Generation...")
    print("=" * 60)
    
    decoder.eval()
    with torch.no_grad():
        generated_logits = decoder.generate_forward(
            latent,
            start_token=1,
            max_length=50  # Generate 50 tokens
        )
    
    print(f"\nGenerated Output (Autoregressive):")
    print(f"  Logits shape: {generated_logits.shape}")
    print(f"  Expected: [batch_size={batch_size}, max_length=50, vocab_size={vocab_size}]")
    
    generated_predictions = torch.argmax(generated_logits, dim=-1)
    print(f"  Generated predictions shape: {generated_predictions.shape}")
    print(f"  First sample generated tokens: {generated_predictions[0, :20].tolist()}")
    
    # ==================== Random Generation ====================
    print(f"\n{'='*60}")
    print("Random Generation from Prior...")
    print("=" * 60)
    
    # Sample from prior distribution N(0, I)
    random_latent = torch.randn(batch_size, latent_size)
    
    decoder.eval()
    with torch.no_grad():
        random_logits = decoder.generate_forward(
            random_latent,
            start_token=1,
            max_length=50
        )
    
    random_predictions = torch.argmax(random_logits, dim=-1)
    print(f"\nRandom Generation Output:")
    print(f"  Random latent shape: {random_latent.shape}")
    print(f"  Generated logits shape: {random_logits.shape}")
    print(f"  First sample random tokens: {random_predictions[0, :20].tolist()}")
    
    # ==================== Summary ====================
    print(f"\n{'='*60}")
    print("Test Summary")
    print("=" * 60)
    print("✓ Encoder created successfully")
    print("✓ Decoder created successfully")
    print("✓ Input tokens created and encoded")
    print("✓ Latent representation sampled")
    print("✓ Teacher forcing decoding completed")
    print("✓ Autoregressive generation completed")
    print("✓ Random generation from prior completed")
    print("\nAll tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()

