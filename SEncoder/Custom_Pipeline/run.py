"""
Simple Test Script - Run this to test the encoder and decoder
"""

import torch
from Genotype_Induced_Drug_Design.SEncoder.Custom_Pipeline.smiles_encoder import SMILESEncoder
from Genotype_Induced_Drug_Design.SEncoder.Custom_Pipeline.smiles_decoder import SMILESDecoder

# Create encoder and decoder
print("Creating encoder and decoder...")
encoder = SMILESEncoder(vocab_size=45, embedding_dim=512, latent_size=512)
decoder = SMILESDecoder(vocab_size=45, embedding_dim=512, latent_size=512)

# Create sample input tokens
# Format: [start_token, ...SMILES_tokens..., end_token, ...padding...]
batch_size = 2
input_tokens = torch.cat([
    torch.ones(batch_size, 1, dtype=torch.long),              # Start token (1)
    torch.randint(3, 45, (batch_size, 25), dtype=torch.long), # SMILES tokens (3-44)
    torch.full((batch_size, 1), 2, dtype=torch.long),         # End token (2)
    torch.zeros(batch_size, 95, dtype=torch.long)             # Padding (0)
], dim=1)

print(f"\nInput tokens shape: {input_tokens.shape}")
print(f"Sample input (first 10 tokens): {input_tokens[0, :10].tolist()}")

# Encode SMILES to latent representation
print("\nEncoding input tokens...")
mu, var = encoder(input_tokens)
print(f"Mu (mean) shape: {mu.shape}")
print(f"Var (variance) shape: {var.shape}")

# Sample from latent distribution
latent = mu + torch.sqrt(var) * torch.randn_like(mu)
print(f"Latent shape: {latent.shape}")

# Decode back to SMILES tokens
print("\nDecoding with teacher forcing...")
decoder_input = input_tokens[:, :-1]  # Remove last token for decoder input
logits = decoder(latent, decoder_input, mode='forced')
print(f"Output logits shape: {logits.shape}")

# Get token predictions
predictions = torch.argmax(logits, dim=-1)
print(f"Predictions shape: {predictions.shape}")
print(f"Sample predictions (first 10): {predictions[0, :10].tolist()}")

# Autoregressive generation
print("\nGenerating new molecules (autoregressive)...")
generated_logits = decoder.generate_forward(latent, start_token=1, max_length=50)
generated_tokens = torch.argmax(generated_logits, dim=-1)
print(f"Generated tokens shape: {generated_tokens.shape}")
print(f"Generated sample (first 15 tokens): {generated_tokens[0, :15].tolist()}")

print("\n✓ All tests passed!")
