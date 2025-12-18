"""
Minimal Test Script for SMILES Encoder and Decoder

A simple test that shows the basic workflow with input tokens and output.
"""

import torch
from Genotype_Induced_Drug_Design.SEncoder.Custom_Pipeline.smiles_encoder import SMILESEncoder
from Genotype_Induced_Drug_Design.SEncoder.Custom_Pipeline.smiles_decoder import SMILESDecoder

# Configuration
vocab_size = 45
embedding_dim = 512
latent_size = 512
batch_size = 2
seq_length = 30

print("Creating models...")
encoder = SMILESEncoder(vocab_size=vocab_size, embedding_dim=embedding_dim, latent_size=latent_size)
decoder = SMILESDecoder(vocab_size=vocab_size, embedding_dim=embedding_dim, latent_size=latent_size)

# Create input tokens (simulating tokenized SMILES)
# Token 0 = padding, Token 1 = start, Token 2 = end, Tokens 3-44 = SMILES tokens
input_tokens = torch.cat([
    torch.ones(batch_size, 1, dtype=torch.long),      # Start token
    torch.randint(3, vocab_size, (batch_size, seq_length)),  # SMILES tokens
    torch.full((batch_size, 1), 2, dtype=torch.long),  # End token
    torch.zeros(batch_size, 90, dtype=torch.long)      # Padding
], dim=1)

print(f"\nInput tokens shape: {input_tokens.shape}")
print(f"Input tokens (first sample, first 10): {input_tokens[0, :10].tolist()}")

# Encode
print("\nEncoding...")
mu, var = encoder(input_tokens)
print(f"Mu (mean) shape: {mu.shape}")
print(f"Var (variance) shape: {var.shape}")

# Sample from latent
latent = mu + torch.sqrt(var) * torch.randn_like(mu)
print(f"Latent shape: {latent.shape}")

# Decode with teacher forcing
print("\nDecoding with teacher forcing...")
decoder_input = input_tokens[:, :-1]  # Remove last token
logits = decoder(latent, decoder_input, mode='forced')
print(f"Output logits shape: {logits.shape}")

# Get predictions
predictions = torch.argmax(logits, dim=-1)
print(f"Predictions shape: {predictions.shape}")
print(f"Predictions (first sample, first 10): {predictions[0, :10].tolist()}")

# Autoregressive generation
print("\nAutoregressive generation...")
generated_logits = decoder.generate_forward(latent, start_token=1, max_length=40)
generated_predictions = torch.argmax(generated_logits, dim=-1)
print(f"Generated predictions shape: {generated_predictions.shape}")
print(f"Generated tokens (first sample, first 15): {generated_predictions[0, :15].tolist()}")

print("\n✓ Test completed successfully!")

