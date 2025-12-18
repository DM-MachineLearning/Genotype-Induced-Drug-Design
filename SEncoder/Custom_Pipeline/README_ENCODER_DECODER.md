# SMILES Encoder and Decoder for Transformer VAE

Custom implementation of encoder and decoder modules for de novo molecule generation using Transformer-based Variational Autoencoder (VAE) architecture.

Based on the paper:
**"A novel molecule generative model of VAE combined with Transformer for unseen structure generation"** by Yoshikai et al.

## Files

- `smiles_encoder.py` - Encoder module that transforms SMILES sequences into latent representations
- `smiles_decoder.py` - Decoder module that transforms latent representations into SMILES sequences
- `encoder_decoder_example.py` - Complete usage examples

## Architecture Overview

### Encoder (`SMILESEncoder`)

The encoder transforms SMILES token sequences into a latent representation suitable for VAE:

1. **Positional Embedding** - Combines token embeddings with sinusoidal positional encodings
2. **Transformer Encoder** - 8 layers with:
   - 512-dimensional embeddings (d_model)
   - 8 attention heads
   - 2048-dimensional feedforward network (4× d_model)
   - NewGELU activation
   - Layer normalization
3. **Pooling Layer** - NoAffinePooler that concatenates:
   - Mean pooling over sequence
   - Start token embedding
   - Max pooling over sequence
4. **VAE Projections** - Linear layers to project pooled features to:
   - `mu` (mean of latent distribution)
   - `var` (variance of latent distribution)

### Decoder (`SMILESDecoder`)

The decoder transforms latent representations back into SMILES token sequences:

1. **Positional Embedding** - Same as encoder
2. **Transformer Decoder** - 8 layers with:
   - 512-dimensional embeddings (d_model)
   - 8 attention heads
   - 2048-dimensional feedforward network
   - Masked self-attention (causal mask for autoregressive generation)
   - NewGELU activation
   - Layer normalization
3. **Output Projection** - Projects decoder outputs to vocabulary logits

## Key Features

- **VAE Architecture** - Generates molecules from continuous latent space, enabling:
  - Property prediction from latent representations
  - Conditional generation based on desired properties
  - Smooth interpolation between molecules

- **Transformer-based** - Modern attention mechanism for better sequence modeling

- **Novel Structure Generation** - Superior performance in generating molecules with unseen structures (as shown in paper)

- **Flexible Latent Size** - Supports variable latent dimensions (paper suggests ~32-512 dimensions)

## Usage

### Basic Setup

```python
import torch
from smiles_encoder import SMILESEncoder
from smiles_decoder import SMILESDecoder

# Hyperparameters (based on paper)
vocab_size = 45  # SMILES vocabulary size
embedding_dim = 512  # Model dimension
latent_size = 512  # Latent space dimension (can be reduced to ~32)
n_layers = 8  # Number of encoder/decoder layers
max_len = 122  # Maximum sequence length

# Create encoder and decoder
encoder = SMILESEncoder(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    n_layers=n_layers,
    latent_size=latent_size,
    max_len=max_len,
    dropout=0.0  # No dropout as per paper
)

decoder = SMILESDecoder(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    n_layers=n_layers,
    latent_size=latent_size,
    max_len=max_len,
    dropout=0.0
)
```

### Encoding SMILES Sequences

```python
# Input: tokenized SMILES sequences
# Shape: [batch_size, sequence_length]
input_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))

# Encode to latent representation
mu, var = encoder(input_tokens)
# mu: [batch_size, latent_size] - mean of latent distribution
# var: [batch_size, latent_size] - variance of latent distribution

# Sample from latent distribution (for VAE)
epsilon = torch.randn_like(mu)
latent = mu + torch.sqrt(var) * epsilon
```

### Decoding (Teacher Forcing - Training)

```python
# Decode with teacher forcing (for training)
# target_tokens: [batch_size, length] - full target sequence
decoder_input = target_tokens[:, :-1]  # Remove last token

logits = decoder(latent, decoder_input, mode='forced')
# logits: [batch_size, length-1, vocab_size]

# Compute reconstruction loss
loss = torch.nn.functional.cross_entropy(
    logits.reshape(-1, vocab_size),
    target_tokens[:, 1:].reshape(-1),
    ignore_index=0  # Ignore padding
)
```

### Decoding (Autoregressive Generation - Inference)

```python
# Generate autoregressively (for inference)
encoder.eval()
decoder.eval()

with torch.no_grad():
    # Encode input
    mu, var = encoder(input_tokens)
    latent = mu + torch.sqrt(var) * torch.randn_like(mu)
    
    # Generate
    generated_logits = decoder.generate_forward(
        latent,
        start_token=1,  # Start token index
        max_length=122
    )
    # generated_logits: [batch_size, max_length, vocab_size]
    
    # Convert to token predictions
    predictions = torch.argmax(generated_logits, dim=-1)
```

### Random Generation from Prior

```python
# Generate new molecules from prior distribution N(0, I)
latent = torch.randn(batch_size, latent_size)  # Sample from prior

generated_logits = decoder.generate_forward(
    latent,
    start_token=1,
    max_length=122
)
```

### Complete Training Example

```python
import torch.nn as nn

# Training loop
encoder.train()
decoder.train()

# Forward pass
mu, var = encoder(input_tokens)
latent = mu + torch.sqrt(var) * torch.randn_like(mu)
decoder_input = target_tokens[:, :-1]
logits = decoder(latent, decoder_input, mode='forced')

# Compute losses
reconstruction_loss = nn.functional.cross_entropy(
    logits.reshape(-1, vocab_size),
    target_tokens[:, 1:].reshape(-1),
    ignore_index=0
)

# KL divergence loss
kl_loss = 0.5 * (
    torch.sum(mu**2) + torch.sum(var) - 
    torch.sum(torch.log(var + 1e-8)) - mu.numel()
)

# Total loss
beta = 0.001  # Weight for KL loss
total_loss = reconstruction_loss + beta * kl_loss

# Backward pass
total_loss.backward()
optimizer.step()
```

## Hyperparameters (Based on Paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_layers` | 8 | Number of encoder/decoder layers |
| `embedding_dim` | 512 | Model dimension (d_model) |
| `nhead` | 8 | Number of attention heads |
| `dim_feedforward` | 2048 | Feedforward dimension (4 × d_model) |
| `dropout` | 0.0 | No dropout (as per paper) |
| `latent_size` | 512 | Latent space dimension (can be reduced to ~32) |
| `max_len` | 122 | Maximum sequence length |
| `beta` | 0.001 | KL loss weight |

## Paper Results

According to the paper, the Transformer VAE model achieves:

- **Reconstruction Accuracy**: ~90% perfect accuracy, ~96% partial accuracy
- **Novel Structure Generation**: Superior performance in generating molecules with unseen structures compared to baseline models
- **Latent Representation**: Successfully predicts molecular properties from latent space
- **Compact Latent Space**: Can be reduced to ~32 dimensions without significant loss

## Dependencies

```
torch >= 1.8.0
```

## Example Script

Run the example script to see complete usage:

```bash
python encoder_decoder_example.py
```

This demonstrates:
- Model creation
- Training step
- Generation from input
- Random generation from prior

## Notes

1. **Vocabulary**: The encoder/decoder expect tokenized SMILES sequences. You need to tokenize SMILES strings according to your vocabulary (typically 45 tokens including special tokens like `<start>`, `<end>`, `<padding>`).

2. **Latent Size**: The paper suggests latent dimensions can be reduced to ~32 for more compact molecular descriptors while maintaining reconstruction quality.

3. **Padding**: Use padding token index 0 (default). The encoder automatically creates padding masks from input sequences.

4. **Training**: The model uses teacher forcing during training and autoregressive generation during inference.

5. **Initialization**: Weights are initialized according to paper specifications (Glorot uniform for attention, Normal(0, 0.02) for feedforward).

## References

Yoshikai, Y., Mizuno, T., Nemoto, S., & Kusuhara, H. (Year). A novel molecule generative model of VAE combined with Transformer for unseen structure generation. *[Journal/Conference]*.

## License

This implementation is based on the TransformerVAE codebase and paper. Please refer to the original repository for licensing information.

