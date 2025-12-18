"""
Example usage of SMILES Encoder and Decoder for Transformer VAE

This script demonstrates how to use the custom encoder and decoder files
for de novo molecule generation based on the Transformer VAE architecture.
"""

import torch
import torch.nn as nn
from smiles_encoder import SMILESEncoder
from smiles_decoder import SMILESDecoder


def create_vae_model(vocab_size=45, embedding_dim=512, latent_size=512, 
                    n_layers=8, max_len=122, dropout=0.0):
    """
    Create a complete Transformer VAE model for SMILES generation.
    
    Parameters
    ----------
    vocab_size : int
        Size of SMILES vocabulary
    embedding_dim : int
        Embedding dimension (d_model)
    latent_size : int
        Dimension of latent space
    n_layers : int
        Number of encoder/decoder layers
    max_len : int
        Maximum sequence length
    dropout : float
        Dropout probability
    
    Returns
    -------
    encoder : SMILESEncoder
        Encoder module
    decoder : SMILESDecoder
        Decoder module
    """
    encoder = SMILESEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        latent_size=latent_size,
        max_len=max_len,
        dropout=dropout
    )
    
    decoder = SMILESDecoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        latent_size=latent_size,
        max_len=max_len,
        dropout=dropout
    )
    
    return encoder, decoder


def sample_from_latent(mu, var, var_coef=1.0, training=True):
    """
    Sample from latent distribution (VAE reparameterization trick).
    
    Parameters
    ----------
    mu : torch.Tensor [batch_size, latent_size]
        Mean of latent distribution
    var : torch.Tensor [batch_size, latent_size]
        Variance of latent distribution
    var_coef : float
        Coefficient for variance scaling
    training : bool
        If True, sample from distribution; if False, use mean
    
    Returns
    -------
    latent : torch.Tensor [batch_size, latent_size]
        Sampled latent representation
    """
    if training:
        # Reparameterization trick: z = mu + std * epsilon
        epsilon = torch.randn_like(mu)
        latent = mu + torch.sqrt(var) * epsilon * var_coef
    else:
        latent = mu
    return latent


def compute_kl_loss(mu, var):
    """
    Compute KL divergence loss between latent distribution and standard normal.
    
    KL(q(z|x) || p(z)) where p(z) = N(0, I)
    
    Parameters
    ----------
    mu : torch.Tensor [batch_size, latent_size]
        Mean of latent distribution
    var : torch.Tensor [batch_size, latent_size]
        Variance of latent distribution
    
    Returns
    -------
    kl_loss : torch.Tensor
        KL divergence loss (scalar)
    """
    # KL divergence: 0.5 * sum(mu^2 + var - log(var) - 1)
    kl_loss = 0.5 * (torch.sum(mu**2) + torch.sum(var) - 
                     torch.sum(torch.log(var + 1e-8)) - mu.numel())
    return kl_loss


def example_training_step(encoder, decoder, input_tokens, target_tokens, 
                         beta=0.001, var_coef=1.0):
    """
    Example training step for Transformer VAE.
    
    Parameters
    ----------
    encoder : SMILESEncoder
        Encoder module
    decoder : SMILESDecoder
        Decoder module
    input_tokens : torch.Tensor [batch_size, length]
        Input SMILES token sequences
    target_tokens : torch.Tensor [batch_size, length]
        Target SMILES token sequences (for teacher forcing)
    beta : float
        Weight for KL loss
    var_coef : float
        Coefficient for variance scaling during training
    
    Returns
    -------
    reconstruction_loss : torch.Tensor
        Cross-entropy loss for reconstruction
    kl_loss : torch.Tensor
        KL divergence loss
    total_loss : torch.Tensor
        Total loss (reconstruction + beta * KL)
    """
    # Encode
    mu, var = encoder(input_tokens)
    
    # Sample from latent
    latent = sample_from_latent(mu, var, var_coef=var_coef, training=True)
    
    # Decode (teacher forcing)
    # Prepare decoder input (shift target by one position for teacher forcing)
    decoder_input = target_tokens[:, :-1]  # Remove last token
    decoder_target = target_tokens[:, 1:]  # Remove first token (start token)
    
    logits = decoder(latent, decoder_input, mode='forced')
    # logits: [batch_size, length-1, vocab_size]
    
    # Compute reconstruction loss (cross-entropy)
    reconstruction_loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        decoder_target.reshape(-1),
        ignore_index=0  # Ignore padding tokens
    )
    
    # Compute KL loss
    kl_loss = compute_kl_loss(mu, var)
    
    # Total loss
    total_loss = reconstruction_loss + beta * kl_loss
    
    return reconstruction_loss, kl_loss, total_loss


def example_generation(encoder, decoder, input_tokens, start_token=1, 
                      max_length=122, use_mean=False):
    """
    Example generation from SMILES sequence.
    
    Parameters
    ----------
    encoder : SMILESEncoder
        Encoder module
    decoder : SMILESDecoder
        Decoder module
    input_tokens : torch.Tensor [batch_size, length]
        Input SMILES token sequences to encode
    start_token : int
        Start token index
    max_length : int
        Maximum generation length
    use_mean : bool
        If True, use mean (mu) instead of sampling for latent
    
    Returns
    -------
    generated_logits : torch.Tensor [batch_size, max_length, vocab_size]
        Generated token logits
    latent : torch.Tensor [batch_size, latent_size]
        Latent representation used for generation
    """
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # Encode to latent
        mu, var = encoder(input_tokens)
        
        if use_mean:
            latent = mu
        else:
            # Sample from latent
            latent = sample_from_latent(mu, var, training=True)
        
        # Generate
        generated_logits = decoder.generate_forward(
            latent, 
            start_token=start_token,
            max_length=max_length
        )
    
    return generated_logits, latent


def example_random_generation(decoder, batch_size=1, latent_size=512,
                             start_token=1, max_length=122):
    """
    Example random generation from prior distribution.
    
    Parameters
    ----------
    decoder : SMILESDecoder
        Decoder module
    batch_size : int
        Number of molecules to generate
    latent_size : int
        Dimension of latent space
    start_token : int
        Start token index
    max_length : int
        Maximum generation length
    
    Returns
    -------
    generated_logits : torch.Tensor [batch_size, max_length, vocab_size]
        Generated token logits
    """
    decoder.eval()
    
    with torch.no_grad():
        # Sample from prior: N(0, I)
        latent = torch.randn(batch_size, latent_size)
        
        # Generate
        generated_logits = decoder.generate_forward(
            latent,
            start_token=start_token,
            max_length=max_length
        )
    
    return generated_logits


if __name__ == "__main__":
    # Example usage
    
    # Hyperparameters (based on paper)
    vocab_size = 45  # Size of SMILES vocabulary
    embedding_dim = 512  # d_model
    latent_size = 512  # Can be reduced to ~32 for smaller descriptors
    n_layers = 8  # Number of encoder/decoder layers
    max_len = 122  # Maximum sequence length
    dropout = 0.0  # No dropout as per paper
    batch_size = 4
    seq_length = 50
    
    # Create model
    print("Creating Transformer VAE model...")
    encoder, decoder = create_vae_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        latent_size=latent_size,
        n_layers=n_layers,
        max_len=max_len,
        dropout=dropout
    )
    
    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    
    # Example 1: Training step
    print("\n--- Example 1: Training Step ---")
    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    target_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    recon_loss, kl_loss, total_loss = example_training_step(
        encoder, decoder, input_tokens, target_tokens, beta=0.001
    )
    
    print(f"Reconstruction loss: {recon_loss.item():.4f}")
    print(f"KL loss: {kl_loss.item():.4f}")
    print(f"Total loss: {total_loss.item():.4f}")
    
    # Example 2: Generation from input
    print("\n--- Example 2: Generation from Input ---")
    generated_logits, latent = example_generation(
        encoder, decoder, input_tokens, use_mean=False
    )
    print(f"Generated logits shape: {generated_logits.shape}")
    print(f"Latent shape: {latent.shape}")
    
    # Example 3: Random generation from prior
    print("\n--- Example 3: Random Generation from Prior ---")
    random_logits = example_random_generation(
        decoder, batch_size=2, latent_size=latent_size
    )
    print(f"Random generation logits shape: {random_logits.shape}")
    
    print("\nAll examples completed successfully!")

