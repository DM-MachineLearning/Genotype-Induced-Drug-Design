"""
SMILES Encoder for Transformer VAE Molecule Generator

This module implements the encoder component of a Transformer-based Variational Autoencoder
for SMILES molecule generation. Based on the paper:
"A novel molecule generative model of VAE combined with Transformer for unseen structure generation"
by Yoshikai et al.

The encoder transforms SMILES string token sequences into a latent representation (mu, var)
suitable for VAE-based molecule generation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NewGELU(nn.Module):
    """
    GELU activation function with improved approximation.
    Used in modern Transformer architectures.
    """
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))
        ))


def get_positional_encoding(length: int, emb_size: int) -> torch.Tensor:
    """
    Generate sinusoidal positional encodings.
    
    Parameters
    ----------
    length : int
        Maximum sequence length
    emb_size : int
        Embedding dimension
    
    Returns
    -------
    pe : torch.Tensor
        Positional encodings of shape [length, 1, emb_size]
    """
    pe = torch.zeros(length, emb_size)
    position = torch.arange(0, length).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, emb_size, 2).float() * 
                        -(math.log(10000.0) / emb_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(1)
    return pe


class PositionalEmbedding(nn.Module):
    """
    Positional embedding layer that combines token embeddings with positional encodings.
    
    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary
    embedding_dim : int
        Dimension of token embeddings
    max_len : int
        Maximum sequence length
    padding_idx : int
        Index of padding token (default: 0)
    dropout : float
        Dropout probability (default: 0.0)
    """
    def __init__(self, vocab_size: int, embedding_dim: int, max_len: int = 122,
                 padding_idx: int = 0, dropout: float = 0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Register positional encodings as buffer
        pe = get_positional_encoding(max_len, embedding_dim)
        self.register_buffer('pe', pe)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor [batch_size, length]
            Token indices
        
        Returns
        -------
        output : torch.Tensor [length, batch_size, embedding_dim]
            Positionally encoded embeddings
        """
        # Transpose to [length, batch_size] for embedding, then to [length, batch_size, emb_dim]
        input = self.embedding(input.transpose(0, 1).contiguous())
        length = input.shape[0]
        
        # Extend positional encoding if needed
        if length > self.max_len:
            print(f"[WARNING] Sequence length {length} exceeds max_len {self.max_len}. Extending PE.")
            pe = get_positional_encoding(length, self.embedding_dim).to(self.pe.device)
            self.register_buffer('pe', pe)
            self.max_len = length
        
        # Add positional encoding
        pe = self.pe[:length]
        output = self.dropout(input + pe)
        return output


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer with self-attention and feedforward network.
    
    Parameters
    ----------
    d_model : int
        Model dimension
    nhead : int
        Number of attention heads
    dim_feedforward : int
        Dimension of feedforward network
    dropout : float
        Dropout probability
    activation : str
        Activation function ('newgelu' or 'relu')
    layer_norm_eps : float
        Epsilon for layer normalization
    """
    def __init__(self, d_model: int = 512, nhead: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.0, activation: str = 'newgelu',
                 layer_norm_eps: float = 1e-9):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False
        )
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Activation
        if activation == 'newgelu':
            self.activation = NewGELU()
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        src : torch.Tensor [length, batch_size, d_model]
            Input sequence
        src_key_padding_mask : torch.Tensor [batch_size, length]
            Boolean mask where True indicates padding tokens
        
        Returns
        -------
        output : torch.Tensor [length, batch_size, d_model]
            Encoded sequence
        """
        # Self-attention with residual connection and layer norm
        src2 = self.norm1(src)
        src2, _ = self.self_attn(
            src2, src2, src2,
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )
        src = src + self.dropout1(src2)
        
        # Feedforward with residual connection and layer norm
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


class NoAffinePooler(nn.Module):
    """
    Pooling layer that concatenates mean, start, and max pooling.
    Uses LayerNorm without affine transformation (NoAffine) as per paper.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.mean_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.start_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.max_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.output_dim = input_dim * 3
    
    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor [length, batch_size, feature_dim]
            Sequence to pool
        padding_mask : torch.Tensor [batch_size, length]
            Boolean mask where True indicates padding tokens
        
        Returns
        -------
        output : torch.Tensor [batch_size, feature_dim * 3]
            Pooled representation (mean + start + max)
        """
        # Convert padding mask: True = padding, False = valid token
        padding_mask_expanded = ~padding_mask.unsqueeze(-1)  # [batch_size, length, 1]
        padding_mask_expanded = padding_mask_expanded.transpose(0, 1)  # [length, batch_size, 1]
        
        # Mean pooling (masked)
        mean_pooled = torch.sum(input * padding_mask_expanded, dim=0) / \
                     torch.sum(padding_mask_expanded, dim=0).clamp(min=1e-9)
        mean_pooled = self.mean_norm(mean_pooled)
        
        # Start token pooling
        start_pooled = self.start_norm(input[0])
        
        # Max pooling (masked)
        masked_input = input.masked_fill(~padding_mask_expanded, float('-inf'))
        max_pooled = torch.max(masked_input, dim=0)[0]
        max_pooled = self.max_norm(max_pooled)
        
        # Concatenate all three
        output = torch.cat([mean_pooled, start_pooled, max_pooled], dim=-1)
        return output


class SMILESEncoder(nn.Module):
    """
    Complete SMILES Encoder for Transformer VAE.
    
    Encodes SMILES token sequences into latent representations (mu, var) for VAE.
    
    Architecture (based on paper):
    - Positional Embedding
    - Transformer Encoder (8 layers, 512 d_model, 8 heads)
    - NoAffinePooler (Mean + Start + Max pooling)
    - Linear projections to mu and var
    
    Parameters
    ----------
    vocab_size : int
        Size of SMILES vocabulary (default: 45)
    embedding_dim : int
        Embedding dimension (default: 512)
    n_layers : int
        Number of encoder layers (default: 8)
    nhead : int
        Number of attention heads (default: 8)
    dim_feedforward : int
        Feedforward dimension (default: 2048, which is 4 * embedding_dim)
    latent_size : int
        Dimension of latent space (default: 512, can be reduced to ~32)
    max_len : int
        Maximum sequence length (default: 122)
    dropout : float
        Dropout probability (default: 0.0 as per paper)
    padding_idx : int
        Padding token index (default: 0)
    """
    def __init__(self, vocab_size: int = 45, embedding_dim: int = 512, n_layers: int = 8,
                 nhead: int = 8, dim_feedforward: int = 2048, latent_size: int = 512,
                 max_len: int = 122, dropout: float = 0.0, padding_idx: int = 0):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.latent_size = latent_size
        self.max_len = max_len
        self.padding_idx = padding_idx
        
        # Positional embedding
        self.embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_len=max_len,
            padding_idx=padding_idx,
            dropout=dropout
        )
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='newgelu',
                layer_norm_eps=1e-9
            ) for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(embedding_dim, eps=1e-9)
        
        # Pooling layer (outputs 3 * embedding_dim)
        pooled_size = embedding_dim * 3
        self.pooler = NoAffinePooler(input_dim=embedding_dim)
        
        # Projection to mu (mean of latent distribution)
        self.latent2mu = nn.Linear(pooled_size, latent_size)
        nn.init.zeros_(self.latent2mu.bias)
        
        # Projection to var (variance of latent distribution)
        self.latent2var = nn.Sequential(
            nn.Linear(pooled_size, latent_size),
            nn.Softplus()  # Ensures positive variance
        )
        nn.init.zeros_(self.latent2var[0].bias)
        
        # Initialize weights according to paper
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights according to paper specifications."""
        for layer in self.encoder_layers:
            # Self-attention weights: Glorot uniform
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.zeros_(layer.self_attn.in_proj_bias)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
            
            # Feedforward weights: Normal(0, 0.02)
            nn.init.normal_(layer.linear1.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.linear1.bias)
            nn.init.normal_(layer.linear2.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.linear2.bias)
    
    def forward(self, input: torch.Tensor, padding_mask: torch.Tensor = None) -> tuple:
        """
        Encode SMILES sequence into latent representation.
        
        Parameters
        ----------
        input : torch.Tensor [batch_size, length]
            Token indices of SMILES sequences
        padding_mask : torch.Tensor [batch_size, length], optional
            Boolean mask where True indicates padding tokens.
            If None, will be created from input == 0 (padding_idx)
        
        Returns
        -------
        mu : torch.Tensor [batch_size, latent_size]
            Mean of latent distribution
        var : torch.Tensor [batch_size, latent_size]
            Variance of latent distribution
        """
        batch_size = input.shape[0]
        
        # Create padding mask if not provided
        if padding_mask is None:
            padding_mask = (input == self.padding_idx)
        
        # Embedding with positional encoding
        # Output: [length, batch_size, embedding_dim]
        embedded = self.embedding(input)
        
        # Transformer encoding
        # Transpose padding_mask: [batch_size, length] -> needed for attention
        memory = embedded
        for layer in self.encoder_layers:
            memory = layer(memory, src_key_padding_mask=padding_mask)
        
        memory = self.final_norm(memory)
        
        # Pooling: [length, batch_size, embedding_dim] -> [batch_size, pooled_size]
        pooled = self.pooler(memory, padding_mask)
        
        # Project to latent space
        mu = self.latent2mu(pooled)
        var = self.latent2var(pooled)
        
        return mu, var
    
    def encode_to_latent(self, input: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Encode and return only the mean (mu) of the latent distribution.
        Useful for inference when sampling from latent is not needed.
        
        Parameters
        ----------
        input : torch.Tensor [batch_size, length]
            Token indices of SMILES sequences
        padding_mask : torch.Tensor [batch_size, length], optional
        
        Returns
        -------
        mu : torch.Tensor [batch_size, latent_size]
            Mean of latent distribution
        """
        mu, _ = self.forward(input, padding_mask)
        return mu

