"""
SMILES Decoder for Transformer VAE Molecule Generator

This module implements the decoder component of a Transformer-based Variational Autoencoder
for SMILES molecule generation. Based on the paper:
"A novel molecule generative model of VAE combined with Transformer for unseen structure generation"
by Yoshikai et al.

The decoder transforms latent representations into SMILES token sequences, supporting
both teacher-forced training and autoregressive generation.
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
        self.dropout = nn.Dropout(dropout)
        
        # Register positional encodings as buffer
        pe = get_positional_encoding(max_len, embedding_dim)
        self.register_buffer('pe', pe)
    
    def forward(self, input: torch.Tensor, position: int = None) -> torch.Tensor:
        """
        Parameters
        ----------
        input : torch.Tensor [batch_size, length] or [batch_size, 1]
            Token indices
        position : int, optional
            Specific position index (for autoregressive decoding)
            If None, uses all positions in sequence
        
        Returns
        -------
        output : torch.Tensor [length, batch_size, embedding_dim] or [1, batch_size, embedding_dim]
            Positionally encoded embeddings
        """
        # Handle single token input (autoregressive) vs sequence input (teacher forcing)
        if input.dim() == 2 and input.shape[1] == 1:
            # Single token for autoregressive decoding
            input_emb = self.embedding(input.transpose(0, 1))  # [1, batch_size, embedding_dim]
            if position is not None:
                pe = self.pe[position:position+1]  # [1, 1, embedding_dim]
            else:
                pe = self.pe[0:1]  # Default to position 0
            output = self.dropout(input_emb + pe)
        else:
            # Full sequence for teacher forcing
            input_emb = self.embedding(input.transpose(0, 1).contiguous())  # [length, batch_size, embedding_dim]
            length = input_emb.shape[0]
            
            # Extend positional encoding if needed
            if length > self.max_len:
                print(f"[WARNING] Sequence length {length} exceeds max_len {self.max_len}. Extending PE.")
                pe = get_positional_encoding(length, self.embedding_dim).to(self.pe.device)
                self.register_buffer('pe', pe)
                self.max_len = length
            
            if position is None:
                pe = self.pe[:length]
            else:
                pe = self.pe[position:position+length]
            
            output = self.dropout(input_emb + pe)
        
        return output


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer decoder layer with masked self-attention and feedforward network.
    
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
    
    def forward(self, tgt: torch.Tensor, attn_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        tgt : torch.Tensor [length, batch_size, d_model]
            Target sequence (decoder input)
        attn_mask : torch.Tensor [length, length]
            Attention mask (typically causal mask)
        tgt_key_padding_mask : torch.Tensor [batch_size, length]
            Boolean mask where True indicates padding tokens
        
        Returns
        -------
        output : torch.Tensor [length, batch_size, d_model]
            Decoded sequence
        """
        # Self-attention with residual connection and layer norm
        tgt2 = self.norm1(tgt)
        tgt2, _ = self.self_attn(
            tgt2, tgt2, tgt2,
            attn_mask=attn_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False
        )
        tgt = tgt + self.dropout1(tgt2)
        
        # Feedforward with residual connection and layer norm
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        
        return tgt
    
    def cell_forward(self, tgt: torch.Tensor, prev_state: torch.Tensor,
                     attn_mask: torch.Tensor = None) -> tuple:
        """
        Single step forward for autoregressive decoding.
        
        Parameters
        ----------
        tgt : torch.Tensor [1, batch_size, d_model]
            Current token embedding
        prev_state : torch.Tensor [prev_length, batch_size, d_model]
            Previous decoder states (accumulated)
        attn_mask : torch.Tensor, optional
            Attention mask
        
        Returns
        -------
        output : torch.Tensor [1, batch_size, d_model]
            Output for current step
        new_state : torch.Tensor [prev_length+1, batch_size, d_model]
            Updated state with current step appended
        """
        # Normalize current input
        cur_y = self.norm1(tgt)
        
        # Concatenate with previous states
        y = torch.cat([prev_state, cur_y], dim=0)
        
        # Self-attention over all previous tokens including current
        cur_attn, _ = self.self_attn(
            cur_y, y, y,
            attn_mask=attn_mask,
            need_weights=False
        )
        tgt = tgt + self.dropout1(cur_attn)
        
        # Feedforward
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        
        # Update state
        new_state = y
        
        return tgt, new_state


class SMILESDecoder(nn.Module):
    """
    Complete SMILES Decoder for Transformer VAE.
    
    Decodes latent representations into SMILES token sequences.
    
    Architecture (based on paper):
    - Positional Embedding
    - Transformer Decoder (8 layers, 512 d_model, 8 heads)
    - Output projection to vocabulary probabilities
    
    Parameters
    ----------
    vocab_size : int
        Size of SMILES vocabulary (default: 45)
    embedding_dim : int
        Embedding dimension (default: 512)
    n_layers : int
        Number of decoder layers (default: 8)
    nhead : int
        Number of attention heads (default: 8)
    dim_feedforward : int
        Feedforward dimension (default: 2048, which is 4 * embedding_dim)
    latent_size : int
        Dimension of latent space (default: 512)
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
        self.n_layers = n_layers
        
        # Positional embedding
        self.embedding = PositionalEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_len=max_len,
            padding_idx=padding_idx,
            dropout=dropout
        )
        
        # Transformer decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=embedding_dim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation='newgelu',
                layer_norm_eps=1e-9
            ) for _ in range(n_layers)
        ])
        
        # Generate causal mask for autoregressive generation
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        )
        
        # Output projection to vocabulary
        # LayerNorm without affine parameters, then linear layer
        self.output_norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.output_proj = nn.Linear(embedding_dim, vocab_size)
        nn.init.zeros_(self.output_proj.bias)
        
        # Initialize weights according to paper
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights according to paper specifications."""
        for layer in self.decoder_layers:
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
    
    def forward(self, latent: torch.Tensor, tgt: torch.Tensor,
                mode: str = 'forced') -> torch.Tensor:
        """
        Decode latent representation to SMILES token probabilities.
        
        Parameters
        ----------
        latent : torch.Tensor [batch_size, latent_size]
            Latent representation from encoder
        tgt : torch.Tensor [batch_size, length]
            Target token sequence (for teacher forcing) or None (for generation)
        mode : str
            'forced' for teacher forcing, 'generate' for autoregressive generation
        
        Returns
        -------
        output : torch.Tensor [batch_size, length, vocab_size]
            Logits over vocabulary for each position
        """
        if mode == 'forced':
            return self.forced_forward(latent, tgt)
        elif mode == 'generate':
            return self.generate_forward(latent, tgt)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'forced' or 'generate'.")
    
    def forced_forward(self, latent: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced forward pass (for training).
        
        Parameters
        ----------
        latent : torch.Tensor [batch_size, latent_size]
            Latent representation
        tgt : torch.Tensor [batch_size, length]
            Target token sequence
        
        Returns
        -------
        output : torch.Tensor [batch_size, length, vocab_size]
            Logits over vocabulary
        """
        batch_size, length = tgt.shape
        
        # Embed target sequence
        # Output: [length, batch_size, embedding_dim]
        tgt_emb = self.embedding(tgt)
        
        # Add latent representation to each position (broadcast)
        # latent: [batch_size, latent_size] -> [1, batch_size, latent_size]
        # If latent_size != embedding_dim, we need a projection
        if self.latent_size == self.embedding_dim:
            tgt_emb = tgt_emb + latent.unsqueeze(0)
        else:
            # Project latent to embedding dimension if needed
            if not hasattr(self, 'latent_proj'):
                self.latent_proj = nn.Linear(self.latent_size, self.embedding_dim).to(latent.device)
            latent_emb = self.latent_proj(latent).unsqueeze(0)
            tgt_emb = tgt_emb + latent_emb
        
        # Get causal mask for this sequence length
        attn_mask = self.causal_mask[:length, :length]
        
        # Decode through all layers
        output = tgt_emb
        for layer in self.decoder_layers:
            output = layer(output, attn_mask=attn_mask)
        
        # Transpose to [batch_size, length, embedding_dim]
        output = output.transpose(0, 1)
        
        # Project to vocabulary
        output = self.output_norm(output)
        output = self.output_proj(output)
        
        return output
    
    def generate_forward(self, latent: torch.Tensor, start_token: int = 1,
                        max_length: int = None) -> torch.Tensor:
        """
        Autoregressive generation (for inference).
        
        Parameters
        ----------
        latent : torch.Tensor [batch_size, latent_size]
            Latent representation
        start_token : int
            Start token index (default: 1)
        max_length : int
            Maximum generation length (default: self.max_len)
        
        Returns
        -------
        output : torch.Tensor [batch_size, max_length, vocab_size]
            Logits over vocabulary for each generated position
        """
        if max_length is None:
            max_length = self.max_len
        
        batch_size = latent.shape[0]
        device = latent.device
        
        # Initialize state for each layer
        states = [
            torch.zeros(0, batch_size, self.embedding_dim, device=device)
            for _ in range(self.n_layers)
        ]
        
        # Initialize with start token
        cur_input = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        outputs = []
        
        # Autoregressive generation
        for step in range(max_length):
            # Embed current token
            # Output: [1, batch_size, embedding_dim]
            cur_emb = self.embedding(cur_input, position=step)
            
            # Add latent representation
            if self.latent_size == self.embedding_dim:
                cur_emb = cur_emb + latent.unsqueeze(0)
            else:
                if not hasattr(self, 'latent_proj'):
                    self.latent_proj = nn.Linear(self.latent_size, self.embedding_dim).to(device)
                latent_emb = self.latent_proj(latent).unsqueeze(0)
                cur_emb = cur_emb + latent_emb
            
            # Decode through all layers with accumulated state
            output = cur_emb
            new_states = []
            for i, layer in enumerate(self.decoder_layers):
                output, new_state = layer.cell_forward(output, states[i])
                new_states.append(new_state)
            states = new_states
            
            # Project to vocabulary
            # Output: [1, batch_size, embedding_dim] -> [batch_size, 1, vocab_size]
            output = output.transpose(0, 1)
            output = self.output_norm(output)
            output = self.output_proj(output)
            
            outputs.append(output)
            
            # Get next token (greedy)
            next_token = torch.argmax(output, dim=-1)  # [batch_size, 1]
            cur_input = next_token
        
        # Concatenate all outputs
        output = torch.cat(outputs, dim=1)  # [batch_size, max_length, vocab_size]
        return output
    
    def prepare_generation(self, latent: torch.Tensor):
        """
        Prepare decoder state for autoregressive generation.
        
        Parameters
        ----------
        latent : torch.Tensor [batch_size, latent_size]
            Latent representation
        
        Returns
        -------
        states : list of torch.Tensor
            Initial states for each decoder layer
        """
        batch_size = latent.shape[0]
        device = latent.device
        
        states = [
            torch.zeros(0, batch_size, self.embedding_dim, device=device)
            for _ in range(self.n_layers)
        ]
        
        return states
    
    def step_generate(self, cur_input: torch.Tensor, latent: torch.Tensor,
                     states: list, position: int) -> tuple:
        """
        Single step of autoregressive generation.
        
        Parameters
        ----------
        cur_input : torch.Tensor [batch_size, 1]
            Current token indices
        latent : torch.Tensor [batch_size, latent_size]
            Latent representation
        states : list of torch.Tensor
            Current decoder states for each layer
        position : int
            Current generation position
        
        Returns
        -------
        output : torch.Tensor [batch_size, 1, vocab_size]
            Logits for current position
        new_states : list of torch.Tensor
            Updated decoder states
        """
        device = latent.device
        
        # Embed current token
        cur_emb = self.embedding(cur_input, position=position)
        
        # Add latent representation
        if self.latent_size == self.embedding_dim:
            cur_emb = cur_emb + latent.unsqueeze(0)
        else:
            if not hasattr(self, 'latent_proj'):
                self.latent_proj = nn.Linear(self.latent_size, self.embedding_dim).to(device)
            latent_emb = self.latent_proj(latent).unsqueeze(0)
            cur_emb = cur_emb + latent_emb
        
        # Decode through all layers
        output = cur_emb
        new_states = []
        for i, layer in enumerate(self.decoder_layers):
            output, new_state = layer.cell_forward(output, states[i])
            new_states.append(new_state)
        
        # Project to vocabulary
        output = output.transpose(0, 1)
        output = self.output_norm(output)
        output = self.output_proj(output)
        
        return output, new_states

