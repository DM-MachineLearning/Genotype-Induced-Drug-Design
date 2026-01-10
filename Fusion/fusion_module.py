"""
Latent Space Fusion Module for Genotype-Induced Drug Design

Fuses SMILES encoder latent space with Profile VAE latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class LatentFusion(nn.Module):
    """
    Fuses SMILES encoder latent space with Profile VAE latent space.
    
    Supports multiple fusion strategies:
    1. Gaussian Addition: z = z_p + z_c
    2. Concatenation: z = [z_p; z_c]
    3. Weighted Combination: z = α*z_p + (1-α)*z_c
    4. Cross-Attention: Cross-modal attention mechanism
    5. Product of Experts: Bayesian fusion of distributions
    """
    
    def __init__(
        self,
        smiles_latent_size: int,
        profile_latent_size: int,
        fused_latent_size: Optional[int] = None,
        fusion_method: str = "gaussian_add",
        learnable_weight: bool = False,
    ):
        """
        Parameters
        ----------
        smiles_latent_size : int
            Dimension of SMILES encoder latent (e.g., 512)
        profile_latent_size : int
            Dimension of Profile VAE latent (e.g., z_dim from PVAE)
        fused_latent_size : int, optional
            Output dimension after fusion. If None, uses fusion_method default
        fusion_method : str
            Fusion strategy: "gaussian_add", "concat", "weighted", "cross_attn", "product_of_experts"
        learnable_weight : bool
            Whether to learn fusion weights (for weighted combination)
        """
        super().__init__()
        
        self.smiles_latent_size = smiles_latent_size
        self.profile_latent_size = profile_latent_size
        self.fusion_method = fusion_method
        
        # Determine output dimension
        if fused_latent_size is None:
            if fusion_method == "concat":
                fused_latent_size = smiles_latent_size + profile_latent_size
            elif fusion_method in ["gaussian_add", "weighted", "product_of_experts"]:
                # Assume same dimension (requires alignment)
                if smiles_latent_size != profile_latent_size:
                    raise ValueError(
                        f"For {fusion_method}, latent sizes must match. "
                        f"Got {smiles_latent_size} and {profile_latent_size}"
                    )
                fused_latent_size = smiles_latent_size
            elif fusion_method == "cross_attn":
                fused_latent_size = smiles_latent_size  # Profile attends to SMILES
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        self.fused_latent_size = fused_latent_size
        
        # Projection layers if dimensions don't match
        if smiles_latent_size != profile_latent_size and fusion_method != "concat":
            self.smiles_proj = nn.Linear(smiles_latent_size, fused_latent_size)
            self.profile_proj = nn.Linear(profile_latent_size, fused_latent_size)
        else:
            self.smiles_proj = nn.Identity()
            self.profile_proj = nn.Identity()
        
        # Fusion-specific layers
        if fusion_method == "weighted":
            if learnable_weight:
                self.alpha = nn.Parameter(torch.tensor(0.5))
            else:
                self.register_buffer("alpha", torch.tensor(0.5))
        
        elif fusion_method == "cross_attn":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=fused_latent_size,
                num_heads=8,
                batch_first=False,
            )
            # Project profile to query, SMILES to key/value
            self.profile_query = nn.Linear(profile_latent_size, fused_latent_size)
        
        elif fusion_method == "concat":
            # Project concatenated features to fused dimension
            if self.fused_latent_size != (smiles_latent_size + profile_latent_size):
                self.fusion_proj = nn.Sequential(
                    nn.Linear(smiles_latent_size + profile_latent_size, fused_latent_size),
                    nn.LayerNorm(fused_latent_size),
                    nn.ReLU(),
                )
            else:
                self.fusion_proj = nn.Identity()
    
    def forward(
        self,
        mu_smiles: torch.Tensor,
        var_smiles: torch.Tensor,
        mu_profile: torch.Tensor,
        logvar_profile: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse SMILES and Profile latent representations.
        
        Parameters
        ----------
        mu_smiles : torch.Tensor [batch_size, smiles_latent_size]
            Mean of SMILES latent distribution
        var_smiles : torch.Tensor [batch_size, smiles_latent_size]
            Variance of SMILES latent distribution
        mu_profile : torch.Tensor [batch_size, profile_latent_size]
            Mean of Profile latent distribution
        logvar_profile : torch.Tensor [batch_size, profile_latent_size]
            Log variance of Profile latent distribution
        sample : bool
            Whether to sample from distributions or use means
            
        Returns
        -------
        mu_fused : torch.Tensor [batch_size, fused_latent_size]
            Mean of fused latent distribution
        var_fused : torch.Tensor [batch_size, fused_latent_size]
            Variance of fused latent distribution
        """
        batch_size = mu_smiles.shape[0]
        
        # Convert logvar to var for profile
        var_profile = torch.exp(logvar_profile)
        
        # Project to common dimension if needed
        mu_s = self.smiles_proj(mu_smiles)
        var_s = self.smiles_proj(var_smiles) if self.smiles_proj is not nn.Identity() else var_smiles
        
        mu_p = self.profile_proj(mu_profile)
        var_p = self.profile_proj(var_profile) if self.profile_proj is not nn.Identity() else var_profile
        
        # Sample or use means
        if sample:
            eps_s = torch.randn_like(mu_s)
            eps_p = torch.randn_like(mu_p)
            z_s = mu_s + torch.sqrt(var_s + 1e-8) * eps_s
            z_p = mu_p + torch.sqrt(var_p + 1e-8) * eps_p
        else:
            z_s = mu_s
            z_p = mu_p
        
        # Apply fusion method
        if self.fusion_method == "gaussian_add":
            mu_fused, var_fused = self._gaussian_addition(mu_s, var_s, mu_p, var_p)
        
        elif self.fusion_method == "concat":
            mu_fused, var_fused = self._concatenation(z_s, z_p, var_s, var_p)
        
        elif self.fusion_method == "weighted":
            mu_fused, var_fused = self._weighted_combination(mu_s, var_s, mu_p, var_p)
        
        elif self.fusion_method == "cross_attn":
            mu_fused, var_fused = self._cross_attention(z_s, z_p, mu_s, mu_p, var_s, var_p)
        
        elif self.fusion_method == "product_of_experts":
            mu_fused, var_fused = self._product_of_experts(mu_s, var_s, mu_p, var_p)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return mu_fused, var_fused
    
    def _gaussian_addition(
        self,
        mu_s: torch.Tensor,
        var_s: torch.Tensor,
        mu_p: torch.Tensor,
        var_p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gaussian Addition: z ~ N(μ_p + μ_c, σ_p² + σ_c²)
        """
        mu_fused = mu_s + mu_p
        var_fused = var_s + var_p
        return mu_fused, var_fused
    
    def _concatenation(
        self,
        z_s: torch.Tensor,
        z_p: torch.Tensor,
        var_s: torch.Tensor,
        var_p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Concatenation: z = [z_p; z_c], doubles dimension
        """
        z_fused = torch.cat([z_s, z_p], dim=-1)
        var_fused = torch.cat([var_s, var_p], dim=-1)
        
        # Project if needed
        mu_fused = self.fusion_proj(z_fused)
        
        # For variance, we take mean or project similarly
        # (variance concatenation may need different handling)
        if isinstance(self.fusion_proj, nn.Identity):
            return z_fused, var_fused
        else:
            # Approximate: project variance similarly
            var_proj = self.fusion_proj(var_fused)
            return mu_fused, var_proj
    
    def _weighted_combination(
        self,
        mu_s: torch.Tensor,
        var_s: torch.Tensor,
        mu_p: torch.Tensor,
        var_p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Weighted Combination: z = α*z_p + (1-α)*z_c
        """
        alpha = torch.sigmoid(self.alpha) if hasattr(self, 'alpha') else 0.5
        mu_fused = alpha * mu_p + (1 - alpha) * mu_s
        var_fused = alpha * var_p + (1 - alpha) * var_s
        return mu_fused, var_fused
    
    def _cross_attention(
        self,
        z_s: torch.Tensor,
        z_p: torch.Tensor,
        mu_s: torch.Tensor,
        mu_p: torch.Tensor,
        var_s: torch.Tensor,
        var_p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-Attention: Profile queries SMILES features
        """
        # Profile as query, SMILES as key/value
        query = self.profile_query(mu_p).unsqueeze(0)  # [1, batch, dim]
        key = value = mu_s.unsqueeze(0)  # [1, batch, dim]
        
        # Cross-attention
        attn_out, _ = self.cross_attn(query, key, value)
        mu_fused = attn_out.squeeze(0)  # [batch, dim]
        
        # For variance, use weighted combination based on attention
        # (simplified - could use attention weights)
        var_fused = (var_s + var_p) / 2.0
        
        return mu_fused, var_fused
    
    def _product_of_experts(
        self,
        mu_s: torch.Tensor,
        var_s: torch.Tensor,
        mu_p: torch.Tensor,
        var_p: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Product of Experts: p(z) ∝ p_p(z) * p_c(z)
        
        For Gaussian experts:
        μ = (μ_p/σ_p² + μ_c/σ_c²) / (1/σ_p² + 1/σ_c²)
        σ² = 1 / (1/σ_p² + 1/σ_c²)
        """
        # Add small epsilon to avoid division by zero
        eps = 1e-8
        var_s = var_s + eps
        var_p = var_p + eps
        
        # Precision (inverse variance)
        prec_s = 1.0 / var_s
        prec_p = 1.0 / var_p
        
        # Combined precision and mean
        prec_fused = prec_s + prec_p
        var_fused = 1.0 / prec_fused
        
        mu_fused = (mu_s * prec_s + mu_p * prec_p) / prec_fused
        
        return mu_fused, var_fused