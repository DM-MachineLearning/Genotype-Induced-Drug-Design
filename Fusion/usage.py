"""
Complete workflow: Pretrained SMILES Encoder + Profile VAE Fusion
"""

import torch
from smiles_encoder import SMILESEncoder
from Genotype_Induced_Drug_Design.PVAE.PVAE import PVAE
from latent_fusion import LatentFusion


class GenotypeGuidedDrugDesign(nn.Module):
    """
    Complete model that fuses SMILES and Profile latents for drug generation.
    """
    
    def __init__(
        self,
        smiles_encoder: SMILESEncoder,
        profile_vae: PVAE,
        fusion_method: str = "gaussian_add",
        smiles_latent_size: int = 512,
        profile_latent_size: int = 512,
    ):
        super().__init__()
        
        # Load pretrained encoders
        self.smiles_encoder = smiles_encoder
        self.profile_vae = profile_vae
        
        # Freeze pretrained encoders (optional)
        # self.smiles_encoder.requires_grad_(False)
        # self.profile_vae.requires_grad_(False)
        
        # Fusion module
        self.fusion = LatentFusion(
            smiles_latent_size=smiles_latent_size,
            profile_latent_size=profile_latent_size,
            fusion_method=fusion_method,
        )
        
        # Decoder for molecule generation (from your existing decoder)
        # self.decoder = SMILESDecoder(...)
    
    def encode_smiles(self, smiles_tokens: torch.Tensor):
        """Encode SMILES to latent."""
        return self.smiles_encoder(smiles_tokens)
    
    def encode_profile(self, dna_features: torch.Tensor, gene_features: torch.Tensor):
        """Encode profile to latent."""
        return self.profile_vae.encoder(dna_features, gene_features)
    
    def fuse_latents(
        self,
        mu_smiles: torch.Tensor,
        var_smiles: torch.Tensor,
        mu_profile: torch.Tensor,
        logvar_profile: torch.Tensor,
    ):
        """Fuse SMILES and Profile latents."""
        return self.fusion(mu_smiles, var_smiles, mu_profile, logvar_profile)
    
    def forward(
        self,
        smiles_tokens: torch.Tensor,
        dna_features: torch.Tensor,
        gene_features: torch.Tensor,
    ):
        """
        Complete forward pass: Encode both modalities and fuse.
        
        Parameters
        ----------
        smiles_tokens : torch.Tensor [batch_size, seq_len]
        dna_features : torch.Tensor [batch_size, dna_dim]
        gene_features : torch.Tensor [batch_size, gene_dim]
        
        Returns
        -------
        mu_fused : torch.Tensor [batch_size, fused_latent_size]
        var_fused : torch.Tensor [batch_size, fused_latent_size]
        """
        # Encode SMILES
        mu_smiles, var_smiles = self.encode_smiles(smiles_tokens)
        
        # Encode Profile
        mu_profile, logvar_profile = self.encode_profile(dna_features, gene_features)
        
        # Fuse latents
        mu_fused, var_fused = self.fuse_latents(
            mu_smiles, var_smiles, mu_profile, logvar_profile
        )
        
        return mu_fused, var_fused


# Usage example:
def main():
    # Load pretrained SMILES encoder
    smiles_encoder = SMILESEncoder(
        vocab_size=45,
        embedding_dim=512,
        n_layers=8,
        latent_size=512,
        max_len=122,
    )
    
    # Load pretrained weights
    checkpoint = torch.load("checkpoints/encoder_pretrain/best_encoder.pt")
    smiles_encoder.load_state_dict(checkpoint["encoder_state"])
    
    # Initialize Profile VAE
    profile_vae = PVAE(
        input_dim_dna=1000,  # Adjust to your data
        input_dim_gene=20000,  # Adjust to your data
        hls_dna=(512, 256, 128),
        hls_gene=(1024, 512, 256),
        hl_bottleneck=256,
        z_dim=512,  # Match SMILES latent_size for fusion
    )
    
    # Create fusion model
    model = GenotypeGuidedDrugDesign(
        smiles_encoder=smiles_encoder,
        profile_vae=profile_vae,
        fusion_method="gaussian_add",  # or "concat", "weighted", etc.
        smiles_latent_size=512,
        profile_latent_size=512,
    )
    
    # Example forward pass
    batch_size = 16
    smiles_tokens = torch.randint(0, 45, (batch_size, 100))
    dna_features = torch.randn(batch_size, 1000)
    gene_features = torch.randn(batch_size, 20000)
    
    mu_fused, var_fused = model(smiles_tokens, dna_features, gene_features)
    
    print(f"Fused latent shape: {mu_fused.shape}")
    print(f"Fused variance shape: {var_fused.shape}")
    
    # Sample from fused latent for decoding
    epsilon = torch.randn_like(mu_fused)
    z_fused = mu_fused + torch.sqrt(var_fused + 1e-8) * epsilon
    
    # Use with decoder for molecule generation
    # logits = decoder(z_fused, ...)


if __name__ == "__main__":
    main()