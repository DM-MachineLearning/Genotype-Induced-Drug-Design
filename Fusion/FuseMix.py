# Usage : python -m Genotype_Induced_Drug_Design.Fusion.FuseMix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from Genotype_Induced_Drug_Design.Fusion.utils import get_positive_negative_indices, build_pairwise_embeddings


class FuseMix(nn.Module):
    def __init__(
        self,
        input_dim_genetic=128,
        input_dim_smiles=756,
        hidden_dim=512,
        output_dim=256,
        alpha=1.0,
        beta=1.0,
        temperature=0.1,
        device="cuda",
    ):
        super().__init__()
        self.device = device

        self.genetic_mlp = nn.Sequential(
            nn.Linear(input_dim_genetic, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

        self.smiles_mlp = nn.Sequential(
            nn.Linear(input_dim_smiles, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

        self.register_buffer('alpha', torch.tensor(alpha))
        self.register_buffer('beta', torch.tensor(beta))
        self.dist = Beta(self.alpha, self.beta)
        self.temperature = temperature
        self.criterion = nn.BCEWithLogitsLoss()



    def forward(self, z_gen_1, z_sm_1, z_gen_2, z_sm_2):
        """
        Takes raw features, projects them, and performs Mixup.
        """
        z_gen_1 = self.genetic_mlp(z_gen_1)
        z_gen_2 = self.genetic_mlp(z_gen_2)
        z_sm_1 = self.smiles_mlp(z_sm_1)
        z_sm_2 = self.smiles_mlp(z_sm_2)
        B = z_gen_1.size(0) 
        
        lamb = self.dist.sample((B,)).view(B, 1).to(self.device) 

        z_g = lamb * z_gen_1 + (1 - lamb) * z_gen_2
        z_s = lamb * z_sm_1 + (1 - lamb) * z_sm_2

        return z_g, z_s
    


    def trainer(self, G, S, pair_matrix, epochs=100, lr=1e-4):
        """
        Iteratively trains the model using Adam.
        
        Args:
            G: (N_cell_lines, d_gen)
            S: (N_drugs, d_smiles)
            pair_matrix: (N_cell_lines, N_drugs) Binary interaction matrix
            epochs: Number of training iterations
            lr: Learning rate
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        pos_pairs, neg_pairs = get_positive_negative_indices(pair_matrix)
        z_gen_pos_raw, z_sm_pos_raw = build_pairwise_embeddings(G, S, pos_pairs)
        z_gen_neg_raw, z_sm_neg_raw = build_pairwise_embeddings(G, S, neg_pairs)

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            optimizer.zero_grad()

            perm_pos = torch.randperm(z_gen_pos_raw.size(0), device=self.device)   # Random Permutations for Mixup (Must happen every epoch)
            perm_neg = torch.randperm(z_gen_neg_raw.size(0), device=self.device)

            
            z_g_pos_mixed, z_s_pos_mixed = self.forward( # Positives
                z_gen_pos_raw, 
                z_sm_pos_raw,
                z_gen_pos_raw[perm_pos], 
                z_sm_pos_raw[perm_pos]
            )

          
            z_g_neg_mixed, z_s_neg_mixed = self.forward(   # Negatives
                z_gen_neg_raw, 
                z_sm_neg_raw,
                z_gen_neg_raw[perm_neg], 
                z_sm_neg_raw[perm_neg]
            )

            z_g_pos_norm = F.normalize(z_g_pos_mixed, p=2, dim=1)
            z_s_pos_norm = F.normalize(z_s_pos_mixed, p=2, dim=1)
            z_g_neg_norm = F.normalize(z_g_neg_mixed, p=2, dim=1)
            z_s_neg_norm = F.normalize(z_s_neg_mixed, p=2, dim=1)

            sim_pos = (z_g_pos_norm * z_s_pos_norm).sum(dim=1) / self.temperature
            sim_neg = (z_g_neg_norm * z_s_neg_norm).sum(dim=1) / self.temperature

    
            logits = torch.cat([sim_pos, sim_neg], dim=0)
            
          
            targets = torch.cat([
                torch.ones_like(sim_pos), 
                torch.zeros_like(sim_neg)
            ], dim=0)

            loss = self.criterion(logits, targets)

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

        return loss.item()

    @torch.no_grad()
    def return_latent_var(self, G, S):
        self.eval()
        z_g = self.genetic_mlp(G)
        z_s = self.smiles_mlp(S)
        return z_g, z_s

