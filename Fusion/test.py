# Usage : python -m Genotype_Induced_Drug_Design.Fusion.test

import torch
import numpy as np
import sys
import os
from Genotype_Induced_Drug_Design.Fusion.FuseMix import FuseMix

def generate_toy_data(n_cells=50, n_drugs=30, d_gen=128, d_smiles=756):
    """
    Generates synthetic random data for testing.
    """
    print(f"\nGenerators Toy Data: {n_cells} Cell Lines, {n_drugs} Drugs")
    
    G = torch.randn(n_cells, d_gen)
    S = torch.randn(n_drugs, d_smiles)
    pair_matrix = torch.randint(0, 2, (n_cells, n_drugs)).float()
    
    print(f"  - G shape: {G.shape}")
    print(f"  - S shape: {S.shape}")
    print(f"  - Matrix shape: {pair_matrix.shape}")
    print(f"  - Positive interactions: {pair_matrix.sum().item()}")
    
    return G, S, pair_matrix

def test_training_loop():

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    print(f"\n--- Testing on device: {device} ---")

    INPUT_DIM_GEN = 128
    INPUT_DIM_SMILES = 756
    HIDDEN_DIM = 128
    OUTPUT_DIM = 64
    
    G, S, pair_matrix = generate_toy_data(d_gen=INPUT_DIM_GEN, d_smiles=INPUT_DIM_SMILES)
    
    G = G.to(device)
    S = S.to(device)
    pair_matrix = pair_matrix.to(device)


    model = FuseMix(
        input_dim_genetic=INPUT_DIM_GEN,
        input_dim_smiles=INPUT_DIM_SMILES,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        device=device
    ).to(device)
    
    print("Model initialized.")

    print("--- Testing Forward Pass (Mixup) ---") # Test Forward Pass Logic manually
    try:
        z_g = torch.randn(10, INPUT_DIM_GEN).to(device) # Creating dummy data
        z_s = torch.randn(10, INPUT_DIM_SMILES).to(device)
        
        out_g, out_s = model(z_g, z_s, z_g, z_s) # Mixing with itself for shape check
        
        assert out_g.shape == (10, OUTPUT_DIM), f"Expected shape (10, {OUTPUT_DIM}), got {out_g.shape}"
        assert out_s.shape == (10, OUTPUT_DIM), f"Expected shape (10, {OUTPUT_DIM}), got {out_s.shape}"
        print(f"Forward pass output shapes correct: {out_g.shape}")
        
    except Exception as e:
        print(f"Forward pass failed: {e}")
        raise e

    print("\n--- Testing Trainer Method ---")  # Testing the Trainer Method
    try:
        final_loss = model.trainer(
            G, S, pair_matrix, 
            epochs=20,    
            lr=1e-3
        )
        print(f"Training finished successfully. Final Loss: {final_loss:.4f}")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Test Inference
    print("\n--- Testing Inference (return_latent_var) ---")
    try:
        lat_g, lat_s = model.return_latent_var(G, S)
        
        # Check shapes
        assert lat_g.shape[0] == G.shape[0], "Latent G batch size mismatch"
        assert lat_s.shape[0] == S.shape[0], "Latent S batch size mismatch"
        assert lat_g.shape[1] == OUTPUT_DIM, "Latent G dimension mismatch"
        
        print("Inference shapes correct.")
        print("FuseMix Test Complete!")
        
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    test_training_loop()