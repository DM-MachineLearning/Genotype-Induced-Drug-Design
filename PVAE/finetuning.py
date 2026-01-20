# Usage: python -m Genotype_Induced_Drug_Design.PVAE.finetuning

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pickle
import itertools

from Genotype_Induced_Drug_Design.PVAE.PVAE import PVAE
from Genotype_Induced_Drug_Design.PVAE.dataloader import return_dataloaders

def build_model(
    input_dim_dna,
    input_dim_gene,
    hls_dna,
    hls_gene,
    hl_bottleneck,
    z_dim,
    use_self_attn,
):
    return PVAE(
        input_dim_dna=input_dim_dna,
        input_dim_gene=input_dim_gene,
        hls_dna=hls_dna,
        hls_gene=hls_gene,
        hl_bottleneck=hl_bottleneck,
        z_dim=z_dim,
        use_self_attn=use_self_attn,
    )

@torch.no_grad()
def evaluate(model, dataloader, device, lamb=1.0):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for x_dna_meth, x_gene_exp in dataloader:
        x_dna_meth = x_dna_meth.to(device)
        x_gene_exp = x_gene_exp.to(device)

        recon_dna, recon_gene, mu, logvar = model(x_dna_meth, x_gene_exp)

        loss = model.loss(
            x_dna_meth=x_dna_meth,
            x_gene_exp=x_gene_exp,
            recon_dna_meth=recon_dna,
            recon_gene_exp=recon_gene,
            mu=mu,
            logvar=logvar,
            lamb=lamb,
        )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)

def main():

    with open("/home/dmlab/Devendra/data/preprocessed_datasets/data_cancer_dm_tcga_tensor1.pkl", "rb") as f:
        dna_meth = pickle.load(f).float()

    with open("/home/dmlab/Devendra/data/preprocessed_datasets/data_cancer_eg_tcga_tensor2.pkl", "rb") as f:
        gene_exp = pickle.load(f).float()

    train_loader, val_loader, test_loader = return_dataloaders(
        X_dna_meth=dna_meth,
        X_gene_exp=gene_exp,
        split_fractions=(0.8, 0.1),
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    input_dim_dna = dna_meth.shape[1]
    input_dim_gene = gene_exp.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    search_space = {
        "z_dim": [64, 128],
        "hl_bottleneck": [128, 256],
        "learning_rate": [1e-3, 5e-3, 1e-2],
        "lamb": [1.0],
        "use_self_attn": [False],
    }

    hls_dna = (2048, 512, 256)
    hls_gene = (2048, 512, 256)

    all_configs = list(itertools.product(*search_space.values()))
    keys = list(search_space.keys())

    results = []
    best_val_loss = float("inf")
    best_config = None

    for idx, values in enumerate(all_configs):
        config = dict(zip(keys, values))
        
        print(f"Running configuration {idx+1}/{len(all_configs)}: {config}")

        model = build_model(
            input_dim_dna=input_dim_dna,
            input_dim_gene=input_dim_gene,
            hls_dna=hls_dna,
            hls_gene=hls_gene,
            hl_bottleneck=config["hl_bottleneck"],
            z_dim=config["z_dim"],
            use_self_attn=config["use_self_attn"],
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        model.trainer(
            train_loader=train_loader,
            optimizer=optimizer,
            num_epochs=50,
            device=device,
            lamb=config["lamb"],
            patience=5,
            verbose=False,
        )

        val_loss = evaluate(model, val_loader, device, lamb=config["lamb"])
        test_loss = evaluate(model, test_loader, device, lamb=config["lamb"])

        print(f"Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")

        result = {
            "config": config,
            "val_loss": val_loss,
            "test_loss": test_loss,
        }

        results.append(result)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = config
            
            print(f"Best Configuration: {best_config} with Validation Loss: {best_val_loss:.4f}")

    torch.save(
        model.state_dict(),
       "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/results/best_finetuned_model.pt",
    )

    print("Best model saved.")

    with open(
        "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/results/finetuning_results.pkl",
        "wb",
    ) as f:
        pickle.dump(
            {
                "results": results,
                "best_config": best_config,
                "best_val_loss": best_val_loss,
            },
            f,
        )

    print("Completed")
    

if __name__ == "__main__":
    main()