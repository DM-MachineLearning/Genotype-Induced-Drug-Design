# Usage : python -m Genotype_Induced_Drug_Design.PVAE.train_script

import numpy as np
import torch
from torch import optim
from Genotype_Induced_Drug_Design.PVAE.PVAE import PVAE
from Genotype_Induced_Drug_Design.PVAE.dataloader import return_dataloaders
import pickle

def build_model(
    input_dim_dna: int,
    input_dim_gene: int,
    use_self_attn: bool = False,
):

    hls_dna = (2048, 512, 256)   
    hls_gene = (2048, 512, 256)
    hl_bottleneck = 256
    z_dim = 128

    model = PVAE(
        input_dim_dna=input_dim_dna,
        input_dim_gene=input_dim_gene,
        hls_dna=hls_dna,
        hls_gene=hls_gene,
        hl_bottleneck=hl_bottleneck,
        z_dim=z_dim,
        use_self_attn=use_self_attn,
    )
    return model

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
    
    n_dna_meth = 15703
    n_genes = 15703

    with open("/home/dmlab/Devendra/data/preprocessed_datasets/data_cancer_dm_tcga_tensor1.pkl", "rb") as f:
        dna_meth = pickle.load(f)
    with open("/home/dmlab/Devendra/data/preprocessed_datasets/data_cancer_eg_tcga_tensor2.pkl", "rb") as f:
        gene_exp = pickle.load(f)

    dna_meth = dna_meth.to(dtype=torch.float32)
    gene_exp = gene_exp.to(dtype=torch.float32)

    train_loader, val_loader, test_loader = return_dataloaders(dna_meth, gene_exp)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    model = build_model(
        input_dim_dna=n_dna_meth,
        input_dim_gene=n_genes,
        use_self_attn=False,  
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    meta = model.trainer(
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=100,
        device=device,
        lamb=1.0,
        log_interval=1,
        patience=5,
        verbose=True,
    )

    train_loss = evaluate(model, train_loader, device)
    val_loss = evaluate(model, val_loader, device)
    test_loss = evaluate(model, test_loader, device)

    model.save_model(model, "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/results/model_state_without_attn.pt")
    
    return train_loss, val_loss, test_loss

if __name__ == "__main__":
   train_loss, val_loss, test_loss = main()
   print(f"Train Loss: {train_loss}, Val Loss: {val_loss}, Test Loss: {test_loss}")


