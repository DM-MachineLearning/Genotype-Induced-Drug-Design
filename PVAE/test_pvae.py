import numpy as np
import torch
from torch import optim
from PVAE2 import PVAE2
from PVAE import PVAE
from dataloader import return_dataloaders


def create_synthetic_data(
    n_samples: int = 128,
    n_cpg: int = 100,
    n_genes: int = 50,
    seed: int = 42,
):
    """
    Creates small synthetic methylation + gene expression matrices and chromosome filters.
      dna_meth: (n_samples, n_cpg) in [0, 1]
      gene_exp: (n_samples, n_genes) > 0 (TPM/FPKM-like)
      chrom_filters: (23, n_cpg) 
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    dna_meth = np.random.beta(a=2.0, b=5.0, size=(n_samples, n_cpg)).astype(np.float32)

    gene_exp = np.random.lognormal(mean=1.0, sigma=0.5, size=(n_samples, n_genes)).astype(np.float32)

    return dna_meth, gene_exp


def build_model(
    input_dim_dna: int,
    input_dim_gene: int,
    use_self_attn: bool = False,
):
    """
    Builds a PVAE2 model with some small hidden layer sizes.
    """
    hls_dna = (16, 32, 64)   
    hls_gene = (32, 64, 64)  
    hl_bottleneck = 32
    z_dim = 10

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


def main():
    
    n_samples = 128
    n_cpg = 100
    n_genes = 50

    dna_meth_np, gene_exp_np= create_synthetic_data(
        n_samples=n_samples,
        n_cpg=n_cpg,
        n_genes=n_genes,
        seed=42,
    )

    # np.save("synthetic_dna_meth.npy", dna_meth_np)
    # np.save("synthetic_gene_exp.npy", gene_exp_np)
    # print("Saved synthetic data to: synthetic_dna_meth.npy, synthetic_gene_exp.npy")

    print(dna_meth_np[:5,:5])
    print(gene_exp_np[:5,:5])

    dna_meth = torch.from_numpy(dna_meth_np)
    gene_exp = torch.from_numpy(gene_exp_np)

    train_loader, val_loader, test_loader = return_dataloaders(dna_meth, gene_exp, split_fractions=(0.6, 0.2))
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    model = build_model(
        input_dim_dna=n_cpg,
        input_dim_gene=n_genes,
        use_self_attn=False,  
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    x_dna_batch, x_gene_batch = next(iter(train_loader))
    x_dna_batch = x_dna_batch.to(device)
    x_gene_batch = x_gene_batch.to(device)

    recon_dna_list, recon_gene, mu, logvar = model(x_dna_batch, x_gene_batch)

    print("Forward pass successful.")
    print(f"Input DNA shape: {x_dna_batch.shape}")
    print(f"Input Gene shape: {x_gene_batch.shape}")
    print(f"Reconstructed gene shape: {recon_gene.shape}")
    print(f"Number of chromosome-specific DNA outputs: {len(recon_dna_list)}")
    print(f"mu shape: {mu.shape}, logvar shape: {logvar.shape}")

    loss = model.loss(
        x_dna_meth=x_dna_batch,
        x_gene_exp=x_gene_batch,
        recon_dna_meth=recon_dna_list,
        recon_gene_exp=recon_gene,
        mu=mu,
        logvar=logvar,
        lamb=1.0,
    )

    print(f"Single batch loss: {loss.item():.4f}")

    history = model.trainer(
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=5,
        device=device,
        lamb=1.0,
        log_interval=1,
        patience=3,
        verbose=True,
    )

if __name__ == "__main__":
    main()