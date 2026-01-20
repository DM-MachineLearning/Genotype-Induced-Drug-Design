# Usage : python -m Genotype_Induced_Drug_Design.PVAE.train_script

import numpy as np
import torch
from torch import optim
from Genotype_Induced_Drug_Design.PVAE.PVAE import PVAE
from Genotype_Induced_Drug_Design.PVAE.dataloader import return_dataloaders
import pickle


def build_model(
    hls_dna: tuple,
    hls_gene: tuple,
    hl_bottleneck: int,
    z_dim: int,
    input_dim_dna: int,
    input_dim_gene: int,
    use_self_attn: bool = False,
):
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
def evaluate(
    model,
    dataloader,
    device,
    lamb=1.0,
    free_bits=0.01,
    target_kl_low=0.01,
    target_kl_high=5.0,
    kl_warm_up=False,
    use_annealing=False,

    # -------- InfoVAE knobs --------
    use_infovae: bool = True,
    alpha: float = 0.0,
    lambda_info: float = 1.0,
    mmd_sigmas=None,
):
    """
    Returns avg total loss over dataloader (keeps original behavior).
    Compatible with InfoVAE loss that returns:
      total_loss, recon_meth, recon_exp, reg, beta, components_dict
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for x_dna_meth, x_gene_exp in dataloader:
        x_dna_meth = x_dna_meth.to(device)
        x_gene_exp = x_gene_exp.to(device)

        recon_dna, recon_gene, mu, logvar = model(x_dna_meth, x_gene_exp)

        loss, _, _, _, _, _ = model.loss(
            x_dna_meth=x_dna_meth,
            x_gene_exp=x_gene_exp,
            recon_dna_meth=recon_dna,
            recon_gene_exp=recon_gene,
            mu=mu,
            logvar=logvar,
            lamb=lamb,
            free_bits=free_bits,
            target_kl_low=target_kl_low,
            target_kl_high=target_kl_high,
            kl_warm_up=kl_warm_up,
            use_annealing=use_annealing,

            use_infovae=use_infovae,
            alpha=alpha,
            lambda_info=lambda_info,
            mmd_sigmas=mmd_sigmas,
        )

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    n_dna_meth = 15703
    n_genes = 15703

    with open(
        "/home/dmlab/Devendra/data/preprocessed_datasets/data_cancer_dm_tcga_tensor1.pkl",
        "rb",
    ) as f:
        dna_meth = pickle.load(f)

    with open(
        "/home/dmlab/Devendra/data/preprocessed_datasets/data_cancer_eg_tcga_tensor2.pkl",
        "rb",
    ) as f:
        gene_exp = pickle.load(f)

    dna_meth = dna_meth.to(dtype=torch.float32)
    gene_exp = gene_exp.to(dtype=torch.float32)

    train_loader, val_loader, test_loader = return_dataloaders(dna_meth, gene_exp)
    print(
        f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}"
    )

    model = build_model(
        (8192, 4096, 2048, 1024, 512, 256),
        (8192, 4096, 2048, 1024, 512, 256),
        256,
        z_dim=128,
        input_dim_dna=n_dna_meth,
        input_dim_gene=n_genes,
        use_self_attn=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    cycle_steps = 7 * len(train_loader)

    # =========================
    # InfoVAE toggles/params
    # =========================
    use_infovae = True
    alpha = 0.0          # (1-alpha)*KL_cond + (alpha+lambda-1)*MMD - lambda*MI
    lambda_info = 2.0    # try 1.0, 2.0, 5.0 if collapse persists

    # Optional: fixed kernel scales (can stabilize MMD)
    mmd_sigmas = None  # e.g., [0.5, 1.0, 2.0, 4.0, 8.0]

    history, mu_logvar_history, test_history = model.trainer(
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=100,
        device=device,
        lamb=1.0,
        free_bits=0.01,
        log_interval=1,
        patience=5,
        verbose=True,
        test_loader=test_loader,
        target_kl_low=1.0,
        target_kl_high=5.0,
        kl_warm_up=False,
        use_annealing=True,
        ratio_increase=0.7,
        cycle_steps=cycle_steps,
        beta_min=0.0,
        beta_max=1.0,

        # -------- InfoVAE --------
        use_infovae=use_infovae,
        alpha=alpha,
        lambda_info=lambda_info,
        mmd_sigmas=mmd_sigmas,
    )

    eval_train_loss = evaluate(
        model,
        train_loader,
        device,
        lamb=1.0,
        free_bits=0.01,
        target_kl_low=1.0,
        target_kl_high=5.0,
        kl_warm_up=False,

        use_infovae=use_infovae,
        alpha=alpha,
        lambda_info=lambda_info,
        mmd_sigmas=mmd_sigmas,
    )
    eval_val_loss = evaluate(
        model,
        val_loader,
        device,
        lamb=1.0,
        free_bits=0.01,
        target_kl_low=1.0,
        target_kl_high=5.0,
        kl_warm_up=False,

        use_infovae=use_infovae,
        alpha=alpha,
        lambda_info=lambda_info,
        mmd_sigmas=mmd_sigmas,
    )
    eval_test_loss = evaluate(
        model,
        test_loader,
        device,
        lamb=1.0,
        free_bits=0.01,
        target_kl_low=1.0,
        target_kl_high=5.0,
        kl_warm_up=False,

        use_infovae=use_infovae,
        alpha=alpha,
        lambda_info=lambda_info,
        mmd_sigmas=mmd_sigmas,
    )

    return history, mu_logvar_history, eval_train_loss, eval_val_loss, eval_test_loss


if __name__ == "__main__":
    history, mu_logvar_history, train_loss, val_loss, test_loss = main()
    print(f"Train Loss: {train_loss}, Val Loss: {val_loss}, Test Loss: {test_loss}")





