import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from Genotype_Induced_Drug_Design.PVAE.utils import celoss, SelfAttention, _rbf_mmd, _kl_diag_gaussians, _mi_upper_bound_diag
import math


class PVAE(nn.Module):

    def __init__(self, input_dim_dna : int, input_dim_gene : int, hls_dna : tuple, hls_gene : tuple, hl_bottleneck : int, z_dim : int, use_self_attn=False):
        """ input_dim_meth : input dimension of a single DNA Methylation vector
            input_dim_gene : input dimension of a single gene expression vector
            hls_dna : tuple of length 5 --- > contains number of units in each hidden layer for dna methylation encoding
            hls_gene : tuple of length 5 --- > contains number of units in each hidden layer for gene expression encoding
            hl_bottleneck : int ----> Number of units in the bottleneck layer
            z_dim : int ----> Dimension of the latent space
            use_self_attn (False by default) ----> Adds a self attention layer in the encoder architecture
        """
        super().__init__()
        
        self.input_dim_dna = input_dim_dna
        self.input_dim_gene = input_dim_gene        
        self.hls_dna = hls_dna
        self.hls_gene = hls_gene
        self.hl_bottleneck = hl_bottleneck
        self.use_self_attn = use_self_attn
        self.z_dim = z_dim 

        self.dna_mlp = nn.Sequential(nn.Linear(input_dim_dna, hls_dna[0]),
                                         nn.LayerNorm(hls_dna[0]),
                                         nn.ReLU(),

                                         nn.Linear(hls_dna[0], hls_dna[1]),
                                         nn.LayerNorm(hls_dna[1]),
                                         nn.ReLU(),

                                         nn.Linear(hls_dna[1], hls_dna[2]),
                                         nn.LayerNorm(hls_dna[2]),
                                         nn.ReLU(),

                                         nn.Linear(hls_dna[2], hls_dna[3]),
                                         nn.LayerNorm(hls_dna[3]),
                                         nn.ReLU(),

                                         nn.Linear(hls_dna[3], hls_dna[4]),
                                         nn.LayerNorm(hls_dna[4]),
                                         nn.ReLU(),

                                         nn.Linear(hls_dna[4], hls_dna[5])
                                         )

        self.gene_mlp = nn.Sequential(nn.Linear(input_dim_gene, hls_gene[0]),
                                         nn.LayerNorm(hls_gene[0]),
                                         nn.ReLU(),

                                         nn.Linear(hls_gene[0], hls_gene[1]),
                                         nn.LayerNorm(hls_gene[1]),
                                         nn.ReLU(),

                                         nn.Linear(hls_gene[1], hls_gene[2]),
                                         nn.LayerNorm(hls_gene[2]),
                                         nn.ReLU(),
                                         
                                         nn.Linear(hls_gene[2], hls_gene[3]),
                                         nn.LayerNorm(hls_gene[3]),
                                         nn.ReLU(),
                                         
                                         nn.Linear(hls_gene[3], hls_gene[4]),
                                         nn.LayerNorm(hls_gene[4]),
                                         nn.ReLU(),
                                         
                                         nn.Linear(hls_gene[4], hls_gene[5]))
                                         
        
        if self.use_self_attn == True:
            self.self_attn = SelfAttention(hls_dna[5]+hls_gene[5], 8)

        self.bottleneck_layer = nn.Sequential(nn.Linear(hls_dna[5]+self.hls_gene[5] , hl_bottleneck),
                                              nn.LayerNorm(hl_bottleneck),
                                              nn.ReLU())

        self.mu_layer = nn.Linear(hl_bottleneck, z_dim)
        self.logvar_layer = nn.Linear(hl_bottleneck, z_dim)


        self.decoder_bottleneck_layer = nn.Sequential(nn.Linear(z_dim, hl_bottleneck),
                                                      nn.LayerNorm(hl_bottleneck),
                                                      nn.ReLU(),
                                                      nn.Linear(hl_bottleneck, hls_dna[5]+ hls_gene[5]),
                                                      nn.LayerNorm(hls_dna[5]+hls_gene[5]))
        
        self.deconcat_dna_meth = nn.Sequential(nn.Linear(hls_dna[5]+hls_gene[5], hls_dna[5]),
                                               nn.LayerNorm(hls_dna[5]),
                                               nn.ReLU())
        
        self.deconcat_gene_exp = nn.Sequential(nn.Linear(hls_dna[5]+hls_gene[5], hls_gene[5]),
                                               nn.LayerNorm(hls_gene[5]),
                                               nn.ReLU())

        self.decoder_dna_mlp = nn.Sequential(
            
                                             nn.Linear(hls_dna[5], hls_dna[4]),
                                             nn.LayerNorm(hls_dna[4]),
                                             nn.ReLU(),

                                             nn.Linear(hls_dna[4], hls_dna[3]),
                                             nn.LayerNorm(hls_dna[3]),
                                             nn.ReLU(),

                                             nn.Linear(hls_dna[3], hls_dna[2]),
                                             nn.LayerNorm(hls_dna[2]),
                                             nn.ReLU(),

                                             nn.Linear(hls_dna[2], hls_dna[1]),
                                             nn.LayerNorm(hls_dna[1]),
                                             nn.ReLU(),

                                             nn.Linear(hls_dna[1], hls_dna[0]),
                                             nn.LayerNorm(hls_dna[0]),
                                             nn.ReLU(),

                                             nn.Linear(hls_dna[0], input_dim_dna))

        self.decoder_gene_mlp = nn.Sequential(
            
                                              nn.Linear(hls_gene[5], hls_gene[4]),
                                              nn.LayerNorm(hls_gene[4]),
                                              nn.ReLU(),

                                              nn.Linear(hls_gene[4], hls_gene[3]),
                                              nn.LayerNorm(hls_gene[3]),
                                              nn.ReLU(),

                                              nn.Linear(hls_gene[3], hls_gene[2]),
                                              nn.LayerNorm(hls_gene[2]),
                                              nn.ReLU(),

                                              nn.Linear(hls_gene[2], hls_gene[1]),
                                              nn.LayerNorm(hls_gene[1]),
                                              nn.ReLU(),

                                              nn.Linear(hls_gene[1], hls_gene[0]),
                                              nn.LayerNorm(hls_gene[0]),
                                              nn.ReLU(),

                                              nn.Linear(hls_gene[0], input_dim_gene))


    def encoder(self, x_dna, x_gene):

        out_dna = self.dna_mlp(x_dna)
        out_gene = self.gene_mlp(x_gene)

        out_concat = torch.cat((out_dna, out_gene), dim=1)

        out = None

        if self.use_self_attn == True:
            out_concat = out_concat.unsqueeze(1)
            out_attn = self.self_attn(out_concat)
            out = self.bottleneck_layer(out_attn.squeeze(1))

        else:
            out = self.bottleneck_layer(out_concat)

        mu = self.mu_layer(out)
        logvar = self.logvar_layer(out)

        return mu, logvar

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decoder(self, z):

        out = self.decoder_bottleneck_layer(z)
        out_dna = self.deconcat_dna_meth(out)
        out_gene = self.deconcat_gene_exp(out)
        recon_dna_meth = self.decoder_dna_mlp(out_dna)
        recon_gene_exp = self.decoder_gene_mlp(out_gene)

        return recon_dna_meth, recon_gene_exp


    def forward(self, x_dna_meth, x_gene):
        mu, logvar = self.encoder(x_dna_meth, x_gene)
        z = self.reparameterize(mu, logvar)
        recon_dna_meth, recon_gene_exp = self.decoder(z)
        return recon_dna_meth, recon_gene_exp, mu, logvar


    def loss(
        self,
        x_dna_meth,
        x_gene_exp,
        recon_dna_meth,
        recon_gene_exp,
        mu,
        logvar,
        lamb=1.0,                 # global weight applied to the InfoVAE regularizer block
        free_bits=0.1,            # still used ONLY for conditional KL term (optional)
        target_kl_low=0.01,
        target_kl_high=5.0,
        kl_warm_up=False,
        use_annealing=False,

        global_step=None,
        cycle_steps=2000,
        ratio_increase=0.5,
        beta_min=0.0,
        beta_max=1.0,

        # -------------------
        # InfoVAE settings
        # -------------------
        use_infovae: bool = True,     # <--- NEW: make InfoVAE the main mode
        alpha: float = 0.0,           # InfoVAE alpha (controls conditional KL weight)
        lambda_info: float = 1.0,     # InfoVAE lambda (MI weight)
        mmd_sigmas=None,              # RBF kernel scales for MMD(q(z), p(z))
        eps: float = 1e-8,
    ):
        """
        InfoVAE (minimization form):

          total_loss = recon_loss + (lamb * beta) * reg

        where:
          recon_loss = recon_dna + recon_gene

          reg = (1 - alpha) * KL(q(z|x) || p(z))
                + (alpha + lambda_info - 1) * D(q(z) || p(z))
                - lambda_info * I_q(x; z)

        - We use D = MMD^2(q(z), p(z)) with samples z ~ q(z|x).
        - I_q(x;z) is approximated with diagonal-Gaussian q(z) estimate from the batch.

        Returns:
          total_loss, recon_dna, recon_gene, reg_loss, beta,
          plus a dict with components for logging.
        """

        # --- reconstruction losses ---
        recon_loss_dna_meth = celoss(x_dna_meth, recon_dna_meth)
        recon_loss_gene_exp = celoss(x_gene_exp, recon_gene_exp)
        recon_loss = recon_loss_dna_meth + recon_loss_gene_exp

        # --- beta schedule ---
        beta = 1.0
        if use_annealing:
            if global_step is None:
                raise ValueError("If use_annealing=True, you must pass global_step (int).")
            t = global_step % cycle_steps
            ramp_steps = max(1, int(ratio_increase * cycle_steps))
            if t < ramp_steps:
                frac = t / float(ramp_steps)
                beta = beta_min + frac * (beta_max - beta_min)
            else:
                beta = beta_max

        # --- Conditional KL(q(z|x)||p(z)) ---
        # diagonal Gaussian KL to N(0, I)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # (B, D)
        kl_mean_per_dim = kl.mean(dim=0)                      # (D,)
        if free_bits is not None and free_bits > 0:
            kl_mean_per_dim = torch.clamp(kl_mean_per_dim, min=free_bits)
        kl_cond = kl_mean_per_dim.sum()                       # scalar

        # Optional warm-up (only meaningful for KL_cond)
        if kl_warm_up:
            with torch.no_grad():
                if kl_cond.item() < target_kl_low:
                    lamb *= 0.9
                elif kl_cond.item() > target_kl_high:
                    lamb *= 1.1

        # --- Sample z ~ q(z|x) ---
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std                  # (B, D)
        z_prior = torch.randn_like(z)                         # (B, D)

        # --- MMD(q(z) || p(z)) ---
        mmd = _rbf_mmd(z, z_prior, sigmas=mmd_sigmas)         # scalar

        # --- MI term I_q(x;z) ≈ E KL(q(z|x)||q(z)) ---
        mi = _mi_upper_bound_diag(mu, logvar, eps=eps)        # scalar

        if not use_infovae:
            # Fallback: behave like your old "KL-only" objective (for debugging)
            reg = kl_cond
            components = {"kl_cond": kl_cond.detach(), "mmd": mmd.detach(), "mi": mi.detach()}
        else:
            # InfoVAE regularizer block
            reg = (1.0 - alpha) * kl_cond + (alpha + lambda_info - 1.0) * mmd - lambda_info * mi
            components = {"kl_cond": kl_cond.detach(), "mmd": mmd.detach(), "mi": mi.detach()}

        total_loss = recon_loss + (lamb * beta) * reg
        return total_loss, recon_loss_dna_meth, recon_loss_gene_exp, reg, beta, components

    def trainer(
        self,
        train_loader,
        optimizer,
        num_epochs: int,
        device=None,
        lamb: float = 1.0,
        free_bits: float = 0.1,
        log_interval: int = 100,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best: bool = True,
        verbose: bool = True,
        test_loader=None,
        target_kl_low: float = 0.01,
        target_kl_high: float = 5.0,
        kl_warm_up: bool = False,
        use_annealing: bool = False,
        ratio_increase: float = 0.5,
        cycle_steps: int = 2000,
        beta_min: float = 0.0,
        beta_max: float = 1.0,

        # InfoVAE knobs
        use_infovae: bool = True,
        alpha: float = 0.0,
        lambda_info: float = 1.0,
        mmd_sigmas=None,
    ):
        """
        Returns:
          history rows:
            [avg_total, avg_recon_meth, avg_recon_exp, avg_reg,
             avg_kl_cond, avg_mmd, avg_mi]
          mu_logvar_history as before
          test_history with same columns as history (if test_loader provided)
        """

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        history = []
        mu_logvar_history = []
        test_history = []

        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, num_epochs + 1):
            self.train()

            epoch_total = 0.0
            epoch_recon_m = 0.0
            epoch_recon_g = 0.0
            epoch_reg = 0.0
            epoch_klc = 0.0
            epoch_mmd = 0.0
            epoch_mi = 0.0
            num_batches = 0

            mu_mean_epoch = 0.0
            mu_std_epoch = 0.0
            mu_norm_epoch = 0.0
            logvar_mean_epoch = 0.0
            logvar_std_epoch = 0.0
            logvar_norm_epoch = 0.0

            for batch_idx, (x_dna_meth, x_gene_exp) in enumerate(train_loader):
                x_dna_meth = x_dna_meth.to(device)
                x_gene_exp = x_gene_exp.to(device)

                optimizer.zero_grad()
                recon_dna_meth, recon_gene_exp, mu, logvar = self(x_dna_meth, x_gene_exp)

                global_step = (epoch - 1) * len(train_loader) + batch_idx

                loss, rec_m, rec_g, reg, beta, comp = self.loss(
                    x_dna_meth=x_dna_meth,
                    x_gene_exp=x_gene_exp,
                    recon_dna_meth=recon_dna_meth,
                    recon_gene_exp=recon_gene_exp,
                    mu=mu,
                    logvar=logvar,
                    lamb=lamb,
                    free_bits=free_bits,
                    target_kl_low=target_kl_low,
                    target_kl_high=target_kl_high,
                    kl_warm_up=kl_warm_up,
                    use_annealing=use_annealing,
                    global_step=global_step,
                    cycle_steps=cycle_steps,
                    ratio_increase=ratio_increase,
                    beta_min=beta_min,
                    beta_max=beta_max,
                    use_infovae=use_infovae,
                    alpha=alpha,
                    lambda_info=lambda_info,
                    mmd_sigmas=mmd_sigmas,
                )

                loss.backward()
                optimizer.step()

                epoch_total += loss.item()
                epoch_recon_m += rec_m.item()
                epoch_recon_g += rec_g.item()
                epoch_reg += reg.item()
                epoch_klc += float(comp["kl_cond"])
                epoch_mmd += float(comp["mmd"])
                epoch_mi += float(comp["mi"])
                num_batches += 1

                mu_mean_epoch += mu.mean().item()
                mu_std_epoch += mu.std().item()
                mu_norm_epoch += mu.norm(dim=1).mean().item()
                logvar_mean_epoch += logvar.mean().item()
                logvar_std_epoch += logvar.std().item()
                logvar_norm_epoch += logvar.norm(dim=1).mean().item()

                if verbose and (batch_idx + 1) % log_interval == 0:
                    print(
                        f"Epoch [{epoch}/{num_epochs}]  "
                        f"Batch [{batch_idx+1}/{len(train_loader)}]  "
                        f"Beta: {beta:.4f}  "
                        f"Loss: {loss.item():.4f}  "
                        f"Recon(M): {rec_m.item():.4f}  Recon(G): {rec_g.item():.4f}  "
                        f"Reg: {reg.item():.4f}  "
                        f"KLc: {float(comp['kl_cond']):.4f}  "
                        f"MMD: {float(comp['mmd']):.4f}  "
                        f"MI: {float(comp['mi']):.4f}  "
                        f"mu_std: {mu.std().item():.4f}  logvar_std: {logvar.std().item():.4f}"
                    )

            denom = max(num_batches, 1)
            avg_total = epoch_total / denom
            avg_rec_m = epoch_recon_m / denom
            avg_rec_g = epoch_recon_g / denom
            avg_reg = epoch_reg / denom
            avg_klc = epoch_klc / denom
            avg_mmd = epoch_mmd / denom
            avg_mi = epoch_mi / denom

            history.append([avg_total, avg_rec_m, avg_rec_g, avg_reg, avg_klc, avg_mmd, avg_mi])

            mu_logvar_history.append([
                (mu_mean_epoch / denom),
                (mu_std_epoch / denom),
                (mu_norm_epoch / denom),
                (logvar_mean_epoch / denom),
                (logvar_std_epoch / denom),
                (logvar_norm_epoch / denom),
            ])

            # -----------------------
            # Test evaluation (no annealing / warm-up)
            # -----------------------
            if test_loader is not None:
                self.eval()
                t_total = t_rec_m = t_rec_g = t_reg = t_klc = t_mmd = t_mi = 0.0
                t_batches = 0

                with torch.no_grad():
                    for x_dna_meth, x_gene_exp in test_loader:
                        x_dna_meth = x_dna_meth.to(device)
                        x_gene_exp = x_gene_exp.to(device)

                        recon_dna_meth, recon_gene_exp, mu, logvar = self(x_dna_meth, x_gene_exp)

                        loss, rec_m, rec_g, reg, _, comp = self.loss(
                            x_dna_meth=x_dna_meth,
                            x_gene_exp=x_gene_exp,
                            recon_dna_meth=recon_dna_meth,
                            recon_gene_exp=recon_gene_exp,
                            mu=mu,
                            logvar=logvar,
                            lamb=lamb,
                            free_bits=free_bits,
                            target_kl_low=target_kl_low,
                            target_kl_high=target_kl_high,
                            kl_warm_up=False,
                            use_annealing=False,
                            use_infovae=use_infovae,
                            alpha=alpha,
                            lambda_info=lambda_info,
                            mmd_sigmas=mmd_sigmas,
                        )

                        t_total += loss.item()
                        t_rec_m += rec_m.item()
                        t_rec_g += rec_g.item()
                        t_reg += reg.item()
                        t_klc += float(comp["kl_cond"])
                        t_mmd += float(comp["mmd"])
                        t_mi += float(comp["mi"])
                        t_batches += 1

                d = max(t_batches, 1)
                test_history.append([
                    t_total / d, t_rec_m / d, t_rec_g / d, t_reg / d, t_klc / d, t_mmd / d, t_mi / d
                ])

            if verbose:
                msg = (
                    f"Epoch [{epoch}/{num_epochs}]  "
                    f"Train: Total {avg_total:.4f} | ReconM {avg_rec_m:.4f} ReconG {avg_rec_g:.4f} | "
                    f"Reg {avg_reg:.4f} (KLc {avg_klc:.4f}, MMD {avg_mmd:.4f}, MI {avg_mi:.4f})"
                )
                if test_loader is not None:
                    tt = test_history[-1]
                    msg += (
                        f"\n             Test : Total {tt[0]:.4f} | ReconM {tt[1]:.4f} ReconG {tt[2]:.4f} | "
                        f"Reg {tt[3]:.4f} (KLc {tt[4]:.4f}, MMD {tt[5]:.4f}, MI {tt[6]:.4f})"
                    )
                print(msg)

            # Early stopping on TRAIN total loss (same logic as before)
            if best_loss - avg_total > min_delta:
                best_loss = avg_total
                best_state = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
                if verbose:
                    print(f"New best train loss: {best_loss:.4f}")
            else:
                epochs_no_improve += 1
                if verbose:
                    print(f"No improvement for {epochs_no_improve} epochs (best train: {best_loss:.4f})")
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping triggered after {epoch} epochs (patience={patience}).")
                    break

        if restore_best and best_state is not None:
            self.load_state_dict(best_state)
            if verbose:
                print(f"Model weights restored to best epoch (Avg Train Total loss={best_loss:.4f}).")

        return history, mu_logvar_history, test_history


    def return_latent_var(self, x_dna_meth, x_gene):

        self.eval()
        device = next(self.parameters()).device

        x_dna_meth = x_dna_meth.to(device).float()
        x_gene = x_gene.to(device).float()

        with torch.no_grad():
            mu, logvar = self.encoder(x_dna_meth, x_gene)
            z = self.reparameterize(mu, logvar)

        return z

    def save_model(self, model, path):
        torch.save(model.to("cpu").state_dict(), path)


    def load_model(self, path, optimizer=None, map_location=None):

        checkpoint = torch.load(path, map_location=map_location)

        self.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        meta = {
            "epoch": checkpoint.get("epoch", None),
            "loss": checkpoint.get("loss", None)
        }

        print(f"Model loaded from: {path}")

        return meta

