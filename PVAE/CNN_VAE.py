import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from Genotype_Induced_Drug_Design.PVAE.utils import celoss, apply_block_masking
import math

class CNNVAE(nn.Module):
    def __init__(self, input_dim: int, z_dim: int, num_classes: int = 2):
        super().__init__()
        self.input_length = input_dim 
        self.z_dim = z_dim
        self.num_classes = num_classes
        
        self.encoder_conv = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Conv1d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16), nn.LeakyReLU(),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32), nn.LeakyReLU(),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.LeakyReLU(),
            
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.LeakyReLU(),

            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.LeakyReLU(),
        )
        
        self.flatten_size = 64 * 491 
        
        self.mu_layer = nn.Linear(self.flatten_size, z_dim)
        self.logvar_layer = nn.Linear(self.flatten_size, z_dim)

        self.decoder_input = nn.Linear(z_dim, self.flatten_size)
        
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(),
            
            nn.ConvTranspose1d(64, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64), nn.LeakyReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32), nn.LeakyReLU(),
            
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(16), nn.LeakyReLU(),
            
            nn.ConvTranspose1d(16, 2, kernel_size=5, stride=2, padding=2, output_padding=0),
        )

        self.classifier_mlp = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, num_classes)
        )

    def encoder(self, x_dna, x_gene):
        x = torch.stack([x_dna, x_gene], dim=1)
        out = self.encoder_conv(x)
        out_flat = out.view(out.size(0), -1) 
        mu = self.mu_layer(out_flat)
        logvar = self.logvar_layer(out_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decoder(self, z):
        h = self.decoder_input(z)
        h = h.view(h.size(0), 64, 491)
        recon = self.decoder_conv(h)
        if recon.size(2) != self.input_length:
            recon = recon[:, :, :self.input_length]
        return recon[:, 0, :], recon[:, 1, :]

    def forward(self, x_dna_meth, x_gene):
        mu, logvar = self.encoder(x_dna_meth, x_gene)
        z = self.reparameterize(mu, logvar)
        recon_dna, recon_gene = self.decoder(z)
        return recon_dna, recon_gene, mu, logvar

    def forward_with_classifier(self, x_dna_meth, x_gene):
        mu, logvar = self.encoder(x_dna_meth, x_gene)
        z = self.reparameterize(mu, logvar)
        recon_dna, recon_gene = self.decoder(z)
        class_logits = self.classifier_mlp(z)
        return recon_dna, recon_gene, mu, logvar, class_logits

    def loss(self, x_dna, x_gene, r_dna, r_gene, mu, logvar, 
             labels=None, preds=None, lamb=1.0, alpha=10.0):
        r_l_dna = celoss(x_dna.unsqueeze(1), r_dna.unsqueeze(1))
        r_l_gene = celoss(x_gene.unsqueeze(1), r_gene.unsqueeze(1))
        recon_loss = r_l_dna + r_l_gene
        
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl)
        
        cls_loss = torch.tensor(0.0, device=x_dna.device)
        if labels is not None and preds is not None:
            loss_fn = nn.CrossEntropyLoss()
            target = labels.view(-1).long()
            cls_loss = loss_fn(preds, target)

        total_loss = recon_loss + (lamb * kl_loss) + (alpha * cls_loss)
        return total_loss, r_l_dna, r_l_gene, kl_loss, cls_loss

    def trainer(
        self,
        train_loader,
        optimizer,
        num_epochs: int,
        device=None,
        lamb: float = 1.0,
        alpha: float = 20.0,
        log_interval: int = 100,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best: bool = True,
        verbose: bool = True,
        test_loader=None,
        apply_masking: bool = False,
        mask_ratio: float = 0.2,
        block_size: int = 200
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        history = []
        mu_logvar_history = []
        test_history = []

        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        warmup_epochs = 20

        for epoch in range(1, num_epochs + 1):
            self.train()

            epoch_total = 0.0
            epoch_recon_m = 0.0
            epoch_recon_g = 0.0
            epoch_kl = 0.0
            epoch_cls = 0.0
            epoch_acc = 0.0
            num_batches = 0

            mu_mean_epoch = 0.0
            mu_std_epoch = 0.0
            logvar_mean_epoch = 0.0
            logvar_std_epoch = 0.0
            
            if epoch < warmup_epochs:
                current_lamb = (epoch / warmup_epochs) * lamb
            else:
                current_lamb = lamb

            for batch_idx, (x_dna_meth, x_gene_exp, labels) in enumerate(train_loader):
                x_dna_meth = x_dna_meth.to(device)
                x_gene_exp = x_gene_exp.to(device)
                labels = labels.to(device)

                if apply_masking:
                    x_dna_in = apply_block_masking(x_dna_meth, mask_ratio, block_size)
                    x_gene_in = apply_block_masking(x_gene_exp, mask_ratio, block_size)
                else:
                    x_dna_in = x_dna_meth
                    x_gene_in = x_gene_exp

                optimizer.zero_grad()
                recon_dna_meth, recon_gene_exp, mu, logvar, class_logits = self.forward_with_classifier(x_dna_in, x_gene_in)

                loss, rec_m, rec_g, kl, cls_loss = self.loss(
                    x_dna=x_dna_meth, 
                    x_gene=x_gene_exp, 
                    r_dna=recon_dna_meth,
                    r_gene=recon_gene_exp,
                    mu=mu,
                    logvar=logvar,
                    labels=labels,
                    preds=class_logits,
                    lamb=current_lamb,
                    alpha=alpha
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_total += loss.item()
                epoch_recon_m += rec_m.item()
                epoch_recon_g += rec_g.item()
                epoch_kl += kl.item()
                epoch_cls += cls_loss.item()
                
                preds_cls = torch.argmax(class_logits, dim=1)
                acc = (preds_cls == labels.view(-1)).float().mean().item()
                epoch_acc += acc
                
                num_batches += 1
                mu_mean_epoch += mu.mean().item()
                mu_std_epoch += mu.std().item()
                logvar_mean_epoch += logvar.mean().item()
                logvar_std_epoch += logvar.std().item()

                if verbose and (batch_idx + 1) % log_interval == 0:
                    print(
                        f"mu_norm: {mu.norm(dim=1).mean().item():.3f} logv_norm: {logvar.norm(dim=1).mean().item():.3f} | "
                        f"Acc: {acc:.4f} ClsLoss: {cls_loss.item():.4f}"
                    )

            denom = max(num_batches, 1)
            avg_total = epoch_total / denom
            avg_rec_m = epoch_recon_m / denom
            avg_rec_g = epoch_recon_g / denom
            avg_kl = epoch_kl / denom
            avg_cls = epoch_cls / denom
            avg_acc = epoch_acc / denom

            history.append([avg_total, avg_rec_m, avg_rec_g, avg_kl, avg_cls, avg_acc])

            mu_logvar_history.append([
                (mu_mean_epoch / denom),
                (mu_std_epoch / denom),
                (logvar_mean_epoch / denom),
                (logvar_std_epoch / denom),
            ])

            if test_loader is not None:
                self.eval()
                t_total = t_rec_m = t_rec_g = t_kl = t_cls = t_acc = 0.0
                t_batches = 0

                with torch.no_grad():
                    for x_dna_meth, x_gene_exp, labels in test_loader:
                        x_dna_meth = x_dna_meth.to(device)
                        x_gene_exp = x_gene_exp.to(device)
                        labels = labels.to(device)

                        recon_dna_meth, recon_gene_exp, mu, logvar, class_logits = self.forward_with_classifier(x_dna_meth, x_gene_exp)

                        loss, rec_m, rec_g, kl, cls_loss = self.loss(
                            x_dna=x_dna_meth,
                            x_gene=x_gene_exp,
                            r_dna=recon_dna_meth,
                            r_gene=recon_gene_exp,
                            mu=mu,
                            logvar=logvar,
                            labels=labels,
                            preds=class_logits,
                            lamb=current_lamb,
                            alpha=alpha
                        )

                        t_total += loss.item()
                        t_rec_m += rec_m.item()
                        t_rec_g += rec_g.item()
                        t_kl += kl.item()
                        t_cls += cls_loss.item()
                        
                        preds_cls = torch.argmax(class_logits, dim=1)
                        t_acc += (preds_cls == labels.view(-1)).float().mean().item()
                        t_batches += 1

                d = max(t_batches, 1)
                test_history.append([t_total / d, t_rec_m / d, t_rec_g / d, t_kl / d, t_cls / d, t_acc / d])

            if verbose:
                msg = (
                    f"Epoch [{epoch}/{num_epochs}] Train: Tot {avg_total:.2f} | ReconM {avg_rec_m:.4f} | ReconG {avg_rec_g:.4f} | KL {avg_kl:.2f} | Cls {avg_cls:.3f} (Acc {avg_acc:.3f})"
                )
                if test_loader is not None:
                    tt = test_history[-1]
                    msg += (
                        f"\n             Test : Tot {tt[0]:.2f} | KL {tt[3]:.2f} | Cls {tt[4]:.3f} (Acc {tt[5]:.3f})"
                    )
                print(msg)

            if best_loss - avg_total > min_delta:
                best_loss = avg_total
                best_state = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"Early stopping triggered after {epoch} epochs.")
                    break

        if restore_best and best_state is not None:
            self.load_state_dict(best_state)
            if verbose:
                print(f"Model weights restored to best epoch.")

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
    

