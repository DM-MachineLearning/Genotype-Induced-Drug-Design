import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import celoss, SelfAttention

class PVAE2(nn.Module):

    def __init__(self, input_dim_dna : int, input_dim_gene : int, chromosome_filters, hls_dna : tuple, hls_gene : tuple, hl_bottleneck : int, z_dim : int, use_self_attn : bool = False):

        """ chromosome_filters : A 2D array (numpy) with chromosomes along rows and CpG sites along columns. 1 if CpG site lies on chromosome otherwise 0"""
        super().__init__()

        self.input_dim_dna = input_dim_dna
        self.input_dim_gene = input_dim_gene
        self.hls_dna = hls_dna
        self.hls_gene = hls_gene
        self.hl_bottleneck = hl_bottleneck
        self.z_dim = z_dim
        self.use_self_attn = use_self_attn


        chrm_filter = torch.from_numpy(chromosome_filters.astype(bool))
        self.register_buffer("chromosome_filters", chrm_filter)  
        self.chrom_indices = [self.chromosome_filters[i].nonzero(as_tuple=True)[0]
                      for i in range(23)]


        self.chromosome_hl_encoder = nn.ModuleList()
        self.chromosome_hl_decoder = nn.ModuleList()


        for i in range(23):
            n_cpg_chrm = int(chromosome_filters[i].sum())
            self.chromosome_hl_encoder.append(nn.Sequential(nn.Linear(n_cpg_chrm, hls_dna[0]),
                                               nn.LayerNorm(hls_dna[0]),
                                               nn.ReLU()))
            
            self.chromosome_hl_decoder.append(nn.Sequential(nn.Linear(hls_dna[0], n_cpg_chrm),
                                                       nn.LayerNorm(n_cpg_chrm),
                                                       nn.ReLU()))
            
        self.dna_mlp = nn.Sequential(nn.Linear(hls_dna[0]*23, hls_dna[1]),
                                     nn.LayerNorm(hls_dna[1]),
                                     nn.ReLU(),
                                     nn.Linear(hls_dna[1], hls_dna[2]))
            
        self.gene_mlp = nn.Sequential(nn.Linear(input_dim_gene, hls_gene[0]),
                                      nn.LayerNorm(hls_gene[0]),
                                      nn.ReLU(),
                                      nn.Linear(hls_gene[0], hls_gene[1]),
                                      nn.LayerNorm(hls_gene[1]),
                                      nn.ReLU(),
                                      nn.Linear(hls_gene[1], hls_gene[2]))
        
        if self.use_self_attn:
            self.self_attn = SelfAttention(hls_dna[2]+hls_gene[2], 8)

        
        # Bottleneck layer
        self.bottleneck_layer = nn.Sequential(nn.Linear(hls_dna[2]+hls_gene[2], hl_bottleneck),
                                              nn.LayerNorm(hl_bottleneck),
                                              nn.ReLU())
        

        self.mu_layer = nn.Linear(hl_bottleneck, z_dim)
        self.logvar_layer = nn.Linear(hl_bottleneck, z_dim)

        self.decoder_bottleneck_layer = nn.Sequential(nn.Linear(z_dim, hl_bottleneck),
                                                      nn.LayerNorm(hl_bottleneck),
                                                      nn.ReLU(),
                                                      nn.Linear(hl_bottleneck, hls_dna[2]+ hls_gene[2]),
                                                      nn.LayerNorm(hls_dna[2]+hls_gene[2]))

        self.dna_decoder_mlp = nn.Sequential(nn.Linear(hls_dna[2], hls_dna[1]),
                                             nn.LayerNorm(hls_dna[1]),
                                             nn.ReLU(),
                                             nn.Linear(hls_dna[1], hls_dna[0]*23),  
                                             nn.LayerNorm(hls_dna[0]*23),
                                             nn.ReLU())
        
        self.gene_decoder_mlp = nn.Sequential(nn.Linear(hls_gene[2], hls_gene[1]),
                                              nn.LayerNorm(hls_gene[1]),
                                              nn.ReLU(),
                                              nn.Linear(hls_gene[1], hls_gene[0]),
                                              nn.LayerNorm(hls_gene[0]),
                                              nn.ReLU(),
                                              nn.Linear(hls_gene[0], input_dim_gene))

    def encoder(self, x_dna, x_gene):

        dna_layer_1_out = None

        for i in range(23):
            idx = self.chrom_indices[i]
            out = self.chromosome_hl_encoder[i](x_dna[:, idx])
            if i==0:
                dna_layer_1_out = out
            else:
                dna_layer_1_out = torch.cat([dna_layer_1_out, out], dim=1)

        dna_out_2 = self.dna_mlp(dna_layer_1_out)
        gene_out = self.gene_mlp(x_gene)
        out_concat = torch.cat([dna_out_2, gene_out], dim=1)

        out = None

        if self.use_self_attn:
            out_concat = out_concat.unsqueeze(1)
            out_attn = self.self_attn(out_concat)
            out = self.bottleneck_layer(out_attn.squeeze(1))

        else:
            out = self.bottleneck_layer(out_concat)

        mu = self.mu_layer(out)
        logvar = self.logvar_layer(out)

        return mu, logvar

    def decoder(self, z):
        
        out = self.decoder_bottleneck_layer(z)
        out_dna, out_gene = torch.split(out, [self.hls_dna[2], self.hls_gene[2]], dim=1)
        out1 = self.dna_decoder_mlp(out_dna)
        out_chromosome_parts = torch.chunk(out1, chunks=23, dim=1)
        recon_dna_meth_profiles = []

        for i in range(23):
            recon_dna_meth_profiles.append(self.chromosome_hl_decoder[i](out_chromosome_parts[i]))

        recon_gene_exp = self.gene_decoder_mlp(out_gene)

        return recon_dna_meth_profiles, recon_gene_exp
          

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x_dna_meth, x_gene):
        mu, logvar = self.encoder(x_dna_meth, x_gene)
        z = self.reparameterize(mu, logvar)
        recon_dna_meth_profiles, recon_gene_exp = self.decoder(z)

        return recon_dna_meth_profiles, recon_gene_exp, mu, logvar

    def loss(self, x_dna_meth, x_gene_exp, recon_dna_meth, recon_gene_exp, mu, logvar, lamb=1):

        ce_meth_loss = 0
        for i in range(23):
            idx = self.chrom_indices[i]
            ce_meth_loss += celoss(x_dna_meth[:, idx], recon_dna_meth[i])

        ce_meth_loss /= 23

        ce_gene_loss = celoss(x_gene_exp, recon_gene_exp)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  
        kl_loss = kl.sum(dim=1).mean()
        total_loss = ce_meth_loss + ce_gene_loss + lamb*kl_loss

        return total_loss

    def trainer(self,
              train_loader,
              optimizer,
              num_epochs : int,
              device = None,
              lamb : float = 1.0,
              log_interval : int = 100,
              patience : int = 10,
              min_delta : float = 0.0,
              restore_best : bool = True,
              verbose : bool = True):
        
        """
        Docstring for trainer
        
        :param train_loader: Description
        :param optimizer: Description
        :param num_epochs: Description
        :type num_epochs: int
        :param device: Description
        :param lamb: Description
        :type lamb: float
        :param log_interval: Description
        :type log_interval: int
        :param patience: Description
        :type patience: int
        :param min_delta: Description
        :type min_delta: float
        :param restore_best: Description
        :type restore_best: bool
        :param verbose: Description
        :type verbose: bool
        """
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)

        history = []
        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, num_epochs+1):
            self.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (x_dna_meth, x_gene_exp) in enumerate(train_loader):
                x_dna_meth = x_dna_meth.to(device)
                x_gene_exp = x_gene_exp.to(device)

                optimizer.zero_grad()

                recon_dna_meth, recon_gene_exp, mu, logvar = self(x_dna_meth, x_gene_exp)

                loss = self.loss(
                    x_dna_meth=x_dna_meth,
                    x_gene_exp=x_gene_exp,
                    recon_dna_meth=recon_dna_meth,
                    recon_gene_exp=recon_gene_exp,
                    mu=mu,
                    logvar=logvar,
                    lamb=lamb,
                )

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if verbose and (batch_idx + 1) % log_interval == 0:
                    print(
                        f"Epoch [{epoch}/{num_epochs}] "
                        f"Batch [{batch_idx+1}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f}"
                    )

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            history.append(avg_epoch_loss)

            if verbose:
                print(f"Epoch [{epoch}/{num_epochs}] Avg Loss: {avg_epoch_loss:.4f}")

            if best_loss - avg_epoch_loss > min_delta:
                best_loss = avg_epoch_loss
                best_state = copy.deepcopy(self.state_dict())
                epochs_no_improve = 0
                if verbose:
                    print(f"New best loss: {best_loss:.4f}")

            else:
                epochs_no_improve += 1
                if verbose:
                    print(
                        f"No improvement for {epochs_no_improve} epochs "
                        f"(best: {best_loss:.4f})"
                    )
                if epochs_no_improve >= patience:
                    if verbose:
                        print(
                            f"Early stopping triggered after {epoch} epochs "
                            f"(patience={patience})."
                        )
                    break

        if restore_best and best_state is not None:
            self.load_state_dict(best_state)
            if verbose:
                print(f"Model weights restored to best epoch (loss={best_loss:.4f}).")

        return history
    
    def return_latent_var(self, x_dna_meth, x_gene):
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encoder(x_dna_meth, x_gene)
            z = self.reparameterize(mu, logvar)
        return z
    
    def save_model(self, path, optimizer=None, epoch=None, loss=None):

        save_dict = {
        "model_state_dict": self.state_dict()
        }

        if optimizer is not None:
            save_dict["optimizer_state_dict"] = optimizer.state_dict()

        if epoch is not None:
            save_dict["epoch"] = epoch

        if loss is not None:
         save_dict["loss"] = loss

        torch.save(save_dict, path)
        print(f"Model saved to: {path}")


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