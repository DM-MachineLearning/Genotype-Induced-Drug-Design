import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import celoss, SelfAttention


class pVAE(nn.Module):

    def __init__(self, input_dim_dna : int, input_dim_gene : int, hls_dna : tuple, hls_gene : tuple, hl_bottleneck : int, z_dim : int, use_self_attn=False):
        """ input_dim_meth : input dimension of a single DNA Methylation vector
            input_dim_gene : input dimension of a single gene expression vector
            hls_dna : tuple of length 3 --- > contains number of units in each hidden layer for dna methylation encoding
            hls_gene : tuple of length 3 --- > contains number of units in each hidden layer for gene expression encoding
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
                                         nn.Linear(hls_dna[1], hls_dna[2]))

        self.gene_mlp = nn.Sequential(nn.Linear(input_dim_gene, hls_gene[0]),
                                         nn.LayerNorm(hls_gene[0]),
                                         nn.ReLU(),
                                         nn.Linear(hls_gene[0], hls_gene[1]),
                                         nn.LayerNorm(hls_gene[1]),
                                         nn.ReLU(),
                                         nn.Linear(hls_gene[1], hls_gene[2]))

        
        if self.use_self_attn == True:
            self.self_attn = SelfAttention(hls_dna[2]+hls_gene[2], 8)

        self.bottleneck_layer = nn.Sequential(nn.Linear(hls_dna[2]+self.hls_gene[2] , hl_bottleneck),
                                              nn.LayerNorm(hl_bottleneck),
                                              nn.ReLU())

        self.mu_layer = nn.Linear(hl_bottleneck, z_dim)
        self.logvar_layer = nn.Linear(hl_bottleneck, z_dim)


        self.decoder_bottleneck_layer = nn.Sequential(nn.Linear(z_dim, hl_bottleneck),
                                                      nn.LayerNorm(hl_bottleneck),
                                                      nn.ReLU(),
                                                      nn.Linear(hl_bottleneck, hls_dna[2]+ hls_gene[2]),
                                                      nn.LayerNorm(hls_dna[2]+hls_gene[2]))

        self.decoder_dna_mlp = nn.Sequential(nn.Linear(hls_dna[2], hls_dna[1]),
                                             nn.LayerNorm(hls_dna[1]),
                                             nn.ReLU(),
                                             nn.Linear(hls_dna[1], hls_dna[0]),
                                             nn.LayerNorm(hls_dna[0]),
                                             nn.ReLU(),
                                             nn.Linear(hls_dna[0], input_dim_dna))

        self.decoder_gene_mlp = nn.Sequential(nn.Linear(hls_gene[2], hls_gene[1]),
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
        out_dna, out_gene = torch.split(out, [self.hls_dna[2], self.hls_gene[2]], dim=1)
        recon_dna_meth = self.decoder_dna_mlp(out_dna)
        recon_gene_exp = self.decoder_gene_mlp(out_gene)

        return recon_dna_meth, recon_gene_exp
    
    
    def forward(self, x_dna_meth, x_gene):
        mu, logvar = self.encoder(x_dna_meth, x_gene)
        z = self.reparameterize(mu, logvar)
        recon_dna_meth, recon_gene_exp = self.decoder(z)
        return recon_dna_meth, recon_gene_exp, mu, logvar

    
    def loss(self, x_dna_meth, x_gene_exp, recon_dna_meth, recon_gene_exp, mu, logvar, lamb=1):

        recon_loss_dna_meth = celoss(x_dna_meth, recon_dna_meth)
        recon_loss_gene_exp = celoss(x_gene_exp, recon_gene_exp)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  
        kl_loss = kl.sum(dim=1).mean()
        total_loss = recon_loss_dna_meth + recon_loss_gene_exp + lamb * kl_loss

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
        
        if device == None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.to(device)

        history = []
        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0

        for epoch in range(1, epoch+1):
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



    

        

