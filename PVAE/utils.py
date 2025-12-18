import torch.nn as nn
import torch.nn.functional as F

def celoss(x, x_recon, eps=1e-8): 
    
    """ x : Input Profiles
        x_recon : Reconstructed Profiles
    """
    x = x.clamp(min=eps)
    x = x / x.sum(dim=1, keepdim=True)
    log_q = F.log_softmax(x_recon, dim=1)
    ce = -(x * log_q).sum(dim=1)
    ce_loss = ce.mean()

    return ce_loss


class SelfAttention(nn.Module):
    def __init__(self, n_features, n_heads):
        super().__init__()
        self.n_features = n_features
        self.n_heads = n_heads
        self.self_attn = nn.MultiheadAttention(self.n_features, self.n_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(n_features)

    def forward(self, x):
        x_norm = self.layer_norm(x)
        out_attn, _ = self.self_attn(x_norm, x_norm, x_norm)
        attn_scaled_output = x + out_attn
        return attn_scaled_output