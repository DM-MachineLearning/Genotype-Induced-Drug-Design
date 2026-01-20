import torch
import torch.nn as nn
import torch.nn.functional as F


def celoss(x, x_recon, eps=1e-8): 
    """ 
    Computes Normalized Cross-Entropy.
    
    Args:
        x : Input Profiles (Target). Raw intensities/counts. 
            Shape: (Batch, Genes) or (Batch, 1, Genes)
        x_recon : Reconstructed Logits (Prediction). 
            Shape: Match x
    """
    x = x.clamp(min=eps)
    x_prob = x / x.sum(dim=-1, keepdim=True)
    log_q = F.log_softmax(x_recon, dim=-1)
    ce = -(x_prob * log_q).sum(dim=-1)
    ce_loss = ce.mean()

    return ce_loss

def _rbf_mmd(x: torch.Tensor, y: torch.Tensor, sigmas=None, eps: float = 1e-8) -> torch.Tensor:
    """
    Unbiased-ish (actually commonly-used biased) MMD^2 with multi-kernel RBF.
    x, y: (B, D)
    Returns: scalar MMD^2
    """
    # Pairwise squared distances
    xx = torch.cdist(x, x, p=2).pow(2)  # (B,B)
    yy = torch.cdist(y, y, p=2).pow(2)  # (B,B)
    xy = torch.cdist(x, y, p=2).pow(2)  # (B,B)

    # Median heuristic for base bandwidth (stop-grad to avoid instability)
    with torch.no_grad():
        # Use xy (cross) distances; more stable than xx/yy when collapse happens
        median_sq = torch.median(xy)
        # Fallback if median is 0 (can happen early / with collapse)
        if median_sq.item() <= 0:
            median_sq = torch.tensor(1.0, device=x.device, dtype=x.dtype)

    base = median_sq.detach() + eps

    if sigmas is None:
        # Multiplicative scales around the median heuristic
        sigmas = [0.5, 1.0, 2.0, 4.0, 8.0]

    K_xx = 0.0
    K_yy = 0.0
    K_xy = 0.0
    for s in sigmas:
        gamma = 1.0 / (2.0 * (s * base))
        K_xx = K_xx + torch.exp(-gamma * xx)
        K_yy = K_yy + torch.exp(-gamma * yy)
        K_xy = K_xy + torch.exp(-gamma * xy)

    # Biased estimator (includes diagonal); widely used and stable
    mmd2 = K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()
    return mmd2

def _kl_diag_gaussians(mu_q, logvar_q, mu_p, var_p, eps=1e-8):
    """
    KL( N(mu_q, diag(var_q)) || N(mu_p, diag(var_p)) ), averaged over batch.
    mu_q/logvar_q: (B, D)
    mu_p: (D,)
    var_p: (D,)  (NOT logvar)
    """
    var_q = torch.exp(logvar_q) + eps
    var_p = var_p + eps
    # KL per sample
    kl = 0.5 * (
        torch.log(var_p) - torch.log(var_q)
        + (var_q + (mu_q - mu_p).pow(2)) / var_p
        - 1.0
    ).sum(dim=1)
    return kl.mean()


def _mi_upper_bound_diag(mu, logvar, eps=1e-8):
    """
    Approx I_q(x;z) = E_x KL(q(z|x) || q(z))
    with q(z) approximated as a diagonal Gaussian using the law of total variance:
      q(z) ~ N( mean(mu),  Var(mu) + E[Var(z|x)] )
    """
    mu_agg = mu.mean(dim=0)  # (D,)
    var_agg = mu.var(dim=0, unbiased=False) + torch.exp(logvar).mean(dim=0)  # (D,)
    mi = _kl_diag_gaussians(mu, logvar, mu_agg, var_agg, eps=eps)
    return mi


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
    

def apply_block_masking(x, mask_ratio=0.2, block_size=200):
    """
    Applies block masking to a tensor (Batch, Length) or (Batch, Channels, Length).
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
        
    # Ensure we work with a clone so we don't modify the original
    masked_x = x.clone()
    
    # Handle dimensions: Ensure we have (Batch, ..., Length)
    if x.dim() == 2: # (Batch, Length)
        B, L = x.shape
        # Create mask for (Batch, Length)
        mask = torch.ones(B, L, device=x.device)
        num_blocks = int((L * mask_ratio) // block_size)
        
        for b in range(B):
            if num_blocks > 0:
                start_indices = torch.randint(0, L - block_size, (num_blocks,))
                for start in start_indices:
                    mask[b, start : start + block_size] = 0
        
        masked_x = masked_x * mask
        
    elif x.dim() == 3: # (Batch, Channels, Length)
        B, C, L = x.shape
        mask = torch.ones(B, C, L, device=x.device)
        num_blocks = int((L * mask_ratio) // block_size)
        
        for b in range(B):
            if num_blocks > 0:
                start_indices = torch.randint(0, L - block_size, (num_blocks,))
                for start in start_indices:
                    # Mask all channels at this location
                    mask[b, :, start : start + block_size] = 0
        
        masked_x = masked_x * mask

    return masked_x


def gaussian_noise_aug(x: torch.Tensor, noise_level=0.1, output_size_factor=2):
    """
    Augments the input tensor x by creating copies with added Gaussian noise.
    
    Args:
        x (torch.Tensor): Input data tensor of shape (N, ...).
        noise_level (float): Standard deviation of the Gaussian noise.
        output_size_factor (int): Factor by which to multiply the dataset size.
                                  (e.g., 2 means output size is 2 * input size).
                                  Must be >= 1.
    
    Returns:
        torch.Tensor: Augmented tensor of shape (N * output_size_factor, ...).
    """
    if output_size_factor < 1:
        raise ValueError("output_size_factor must be >= 1")
    
    # 1. Start with the original data
    augmented_data = [x]
    
    # 2. Generate (factor - 1) noisy copies
    for _ in range(output_size_factor - 1):
        noise = torch.randn_like(x) * noise_level
        noisy_x = x + noise
        augmented_data.append(noisy_x)
        
    # 3. Concatenate all copies
    return torch.cat(augmented_data, dim=0)