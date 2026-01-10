"""
Encoder-only pretraining for the SMILES Transformer encoder.

Workflow:
    1) Fetch SMILES:        python fetch_chembl.py --out_csv data/chembl/chembl_smiles.csv --limit 200000
    2) Tokenize SMILES:     python preprocess_chembl.py --input_csv data/chembl/chembl_smiles.csv --output_dir data/chembl/tokenized
    3) Pretrain encoder:    python pretrain_encoder.py --data_dir data/chembl/tokenized --checkpoint_dir checkpoints/encoder_pretrain
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from smiles_encoder import SMILESEncoder
from train_chembl import load_split

from datetime import datetime

class SimpleDecoderHead(nn.Module):
    """
    Lightweight decoder head that uses the encoder's mean vector (mu)
    to reconstruct token logits for every position.
    """

    def __init__(self, latent_size: int, max_len: int, vocab_size: int, dropout: float = 0.0):
        super().__init__()
        self.max_len = max_len
        self.positional = nn.Embedding(max_len, latent_size)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(latent_size, vocab_size)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        mu : torch.Tensor [batch_size, latent_size]

        Returns
        -------
        logits : torch.Tensor [batch_size, max_len, vocab_size]
        """
        batch_size, latent_size = mu.shape
        # Expand mu across sequence length and add positional embedding
        mu_expanded = mu.unsqueeze(1).expand(batch_size, self.max_len, latent_size)
        positions = torch.arange(self.max_len, device=mu.device).unsqueeze(0).expand(batch_size, -1)
        hidden = mu_expanded + self.positional(positions)
        hidden = self.dropout(hidden)
        logits = self.proj(hidden)
        return logits


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, Dict]:
    train_obj = load_split(os.path.join(data_dir, "train.pt"))
    val_obj = load_split(os.path.join(data_dir, "val.pt"))
    meta = train_obj["meta"]

    train_ds = TensorDataset(train_obj["data"])
    val_ds = TensorDataset(val_obj["data"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, meta


def prepare_batch(batch: Tuple[torch.Tensor], device: torch.device) -> torch.Tensor:
    return batch[0].to(device)


def save_checkpoint(path: str, encoder: nn.Module, meta: Dict, step: int, val_loss: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "meta": meta,
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )
    print(f"[INFO] Saved checkpoint to {path} (step={step}, val_loss={val_loss:.4f})")


def plot_losses(train_losses: List[float], val_losses: List[float], output_path: str) -> None:
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved loss plot to {output_path}")
    plt.close()


def compute_metrics(encoder: nn.Module, head: nn.Module, dataloader: DataLoader, 
                   device: torch.device, meta: Dict, num_samples: int = 1000) -> Dict:
    """Compute comprehensive evaluation metrics for encoder capacity."""
    encoder.eval()
    head.eval()
    
    pad_id = meta["pad_id"]
    vocab = meta["vocab"]
    
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    correct_sequences = 0
    total_sequences = 0
    
    mu_list = []
    var_list = []
    
    ce_loss = nn.CrossEntropyLoss(ignore_index=pad_id, reduction='sum')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx * dataloader.batch_size >= num_samples:
                break
                
            tokens = batch[0].to(device)
            batch_size = tokens.shape[0]
            
            mu, var = encoder(tokens)
            logits = head(mu)
            
            # Loss and perplexity
            loss = ce_loss(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))
            total_loss += loss.item()
            
            # Count non-padding tokens
            mask = (tokens != pad_id)
            total_tokens += mask.sum().item()
            
            # Token-level accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct_tokens += ((predictions == tokens) & mask).sum().item()
            
            # Sequence-level accuracy (exact match, ignoring padding)
            for i in range(batch_size):
                seq_mask = tokens[i] != pad_id
                if seq_mask.sum() > 0:
                    seq_pred = predictions[i][seq_mask]
                    seq_true = tokens[i][seq_mask]
                    if torch.equal(seq_pred, seq_true):
                        correct_sequences += 1
                    total_sequences += 1
            
            # Collect latent statistics
            mu_list.append(mu.cpu())
            var_list.append(var.cpu())
    
    # Compute metrics
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(avg_loss)
    token_accuracy = correct_tokens / max(total_tokens, 1)
    sequence_accuracy = correct_sequences / max(total_sequences, 1)
    
    # Latent space statistics
    all_mu = torch.cat(mu_list, dim=0)
    all_var = torch.cat(var_list, dim=0)
    
    mu_mean = all_mu.mean(dim=0).numpy()
    mu_std = all_mu.std(dim=0).numpy()
    var_mean = all_var.mean(dim=0).numpy()
    
    # Compute latent space diversity (mean pairwise distance)
    sample_size = min(500, all_mu.shape[0])
    sample_indices = torch.randperm(all_mu.shape[0])[:sample_size]
    mu_sample = all_mu[sample_indices]
    
    # Pairwise L2 distances
    mu_expanded1 = mu_sample.unsqueeze(1)  # [sample_size, 1, latent_size]
    mu_expanded2 = mu_sample.unsqueeze(0)  # [1, sample_size, latent_size]
    pairwise_distances = torch.norm(mu_expanded1 - mu_expanded2, dim=2)
    # Exclude diagonal (self-distances)
    mask = ~torch.eye(sample_size, dtype=torch.bool)
    mean_pairwise_distance = pairwise_distances[mask].mean().item()
    
    metrics = {
        "loss": avg_loss,
        "perplexity": perplexity,
        "token_accuracy": token_accuracy,
        "sequence_accuracy": sequence_accuracy,
        "latent_stats": {
            "mu_mean": float(mu_mean.mean()),
            "mu_std": float(mu_std.mean()),
            "mu_std_per_dim": mu_std.tolist(),
            "var_mean": float(var_mean.mean()),
            "mean_pairwise_distance": mean_pairwise_distance,
        },
        "num_samples_evaluated": total_sequences,
    }
    
    return metrics


def print_metrics(metrics: Dict) -> None:
    """Print evaluation metrics in a formatted way."""
    print("\n" + "="*70)
    print("ENCODER CAPACITY EVALUATION METRICS")
    print("="*70)
    print(f"\n📊 Reconstruction Performance:")
    print(f"  • Loss:              {metrics['loss']:.4f}")
    print(f"  • Perplexity:        {metrics['perplexity']:.2f}")
    print(f"  • Token Accuracy:    {metrics['token_accuracy']*100:.2f}%")
    print(f"  • Sequence Accuracy: {metrics['sequence_accuracy']*100:.2f}%")
    
    print(f"\n🔬 Latent Space Statistics:")
    latent = metrics['latent_stats']
    print(f"  • Mean (mu):         {latent['mu_mean']:.4f}")
    print(f"  • Std (mu):          {latent['mu_std']:.4f}")
    print(f"  • Mean Variance:     {latent['var_mean']:.4f}")
    print(f"  • Pairwise Distance: {latent['mean_pairwise_distance']:.4f}")
    
    print(f"\n📈 Samples Evaluated: {metrics['num_samples_evaluated']}")
    print("="*70 + "\n")


def plot_latent_distribution(mu_samples: torch.Tensor, output_path: str, num_dims: int = 8) -> None:
    """Plot distribution of latent space dimensions."""
    mu_np = mu_samples.numpy()
    num_dims = min(num_dims, mu_np.shape[1])
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(num_dims):
        ax = axes[i]
        ax.hist(mu_np[:, i], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f'Latent Dim {i}', fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(num_dims, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Latent Space Distribution (First 8 Dimensions)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved latent distribution plot to {output_path}")
    plt.close()


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, meta = build_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    print("Number of batches: ", len(train_loader))

    encoder = SMILESEncoder(
        vocab_size=len(meta["vocab"]),
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        latent_size=args.latent_size,
        max_len=meta["max_len"],
        dropout=args.dropout,
    ).to(device)

    head = SimpleDecoderHead(
        latent_size=args.latent_size,
        max_len=meta["max_len"],
        vocab_size=len(meta["vocab"]),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )

    def lr_lambda(step: int):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        return max(0.0, (args.max_steps - step) / max(1, args.max_steps - args.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    ce_loss = nn.CrossEntropyLoss(ignore_index=meta["pad_id"])

    global_step = 0
    best_val = float("inf")
    
    # Track losses for plotting
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        encoder.train()
        head.train()
        epoch_train_loss = 0.0
        epoch_train_batches = 0
        
        for batch in train_loader:
            global_step += 1
            tokens = prepare_batch(batch, device)

            mu, _ = encoder(tokens)
            logits = head(mu)

            loss = ce_loss(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), args.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_train_loss += loss.item()
            epoch_train_batches += 1

            if global_step % args.log_every == 0:
                print(
                    f"[train] epoch={epoch} step={global_step} "
                    f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if global_step >= args.max_steps:
                break

        # Average training loss for epoch
        avg_train_loss = epoch_train_loss / max(epoch_train_batches, 1)
        train_losses.append(avg_train_loss)

        # Validation
        encoder.eval()
        head.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                tokens = prepare_batch(batch, device)
                mu, _ = encoder(tokens)
                logits = head(mu)
                loss = ce_loss(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))
                val_loss += loss.item()
                val_batches += 1
        val_loss = val_loss / max(1, val_batches)
        val_losses.append(val_loss)
        print(f"[val] epoch={epoch} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "best_encoder.pt"),
                encoder,
                meta,
                step=global_step,
                val_loss=val_loss,
            )

        if global_step >= args.max_steps:
            break
    
    # Plot training curves
    plot_path = os.path.join(args.checkpoint_dir, "training_losses.png")
    plot_losses(train_losses, val_losses, plot_path)
    
    # Final evaluation metrics
    print("\n[INFO] Computing final evaluation metrics...")
    metrics = compute_metrics(encoder, head, val_loader, device, meta, num_samples=args.eval_samples)
    print_metrics(metrics)
    
    # Save metrics to JSON
    metrics_path = os.path.join(args.checkpoint_dir, "evaluation_metrics.json")
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = {
        "loss": metrics["loss"],
        "perplexity": metrics["perplexity"],
        "token_accuracy": metrics["token_accuracy"],
        "sequence_accuracy": metrics["sequence_accuracy"],
        "latent_stats": {
            "mu_mean": metrics["latent_stats"]["mu_mean"],
            "mu_std": metrics["latent_stats"]["mu_std"],
            "var_mean": metrics["latent_stats"]["var_mean"],
            "mean_pairwise_distance": metrics["latent_stats"]["mean_pairwise_distance"],
        },
        "num_samples_evaluated": metrics["num_samples_evaluated"],
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"[INFO] Saved evaluation metrics to {metrics_path}")
    
    # Plot latent space distribution
    print("\n[INFO] Collecting latent space samples for visualization...")
    encoder.eval()
    mu_samples = []
    with torch.no_grad():
        sample_count = 0
        for batch in val_loader:
            if sample_count >= 1000:
                break
            tokens = prepare_batch(batch, device)
            mu, _ = encoder(tokens)
            mu_samples.append(mu.cpu())
            sample_count += mu.shape[0]
    
    if mu_samples:
        all_mu = torch.cat(mu_samples, dim=0)
        latent_plot_path = os.path.join(args.checkpoint_dir, "latent_distribution.png")
        plot_latent_distribution(all_mu, latent_plot_path)
    
    print(f"\n[INFO] Pretraining complete! Checkpoints and plots saved to {args.checkpoint_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encoder-only pretraining on ChEMBL tokenized SMILES.")
    parser.add_argument("--data_dir", type=str, default="data/chembl/tokenized", help="Directory with train.pt / val.pt")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parser.add_argument("--checkpoint_dir", type=str, default=f"checkpoints/encoder_pretrain_{timestamp}", help="Where to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=5000, help="Stop after this many steps (across epochs)")
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_samples", type=int, default=1000, help="Number of samples to use for final evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
