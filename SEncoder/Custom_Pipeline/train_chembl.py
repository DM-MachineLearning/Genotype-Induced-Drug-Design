"""
Lightweight training loop to pretrain the Custom Pipeline Transformer VAE on ChEMBL.

Prereqs:
    1) Run fetch_chembl.py to download SMILES.
    2) Run preprocess_chembl.py to produce train.pt / val.pt.

Usage:
    python train_chembl.py --data_dir data/chembl/tokenized --epochs 1 --max_steps 2000
"""

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from smiles_encoder import SMILESEncoder
from smiles_decoder import SMILESDecoder

import matplotlib.pyplot as plt


@dataclass
class TrainBatch:
    tokens: torch.Tensor
    decoder_input: torch.Tensor
    decoder_target: torch.Tensor


def compute_kl(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    # KL divergence: 0.5 * sum(mu^2 + var - log(var) - 1)
    return 0.5 * (torch.sum(mu ** 2) + torch.sum(var) - torch.sum(torch.log(var + 1e-8)) - mu.numel())


def load_split(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return torch.load(path, map_location="cpu")


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, Dict]:
    train_obj = load_split(os.path.join(data_dir, "train.pt"))
    val_obj = load_split(os.path.join(data_dir, "val.pt"))
    meta = train_obj["meta"]

    train_ds = TensorDataset(train_obj["data"])
    val_ds = TensorDataset(val_obj["data"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, meta


def make_models(vocab_size: int, embedding_dim: int, latent_size: int, n_layers: int, max_len: int, dropout: float, device: torch.device):
    encoder = SMILESEncoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        latent_size=latent_size,
        max_len=max_len,
        dropout=dropout,
    ).to(device)

    decoder = SMILESDecoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        latent_size=latent_size,
        max_len=max_len,
        dropout=dropout,
    ).to(device)
    return encoder, decoder


def prepare_batch(batch: Tuple[torch.Tensor], device: torch.device) -> TrainBatch:
    tokens = batch[0].to(device)
    decoder_input = tokens[:, :-1]
    decoder_target = tokens[:, 1:]
    return TrainBatch(tokens=tokens, decoder_input=decoder_input, decoder_target=decoder_target)


def save_checkpoint(path: str, encoder: nn.Module, decoder: nn.Module, meta: Dict, step: int, val_loss: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "encoder_state": encoder.state_dict(),
            "decoder_state": decoder.state_dict(),
            "meta": meta,
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )
    print(f"[INFO] Saved checkpoint to {path} (step={step}, val_loss={val_loss:.4f})")


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, meta = build_dataloaders(args.data_dir, args.batch_size, args.num_workers)

    encoder, decoder = make_models(
        vocab_size=len(meta["vocab"]),
        embedding_dim=args.embedding_dim,
        latent_size=args.latent_size,
        n_layers=args.n_layers,
        max_len=meta["max_len"],
        dropout=args.dropout,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
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

    train_total_losses = []
    train_recon_losses = []
    train_kl_losses = []

    val_total_losses = []
    val_recon_losses = []
    val_kl_losses = []

    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()

        train_total = 0.0
        train_recon = 0.0
        train_kl = 0.0
        train_batches = 0

        for batch in train_loader:
            global_step += 1
            batch_t = prepare_batch(batch, device)

            mu, var = encoder(batch_t.tokens)
            latent = mu + torch.sqrt(var) * torch.randn_like(mu)

            logits = decoder(latent, batch_t.decoder_input, mode="forced")
            recon = ce_loss(logits.reshape(-1, logits.size(-1)), batch_t.decoder_target.reshape(-1))
            kl = compute_kl(mu, var)
            loss = recon + args.beta * kl

            train_total += loss.item()
            train_recon += recon.item()
            train_kl += kl.item()
            train_batches += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), args.grad_clip)
            optimizer.step()
            scheduler.step()

            if global_step % args.log_every == 0:
                print(
                    f"[train] epoch={epoch} step={global_step} "
                    f"loss={loss.item():.4f} recon={recon.item():.4f} kl={kl.item():.2f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if global_step >= args.max_steps:
                break
        
        train_total_losses.append(train_total / max(1, train_batches))
        train_recon_losses.append(train_recon / max(1, train_batches))
        train_kl_losses.append(train_kl / max(1, train_batches))

        # Validation
        encoder.eval()
        decoder.eval()

        val_total = 0.0
        val_recon = 0.0
        val_kl = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch_t = prepare_batch(batch, device)

                mu, var = encoder(batch_t.tokens)
                latent = mu + torch.sqrt(var) * torch.randn_like(mu)

                logits = decoder(latent, batch_t.decoder_input, mode="forced")
                recon = ce_loss(
                    logits.reshape(-1, logits.size(-1)),
                    batch_t.decoder_target.reshape(-1)
                )
                kl = compute_kl(mu, var)
                loss = recon + args.beta * kl

                val_total += loss.item()
                val_recon += recon.item()
                val_kl += kl.item()
                val_batches += 1

        val_total /= max(1, val_batches)
        val_recon /= max(1, val_batches)
        val_kl /= max(1, val_batches)

        print(f"[val] epoch={epoch} val_loss={val_total:.4f}")

        val_total_losses.append(val_total)
        val_recon_losses.append(val_recon)
        val_kl_losses.append(val_kl)

        val_loss = val_total

        # Checkpoint
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                os.path.join(args.checkpoint_dir, "best.pt"),
                encoder,
                decoder,
                meta,
                step=global_step,
                val_loss=val_loss,
            )

        if global_step >= args.max_steps:
            break

    epochs = range(1, len(train_total_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_total_losses, label="Train Total")
    plt.plot(epochs, val_total_losses, label="Val Total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.checkpoint_dir, "total_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_recon_losses, label="Train Recon")
    plt.plot(epochs, val_recon_losses, label="Val Recon")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.checkpoint_dir, "recon_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_kl_losses, label="Train KL")
    plt.plot(epochs, val_kl_losses, label="Val KL")
    plt.xlabel("Epoch")
    plt.ylabel("KL")
    plt.title("KL Divergence")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.checkpoint_dir, "kl_loss.png"))
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer VAE on ChEMBL subset.")
    parser.add_argument("--data_dir", type=str, default="data/chembl/tokenized", help="Directory with train.pt / val.pt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/chembl_sanity", help="Where to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=2000, help="Stop after this many steps (across epochs)")
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--beta", type=float, default=0.001, help="KL weight")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

