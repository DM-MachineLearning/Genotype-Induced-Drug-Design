"""
Bayesian optimization for pretraining hyperparameters using Optuna.

This script uses Optuna to efficiently search the hyperparameter space.
To reduce training time, it:
  1. Uses early stopping/pruning for unpromising trials
  2. Optionally uses a subset of data for faster iterations
  3. Runs fewer epochs/steps during search, then full training on best config

Usage:
    python tune_hyperparameters.py --data_dir data/chembl/tokenized --n_trials 50 --pruning
"""

import argparse
import json
import os
import shutil
from typing import Dict, Optional

import optuna
import torch
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from pretrain_encoder import (
    build_dataloaders,
    prepare_batch,
    save_checkpoint,
    compute_metrics,
    print_metrics,
)

from smiles_encoder import SMILESEncoder


class SimpleDecoderHead(nn.Module):
    """Lightweight decoder head for reconstruction."""
    
    def __init__(self, latent_size: int, max_len: int, vocab_size: int, dropout: float = 0.0):
        super().__init__()
        self.max_len = max_len
        self.positional = nn.Embedding(max_len, latent_size)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(latent_size, vocab_size)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        batch_size, latent_size = mu.shape
        mu_expanded = mu.unsqueeze(1).expand(batch_size, self.max_len, latent_size)
        positions = torch.arange(self.max_len, device=mu.device).unsqueeze(0).expand(batch_size, -1)
        hidden = mu_expanded + self.positional(positions)
        hidden = self.dropout(hidden)
        logits = self.proj(hidden)
        return logits


def train_with_early_stopping(
    trial: optuna.Trial,
    train_loader,
    val_loader,
    meta: Dict,
    device: torch.device,
    max_epochs: int = 3,
    max_steps: Optional[int] = None,
    eval_every_n_epochs: int = 1,
    patience: int = 2,
) -> float:
    """
    Train model with early stopping for hyperparameter tuning.
    Returns best validation loss.
    """
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    n_layers = trial.suggest_int("n_layers", 4, 12)
    latent_size = trial.suggest_categorical("latent_size", [256, 512, 768, 1024])
    embedding_dim = trial.suggest_categorical("embedding_dim", [256, 512, 768, 1024])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    warmup_steps = trial.suggest_int("warmup_steps", 1000, 8000, step=1000)
    grad_clip = trial.suggest_float("grad_clip", 0.1, 2.0)

    # Rebuild dataloaders with new batch size
    train_obj = meta.get("train_obj")
    val_obj = meta.get("val_obj")
    data_dir = meta.get("data_dir")
    
    if train_obj is None or val_obj is None:
        from train_chembl import load_split
        train_obj = load_split(os.path.join(data_dir, "train.pt"))
        val_obj = load_split(os.path.join(data_dir, "val.pt"))
        meta["train_obj"] = train_obj
        meta["val_obj"] = val_obj
    
    from torch.utils.data import DataLoader, TensorDataset
    
    # Optionally use subset of data for faster tuning
    data_subset_factor = meta.get("data_subset_factor", 1.0)
    train_data = train_obj["data"]  # This is a tensor
    
    if data_subset_factor < 1.0:
        subset_size = int(train_data.shape[0] * data_subset_factor)
        train_data = train_data[:subset_size]
    
    train_ds = TensorDataset(train_data)
    val_ds = TensorDataset(val_obj["data"])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build model
    encoder = SMILESEncoder(
        vocab_size=len(meta["vocab"]),
        embedding_dim=embedding_dim,
        n_layers=n_layers,
        latent_size=latent_size,
        max_len=meta["max_len"],
        dropout=dropout,
    ).to(device)

    head = SimpleDecoderHead(
        latent_size=latent_size,
        max_len=meta["max_len"],
        vocab_size=len(meta["vocab"]),
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=lr,
        betas=(0.9, 0.98),
        weight_decay=weight_decay,
    )

    # Calculate max_steps if not provided
    if max_steps is None:
        max_steps = len(train_loader) * max_epochs
    
    def lr_lambda(step: int):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (max_steps - step) / max(1, max_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    ce_loss = nn.CrossEntropyLoss(ignore_index=meta["pad_id"])

    global_step = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    
    for epoch in range(max_epochs):
        encoder.train()
        head.train()
        
        for batch in train_loader:
            global_step += 1
            tokens = prepare_batch(batch, device)

            mu, _ = encoder(tokens)
            logits = head(mu)
            loss = ce_loss(logits.reshape(-1, logits.size(-1)), tokens.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(head.parameters()), grad_clip
                )
            optimizer.step()
            scheduler.step()

            if global_step >= max_steps:
                break

        # Evaluate every N epochs
        if (epoch + 1) % eval_every_n_epochs == 0:
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
            
            # Report intermediate value for pruning
            trial.report(val_loss, epoch)
            
            # Pruning based on intermediate results
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            # Track best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Early stopping
            if epochs_without_improvement >= patience:
                break

        if global_step >= max_steps:
            break

    return best_val_loss


def objective(trial: optuna.Trial, args: argparse.Namespace, meta: Dict) -> float:
    """Objective function for Optuna optimization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure data_dir is in meta
    meta["data_dir"] = args.data_dir
    
    # Create dummy loaders (will be recreated with different batch sizes in train_with_early_stopping)
    from torch.utils.data import DataLoader, TensorDataset
    from train_chembl import load_split
    
    # Load data once and store in meta for reuse across trials
    if "train_obj" not in meta or "val_obj" not in meta:
        train_obj = load_split(os.path.join(args.data_dir, "train.pt"))
        val_obj = load_split(os.path.join(args.data_dir, "val.pt"))
        meta["train_obj"] = train_obj
        meta["val_obj"] = val_obj
    
    train_ds = TensorDataset(meta["train_obj"]["data"])
    val_ds = TensorDataset(meta["val_obj"]["data"])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    return train_with_early_stopping(
        trial=trial,
        train_loader=train_loader,
        val_loader=val_loader,
        meta=meta,
        device=device,
        max_epochs=args.search_epochs,
        max_steps=args.search_max_steps,
        eval_every_n_epochs=1,
        patience=args.patience,
    )


def main(args: argparse.Namespace):
    # Load metadata
    from train_chembl import load_split
    train_obj = load_split(os.path.join(args.data_dir, "train.pt"))
    meta = train_obj["meta"].copy()
    
    # Create study
    study_name = args.study_name or f"pretrain_optuna_{os.path.basename(args.data_dir)}"
    storage = f"sqlite:///{args.output_dir}/optuna.db" if args.storage else None
    
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2) if args.pruning else None
    
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )
    
    print(f"\n{'='*70}")
    print(f"Starting hyperparameter optimization")
    print(f"{'='*70}")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Search epochs per trial: {args.search_epochs}")
    print(f"Pruning: {args.pruning}")
    print(f"Data subset factor: {args.data_subset_factor}")
    print(f"{'='*70}\n")
    
    # Store meta in a way accessible to objective
    meta_dict = meta.copy()
    meta_dict["data_subset_factor"] = args.data_subset_factor
    
    # Optimize
    study.optimize(
        lambda trial: objective(trial, args, meta_dict),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )
    
    # Print results
    print(f"\n{'='*70}")
    print("OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    print(f"\nBest trial:")
    trial = study.best_trial
    print(f"  Value (val_loss): {trial.value:.4f}")
    print(f"  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save best hyperparameters
    best_params_path = os.path.join(args.output_dir, "best_hyperparameters.json")
    with open(best_params_path, 'w') as f:
        json.dump({
            "best_value": trial.value,
            "params": trial.params,
            "number": trial.number,
        }, f, indent=2)
    print(f"\nSaved best hyperparameters to {best_params_path}")
    
    # Save study visualization (requires plotly)
    try:
        import optuna.visualization as vis
        
        # Importance plot
        try:
            fig = vis.plot_param_importances(study)
            importance_path = os.path.join(args.output_dir, "param_importance.html")
            fig.write_html(importance_path)
            print(f"Saved parameter importance plot to {importance_path}")
        except Exception as e:
            print(f"Could not generate importance plot: {e}")
        
        # Optimization history
        try:
            fig = vis.plot_optimization_history(study)
            history_path = os.path.join(args.output_dir, "optimization_history.html")
            fig.write_html(history_path)
            print(f"Saved optimization history to {history_path}")
        except Exception as e:
            print(f"Could not generate history plot: {e}")
        
    except ImportError:
        print("Install plotly and kaleido to generate visualizations: pip install plotly kaleido")
    
    print(f"\n{'='*70}")
    print("RECOMMENDATION:")
    print(f"{'='*70}")
    print("Now train with best hyperparameters using:")
    print(f"  python pretrain_encoder.py --data_dir {args.data_dir} \\")
    for key, value in trial.params.items():
        if key == "batch_size":
            print(f"    --{key} {value} \\")
        elif key == "n_layers":
            print(f"    --{key} {value} \\")
        elif key == "embedding_dim":
            print(f"    --{key} {value} \\")
        elif key == "latent_size":
            print(f"    --{key} {value} \\")
        elif key == "dropout":
            print(f"    --{key} {value} \\")
        elif key == "lr":
            print(f"    --{key} {value} \\")
        elif key == "weight_decay":
            print(f"    --{key} {value} \\")
        elif key == "warmup_steps":
            print(f"    --{key} {value} \\")
        elif key == "grad_clip":
            print(f"    --{key} {value} \\")
    print("    --epochs 10")
    print(f"{'='*70}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimization for encoder pretraining"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/chembl/tokenized",
        help="Directory with train.pt / val.pt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/hyperparameter_search",
        help="Directory to save optimization results",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of optimization trials",
    )
    parser.add_argument(
        "--search_epochs",
        type=int,
        default=2,
        help="Number of epochs per trial during search (fewer = faster)",
    )
    parser.add_argument(
        "--search_max_steps",
        type=int,
        default=None,
        help="Max steps per trial (overrides search_epochs if set)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--pruning",
        action="store_true",
        help="Enable Optuna pruning for bad trials",
    )
    parser.add_argument(
        "--data_subset_factor",
        type=float,
        default=0.2,
        help="Use subset of training data for faster search (0.2 = 20%%)",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Name for Optuna study (for resuming)",
    )
    parser.add_argument(
        "--storage",
        action="store_true",
        help="Save study to SQLite database for resuming",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
