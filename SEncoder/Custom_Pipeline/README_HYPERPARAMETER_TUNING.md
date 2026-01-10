# Hyperparameter Tuning for Encoder Pretraining

This guide explains how to use Bayesian optimization to find optimal hyperparameters for encoder pretraining.

## Overview

Given that full training with 7703 batches and 10 epochs takes hours, using Bayesian optimization (via Optuna) can help you:
1. **Find better hyperparameters** faster than manual search
2. **Prune bad trials early** to save time
3. **Use a subset of data** during search for faster iterations
4. **Resume optimization** from saved checkpoints

## Installation

First, install Optuna:

```bash
pip install optuna
# Optional: for visualization plots
pip install plotly kaleido
```

## Quick Start

### Basic Usage (Recommended for First Run)

```bash
python tune_hyperparameters.py \
    --data_dir data/chembl/tokenized \
    --n_trials 50 \
    --search_epochs 2 \
    --data_subset_factor 0.2 \
    --pruning
```

This will:
- Run 50 trials
- Use 2 epochs per trial (quick evaluation)
- Use 20% of training data (faster iterations)
- Enable pruning to stop bad trials early

### Full Search (More Thorough)

```bash
python tune_hyperparameters.py \
    --data_dir data/chembl/tokenized \
    --n_trials 100 \
    --search_epochs 3 \
    --data_subset_factor 0.5 \
    --pruning \
    --patience 2 \
    --storage
```

## Parameters Explained

### Essential Parameters

- `--data_dir`: Directory containing `train.pt` and `val.pt` files
- `--n_trials`: Number of hyperparameter combinations to try (50-100 recommended)
- `--search_epochs`: Epochs per trial (2-3 recommended for search, then full training on best)
- `--data_subset_factor`: Fraction of training data to use (0.2-0.5 for search, 1.0 for final training)

### Optimization Parameters

- `--pruning`: Enable early stopping for unpromising trials (recommended)
- `--patience`: Epochs without improvement before early stopping (default: 2)
- `--search_max_steps`: Max training steps per trial (overrides epochs if set)
- `--storage`: Save study to SQLite database (allows resuming)
- `--study_name`: Name for the study (useful for resuming)

## Hyperparameters Being Optimized

The script optimizes these key hyperparameters:

| Parameter | Search Range | Impact |
|-----------|--------------|--------|
| `lr` | 1e-5 to 1e-3 (log) | **Critical** - Learning rate |
| `dropout` | 0.0 to 0.3 | Prevents overfitting |
| `weight_decay` | 1e-5 to 1e-3 (log) | Regularization |
| `n_layers` | 4 to 12 | Model capacity |
| `latent_size` | [256, 512, 768, 1024] | Latent representation size |
| `embedding_dim` | [256, 512, 768, 1024] | Token embedding size |
| `batch_size` | [16, 32, 64, 128] | Training stability & speed |
| `warmup_steps` | 1000 to 8000 | Learning rate warmup |
| `grad_clip` | 0.1 to 2.0 | Gradient clipping |

## Workflow

### Step 1: Run Hyperparameter Search

```bash
python tune_hyperparameters.py \
    --data_dir data/chembl/tokenized \
    --n_trials 50 \
    --search_epochs 2 \
    --data_subset_factor 0.2 \
    --pruning \
    --output_dir checkpoints/hyperparameter_search
```

### Step 2: Review Results

The script will save:
- `best_hyperparameters.json`: Best configuration found
- `param_importance.html`: Which hyperparameters matter most
- `optimization_history.html`: Progress over trials
- `optuna.db`: Database for resuming (if `--storage` used)

### Step 3: Train with Best Hyperparameters

Use the recommended command printed at the end, or manually:

```bash
python pretrain_encoder.py \
    --data_dir data/chembl/tokenized \
    --batch_size 64 \
    --n_layers 8 \
    --embedding_dim 512 \
    --latent_size 512 \
    --dropout 0.1 \
    --lr 5e-5 \
    --weight_decay 1e-4 \
    --warmup_steps 4000 \
    --grad_clip 1.0 \
    --epochs 10
```

## Time Estimates

Assuming ~7703 batches per epoch:

- **1 trial with 2 epochs (20% data)**: ~30-60 minutes
- **50 trials with pruning**: ~20-40 hours (many trials pruned early)
- **100 trials with pruning**: ~40-80 hours

With pruning enabled, bad trials are stopped early, significantly reducing total time.

## Tips

1. **Start small**: Run 20-30 trials first to see if optimization is working
2. **Check visualizations**: Look at `param_importance.html` to see what matters
3. **Resume if needed**: Use `--storage` and `--study_name` to resume interrupted searches
4. **Balance speed vs quality**: Lower `data_subset_factor` = faster but less accurate rankings
5. **Pruning is key**: Always use `--pruning` to save time on bad configurations

## Resuming a Study

If you used `--storage`, you can resume:

```bash
python tune_hyperparameters.py \
    --data_dir data/chembl/tokenized \
    --study_name pretrain_optuna_tokenized \
    --n_trials 100 \
    --storage
```

## Example Output

```
======================================================================
Starting hyperparameter optimization
======================================================================
Study name: pretrain_optuna_chembl_tokenized
Number of trials: 50
Search epochs per trial: 2
Pruning: True
Data subset factor: 0.2
======================================================================

[I 2025-01-10 16:00:00,000] Trial 0 finished with value: 2.3456 and parameters: {...}
[I 2025-01-10 16:30:00,000] Trial 1 pruned.
...

======================================================================
OPTIMIZATION COMPLETE
======================================================================
Number of finished trials: 50
Number of pruned trials: 15
Number of complete trials: 35

Best trial:
  Value (val_loss): 1.8234
  Params:
    lr: 0.000123
    dropout: 0.15
    weight_decay: 0.0001
    n_layers: 8
    latent_size: 512
    embedding_dim: 512
    batch_size: 64
    warmup_steps: 4000
    grad_clip: 1.0
```

## Next Steps

After finding good hyperparameters:
1. Train full model with best config (10 epochs on full data)
2. Evaluate on downstream tasks
3. Consider fine-tuning architecture if needed
4. Compare with baseline performance
