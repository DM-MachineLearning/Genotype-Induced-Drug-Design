# python -m Genotype_Induced_Drug_Design.PVAE.mse_optuna_finetune
# dashboard: optuna-dashboard sqlite:///mse_cnn_vae_masking_optuna.db

import optuna
import torch
import pickle
from torch import optim

from Genotype_Induced_Drug_Design.PVAE.MSE_CNN_VAE import MSECNNVAE
from Genotype_Induced_Drug_Design.PVAE.dataloader import return_dataloaders_supervised
from Genotype_Induced_Drug_Design.PVAE.mse_cnn_vae_train_script import evaluate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 15703
DATA_PATH = "/home/dmlab/Devendra/data/preprocessed_datasets/"


with open(DATA_PATH + "methylation_tensor_tcga.pkl", "rb") as f:
    dna_meth = pickle.load(f).float()

with open(DATA_PATH + "gene_expression_tensor_tcga.pkl", "rb") as f:
    gene_exp = pickle.load(f).float()

with open(DATA_PATH + "cancer_tags_tensor_tcga.pkl", "rb") as f:
    labels = pickle.load(f)

if labels.dim() > 1:
    labels = torch.argmax(labels, dim=1)

labels = labels.long()
NUM_CLASSES = len(torch.unique(labels))


def objective(trial):

    # Hyperparameters to tune (ONLY these)
    lamb = trial.suggest_float("lamb", 0.1, 2.0)
    alpha = trial.suggest_float("alpha", 5.0, 30.0)
    mask_ratio = trial.suggest_float("mask_ratio", 0.15, 0.4)
    block_size = trial.suggest_int("block_size", 50, 400, step=50)
    z_dim = trial.suggest_categorical("z_dim", [128])

    # Fixed choices (intentionally)
    lr = 1e-4
    weight_decay = 1e-4

    train_loader, val_loader, _ = return_dataloaders_supervised(
        dna_meth,
        gene_exp,
        labels,
        split_fractions=(0.8, 0.1),
    )

    model = MSECNNVAE(
        input_dim=INPUT_DIM,
        z_dim=z_dim,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # Train
    history, _, _ = model.trainer(
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=60,
        device=DEVICE,
        lamb=lamb,
        alpha=alpha,
        patience=8,
        verbose=False,
        apply_masking=True,
        mask_ratio=mask_ratio,
        block_size=block_size
    )

    # Save training curves for later visualization
    trial.set_user_attr("train_history", history)

    # Validation loss (true objective)
    val_loss, val_acc = evaluate(
        model,
        val_loader,
        DEVICE,
        lamb=lamb,
        alpha=alpha
    )

    final_train_loss = history[-1][0]
    trial.report(final_train_loss, step=1)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return val_loss


if __name__ == "__main__":

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=1
        ),
        study_name="mse_cnn_vae_masking_tradeoff",
        storage="sqlite:///mse_cnn_vae_masking_on_optuna.db",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=40)

    print("\n Best Configuration")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
    print("Best Val Loss:", study.best_value)
