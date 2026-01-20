import optuna
import torch
import pickle
from torch.optim import Adam

from Genotype_Induced_Drug_Design.PVAE.CNN_VAE import CNNVAE
from Genotype_Induced_Drug_Design.PVAE.dataloader import return_dataloaders_supervised
from Genotype_Induced_Drug_Design.PVAE.cnn_vae_train_script import evaluate
from Genotype_Induced_Drug_Design.PVAE.cnn_vae_train_script import (
    BEST_LAMB, BEST_ALPHA, BEST_MASK, BEST_BLOCK, BEST_ZDIM
)

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

    # CNN training knobs
    lr_encoder = trial.suggest_float("lr_encoder", 1e-5, 3e-4, log=True)
    lr_decoder = trial.suggest_float("lr_decoder", 1e-5, 3e-4, log=True)
    lr_classifier = trial.suggest_float("lr_classifier", 1e-4, 1e-3, log=True)
    lr_latent = trial.suggest_float("lr_latent", 1e-5, 3e-4, log=True)

    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    freeze_depth = trial.suggest_categorical("freeze_depth", [0, 2, 4, 6])

    train_loader, val_loader, _ = return_dataloaders_supervised(
        dna_meth,
        gene_exp,
        labels,
        split_fractions=(0.8, 0.1)
    )

    model = CNNVAE(
        input_dim=INPUT_DIM,
        z_dim=BEST_ZDIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    # Freeze early CNN layers
    for i, layer in enumerate(model.encoder_conv):
        if i < freeze_depth:
            for p in layer.parameters():
                p.requires_grad = False

    optimizer = Adam(
        [
            {"params": model.encoder_conv.parameters(), "lr": lr_encoder},
            {"params": model.decoder_conv.parameters(), "lr": lr_decoder},
            {"params": model.classifier_mlp.parameters(), "lr": lr_classifier},
            {"params": model.mu_layer.parameters(), "lr": lr_latent},
            {"params": model.logvar_layer.parameters(), "lr": lr_latent},
        ],
        weight_decay=weight_decay
    )

    history, _, _ = model.trainer(
        train_loader=train_loader,
        optimizer=optimizer,
        num_epochs=60,
        device=DEVICE,
        lamb=BEST_LAMB,
        alpha=BEST_ALPHA,
        patience=8,
        verbose=False,
        apply_masking=True,
        mask_ratio=BEST_MASK,
        block_size=BEST_BLOCK
    )

    trial.set_user_attr("train_history", history)

    val_loss, _ = evaluate(
        model,
        val_loader,
        DEVICE,
        lamb=BEST_LAMB,
        alpha=BEST_ALPHA
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
            n_startup_trials=3,
            n_warmup_steps=1
        ),
        study_name="cnn_finetune",
        storage="sqlite:///cnn_finetune_optuna_1.db",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=25)

    print("\n Best CNN training setup")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
    print("Best Val Loss:", study.best_value)