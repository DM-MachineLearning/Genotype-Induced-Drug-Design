import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

HISTORY_PKL = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/results_/cnn_vae/cnn_vae_supervised_history_noisy_128_mask_off_mu_mse.pkl"
SAVE_DIR = "/home/dmlab/Devendra/Genotype_Induced_Drug_Design/PVAE/results_/cnn_vae/plots"  

# Column layout in your trainer/test_history:
# [total, rec_m, rec_g, kl, cls, acc]
METRICS = [
    ("Total Loss", 0),
    ("Recon Loss (DNA-meth)", 1),
    ("Recon Loss (Gene-exp)", 2),
    ("KL Loss", 3),
    ("Classifier Loss", 4),
]

def load_histories(pkl_path: str):
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    train_history = obj.get("train_history", obj.get("history", None))
    test_history = obj.get("test_history", None)

    if train_history is None:
        raise ValueError("Pickle does not contain 'train_history' (or fallback 'history').")

    train = np.asarray(train_history, dtype=np.float64)
    test = None if test_history is None else np.asarray(test_history, dtype=np.float64)

    if train.ndim != 2 or train.shape[1] < 5:
        raise ValueError(f"Unexpected train_history shape: {train.shape} (expected [epochs, >=5]).")

    if test is not None and (test.ndim != 2 or test.shape[1] < 5):
        raise ValueError(f"Unexpected test_history shape: {test.shape} (expected [epochs, >=5]).")

    return train, test

def plot_train_test_curve(train, test, title, col_idx, save_dir=None):
    e_train = np.arange(1, train.shape[0] + 1)
    y_train = train[:, col_idx]

    plt.figure()
    plt.plot(e_train, y_train, label="Train")

    if test is not None and len(test) > 0:
        e_test = np.arange(1, test.shape[0] + 1)
        y_test = test[:, col_idx]
        plt.plot(e_test, y_test, label="Test")

    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(f"{title} vs Epochs")
    plt.legend()
    plt.grid(True)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = title.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(save_dir, f"{fname}_curve.png"), dpi=200, bbox_inches="tight")

    plt.show()

def main():
    train, test = load_histories(HISTORY_PKL)

    for name, idx in METRICS:
        plot_train_test_curve(train, test, name, idx, save_dir=SAVE_DIR)

if __name__ == "__main__":
    main()
