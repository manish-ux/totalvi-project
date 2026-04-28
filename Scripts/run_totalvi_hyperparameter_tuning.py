import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import scvi
import scanpy as sc

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Output folders
# ---------------------------------------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/tuning_models", exist_ok=True)
os.makedirs("outputs/tuning_figures", exist_ok=True)

print("scvi-tools version:", scvi.__version__)

# ---------------------------------------------------------
# 1. Load official CITE-seq dataset
# ---------------------------------------------------------
adata_full = scvi.data.pbmcs_10x_cite_seq(
    save_path="data",
    protein_join="inner"
)

print("\nLoaded dataset:")
print(adata_full)
print("RNA shape:", adata_full.X.shape)
print("Protein shape:", adata_full.obsm["protein_expression"].shape)

# ---------------------------------------------------------
# 2. Use subset for laptop-safe tuning
# ---------------------------------------------------------
# Increase this later if your computer handles it.
adata_full = adata_full[:2000].copy()

print("\nSubset used for tuning:")
print(adata_full)

# ---------------------------------------------------------
# 3. Hyperparameter configurations
# ---------------------------------------------------------
param_grid = [
    {
        "trial": "trial_01_baseline",
        "n_latent": 10,
        "n_layers": 1,
        "n_hidden": 128,
        "dropout_rate": 0.1,
        "lr": 1e-3,
        "batch_size": 128,
        "max_epochs": 20,
    },
    {
        "trial": "trial_02_more_layers",
        "n_latent": 10,
        "n_layers": 2,
        "n_hidden": 128,
        "dropout_rate": 0.1,
        "lr": 1e-3,
        "batch_size": 128,
        "max_epochs": 20,
    },
    {
        "trial": "trial_03_more_latent",
        "n_latent": 20,
        "n_layers": 1,
        "n_hidden": 128,
        "dropout_rate": 0.1,
        "lr": 1e-3,
        "batch_size": 128,
        "max_epochs": 20,
    },
    {
        "trial": "trial_04_latent_layers",
        "n_latent": 20,
        "n_layers": 2,
        "n_hidden": 128,
        "dropout_rate": 0.1,
        "lr": 1e-3,
        "batch_size": 128,
        "max_epochs": 20,
    },
    {
        "trial": "trial_05_wider_network",
        "n_latent": 20,
        "n_layers": 2,
        "n_hidden": 256,
        "dropout_rate": 0.1,
        "lr": 1e-3,
        "batch_size": 128,
        "max_epochs": 20,
    },
    {
        "trial": "trial_06_more_dropout",
        "n_latent": 20,
        "n_layers": 2,
        "n_hidden": 256,
        "dropout_rate": 0.2,
        "lr": 1e-3,
        "batch_size": 128,
        "max_epochs": 20,
    },
    {
        "trial": "trial_07_larger_latent",
        "n_latent": 30,
        "n_layers": 2,
        "n_hidden": 256,
        "dropout_rate": 0.1,
        "lr": 1e-3,
        "batch_size": 128,
        "max_epochs": 20,
    },
    {
        "trial": "trial_08_lower_lr",
        "n_latent": 20,
        "n_layers": 2,
        "n_hidden": 256,
        "dropout_rate": 0.1,
        "lr": 5e-4,
        "batch_size": 128,
        "max_epochs": 20,
    },
]

results = []

# ---------------------------------------------------------
# 4. Run tuning loop
# ---------------------------------------------------------
for params in param_grid:
    trial_name = params["trial"]

    print("\n" + "=" * 70)
    print(f"Running {trial_name}")
    print(params)
    print("=" * 70)

    # Important: copy original AnnData for each trial
    adata = adata_full.copy()

    scvi.model.TOTALVI.setup_anndata(
        adata,
        protein_expression_obsm_key="protein_expression",
    )

    start_time = time.time()

    model = scvi.model.TOTALVI(
    adata,
    n_latent=params["n_latent"],
    n_layers_encoder=params["n_layers"],
    n_layers_decoder=params["n_layers"],
    n_hidden=params["n_hidden"],
)

    model.train(
        max_epochs=params["max_epochs"],
        lr=params["lr"],
        batch_size=params["batch_size"],
        accelerator="cpu",
        train_size=0.9,
        validation_size=0.1,
        early_stopping=True,
        early_stopping_patience=10,
    )

    runtime = time.time() - start_time

    # -----------------------------------------------------
    # Extract latent representation
    # -----------------------------------------------------
    latent = model.get_latent_representation()
    adata.obsm["X_totalVI"] = latent

    # -----------------------------------------------------
    # Try to collect final training/validation values
    # -----------------------------------------------------
    final_train_elbo = np.nan
    final_val_elbo = np.nan

    try:
        history = model.history

        if "elbo_train" in history:
            final_train_elbo = float(history["elbo_train"].iloc[-1])

        if "elbo_validation" in history:
            final_val_elbo = float(history["elbo_validation"].iloc[-1])

    except Exception as e:
        print("Could not extract ELBO history:", e)

    # -----------------------------------------------------
    # Save latent representation
    # -----------------------------------------------------
    latent_df = pd.DataFrame(latent)
    latent_path = f"outputs/{trial_name}_latent_representation.csv"
    latent_df.to_csv(latent_path, index=False)

    # -----------------------------------------------------
    # UMAP figure
    # -----------------------------------------------------
    try:
        sc.pp.neighbors(adata, use_rep="X_totalVI")
        sc.tl.umap(adata)

        sc.pl.umap(adata, show=False)
        plt.title(trial_name)
        plt.savefig(
            f"outputs/tuning_figures/{trial_name}_umap.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
    except Exception as e:
        print("UMAP failed for", trial_name, ":", e)

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------
    model.save(f"outputs/tuning_models/{trial_name}", overwrite=True)

    # -----------------------------------------------------
    # Store result row
    # -----------------------------------------------------
    results.append(
        {
            "trial": trial_name,
            "n_latent": params["n_latent"],
            "n_layers": params["n_layers"],
            "n_hidden": params["n_hidden"],
            "dropout_rate": params["dropout_rate"],
            "lr": params["lr"],
            "batch_size": params["batch_size"],
            "max_epochs": params["max_epochs"],
            "runtime_seconds": runtime,
            "final_train_elbo": final_train_elbo,
            "final_validation_elbo": final_val_elbo,
            "latent_file": latent_path,
        }
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv("outputs/totalvi_hyperparameter_tuning_results.csv", index=False)

# ---------------------------------------------------------
# 5. Save final comparison table
# ---------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("outputs/totalvi_hyperparameter_tuning_results.csv", index=False)

print("\nHyperparameter tuning finished.")
print(results_df)

# ---------------------------------------------------------
# 6. Choose best model
# ---------------------------------------------------------
# If validation ELBO is available, choose the lowest validation ELBO.
# If not available, choose manually using UMAP + runtime + stable training.
valid_results = results_df.dropna(subset=["final_validation_elbo"])

if len(valid_results) > 0:
    best_row = valid_results.sort_values("final_validation_elbo", ascending=True).iloc[0]
else:
    best_row = results_df.iloc[0]

best_row.to_frame().T.to_csv("outputs/best_totalvi_hyperparameters.csv", index=False)

print("\nBest hyperparameter setting:")
print(best_row)
print("\nSaved:")
print("outputs/totalvi_hyperparameter_tuning_results.csv")
print("outputs/best_totalvi_hyperparameters.csv")
print("outputs/tuning_figures/")