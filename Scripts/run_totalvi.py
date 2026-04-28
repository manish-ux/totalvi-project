import os
import pandas as pd
import matplotlib.pyplot as plt

import scvi
import scanpy as sc

# Create output folders
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("scvi-tools version:", scvi.__version__)

# ---------------------------------------------------------
# 1. Load official CITE-seq dataset
# ---------------------------------------------------------
# This dataset contains paired RNA + protein measurements.
# It downloads automatically the first time.
adata = scvi.data.pbmcs_10x_cite_seq(
    save_path="data",
    protein_join="inner"
)

print("\nLoaded dataset:")
print(adata)
print("RNA matrix shape:", adata.X.shape)
print("Protein matrix shape:", adata.obsm["protein_expression"].shape)

# ---------------------------------------------------------
# 2. Use a smaller subset first for laptop safety
# ---------------------------------------------------------
# Increase this later if your laptop handles it.
adata = adata[:2000].copy()

print("\nSubset dataset:")
print(adata)

# ---------------------------------------------------------
# 3. Register AnnData object for totalVI
# ---------------------------------------------------------
scvi.model.TOTALVI.setup_anndata(
    adata,
    protein_expression_obsm_key="protein_expression"
)

# ---------------------------------------------------------
# 4. Build and train totalVI model
# ---------------------------------------------------------
model = scvi.model.TOTALVI(adata)

model.train(
    max_epochs=20,
    accelerator="cpu"
)

# ---------------------------------------------------------
# 5. Extract latent representation
# ---------------------------------------------------------
latent = model.get_latent_representation()
adata.obsm["X_totalVI"] = latent

print("\nLatent representation shape:", latent.shape)

latent_df = pd.DataFrame(latent)
latent_df.to_csv("outputs/totalvi_latent_representation.csv", index=False)

# ---------------------------------------------------------
# 6. Create UMAP visualization
# ---------------------------------------------------------
sc.pp.neighbors(adata, use_rep="X_totalVI")
sc.tl.umap(adata)

sc.pl.umap(adata, show=False)
plt.savefig("outputs/totalvi_umap.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 7. Save trained model
# ---------------------------------------------------------
model.save("outputs/totalvi_model", overwrite=True)

print("\nFinished successfully.")
print("Saved files:")
print("outputs/totalvi_latent_representation.csv")
print("outputs/totalvi_umap.png")
print("outputs/totalvi_model/")