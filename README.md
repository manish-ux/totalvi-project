\# totalVI Single-Cell Multi-Omics Project



\## Project Overview



This repository contains my Advanced Data Mining final project implementation based on \*\*totalVI\*\*, a variational autoencoder-based model for joint analysis of single-cell RNA and protein data.



The project uses the `scvi-tools` Python package to train a totalVI model, generate a joint latent representation, perform hyperparameter tuning, and save output files such as UMAP visualizations, latent feature matrices, and tuning result tables.



This project is connected to the topic:



\*\*Autoencoder-Based Feature Selection / Representation Learning for Biological Data\*\*



\---



\## Important Note About `scvi-tools`



The original `scvi-tools/` source-code folder is \*\*not included\*\* in this repository.



This is intentional.



The project does \*\*not\*\* require uploading or modifying the full `scvi-tools` GitHub source code. Instead, `scvi-tools` is installed as a Python package using `pip` through the `requirements.txt` file.



This keeps the repository clean, smaller, and easier to run.



To install all required packages, use:



```bash

pip install -r requirements.txt



**Research Paper**



**Paper Title**: Joint probabilistic modeling of single-cell multi-omic data with totalVI

**Authors**: Gayoso et al.

**Year**: 2021

**Paper Link**: https://www.nature.com/articles/s41592-020-01050-x

**Code Package**: https://github.com/scverse/scvi-tools

**Documentation**: https://docs.scvi-tools.org/



**Method Summary**



totalVI is a deep probabilistic model designed for CITE-seq data, where each cell contains both:



* RNA gene-expression measurements
* Protein abundance measurements



The model uses a variational autoencoder structure:



* The **encoder** maps high-dimensional RNA and protein measurements into a lower-dimensional latent space.
* The **latent representation** captures shared biological variation across both modalities.
* The **decoder** reconstructs the original RNA and protein measurements from the latent representation.



The learned latent representation can then be used for visualization, clustering, downstream biological analysis, and model comparison.



**Feature Selection Interpretation**



totalVI does not perform hard feature selection.



This means it does not directly return a fixed subset of selected genes or proteins.



Instead, totalVI learns a compressed latent representation from high-dimensional RNA and protein data. Therefore, in this project, totalVI is treated as a **soft representation-learning method** rather than a strict hard feature-selection method.



This distinction is important because autoencoder-based methods can either:



1. directly select original input features, or
2. learn hidden latent representations and indirectly support feature interpretation.



totalVI belongs to the second category.



**Repository Structure**



ADM\_Final\_Project/

│

├── Data/

│   └── .gitkeep

│

├── Outputs/

│   └── .gitkeep

│

├── Scripts/

│   ├── run\_totalvi.py

│   ├── run\_totalvi\_hyperparameter\_tuning.py

│   └── run\_totalvi\_best\_model.py

│

├── requirements.txt

├── README.md

└── .gitignore



**Folder Explanation**

**Data/**



This folder is included as a placeholder.



The dataset is downloaded automatically when the script is run for the first time. Large downloaded dataset files are not stored in GitHub to keep the repository lightweight.



**Outputs/**



This folder is included as a placeholder.



When the scripts are run, output files such as UMAP figures, latent representations, tuning results, and model outputs are saved here.



**Scripts/**



This folder contains the project Python scripts.



**Scripts**

**1. Scripts/run\_totalvi.py**



Runs the basic totalVI model.



This script:



* loads the CITE-seq dataset,
* trains a totalVI model,
* extracts the latent representation,
* generates a UMAP figure,
* saves the trained model and output files.



Run with:

python Scripts/run\_totalvi.py





**2. Scripts/run\_totalvi\_hyperparameter\_tuning.py**



Runs hyperparameter tuning for totalVI.



This script compares different model settings such as:



* latent dimension,
* number of encoder/decoder layers,
* number of hidden units,
* learning rate,
* batch size,
* number of epochs.



It saves the tuning results in the Outputs/ folder.



Run with:

python Scripts/run\_totalvi\_hyperparameter\_tuning.py



**3. Scripts/run\_totalvi\_best\_model.py**



Runs the final best model after hyperparameter tuning.



This script should use the best hyperparameter setting selected from the tuning results.



Run with:

python Scripts/run\_totalvi\_best\_model.py



**Installation and Running Instructions**



**Step 1: Clone the Repository**



git clone <your-github-repository-link>

cd <repository-folder-name>



**Step 2: Create a Virtual Environment**



For Windows PowerShell:



python -m venv venv\_totalvi

venv\_totalvi\\Scripts\\activate



For Mac/Linux:



python -m venv venv\_totalvi

source venv\_totalvi/bin/activate



**Step 3: Install Required Packages**



python -m pip install --upgrade pip setuptools wheel

pip install -r requirements.txt



**Step 4: Run the Basic totalVI Script**



python Scripts/run\_totalvi.py



This will automatically create/download the required data into the Data/ folder and save outputs into the Outputs/ folder.



**Step 5: Run Hyperparameter Tuning**



python Scripts/run\_totalvi\_hyperparameter\_tuning.py



This generates:



Outputs/totalvi\_hyperparameter\_tuning\_results.csv

Outputs/best\_totalvi\_hyperparameters.csv

Outputs/tuning\_figures/



**Step 6: Run the Best Final Model**



python Scripts/run\_totalvi\_best\_model.py



This generates final model outputs inside the Outputs/ folder.



**Expected Outputs**



After running the scripts, the Outputs/ folder may contain:



totalvi\_latent\_representation.csv

totalvi\_umap.png

totalvi\_hyperparameter\_tuning\_results.csv

best\_totalvi\_hyperparameters.csv

best\_model/

tuning\_figures/



Important output files:



* totalvi\_latent\_representation.csv

&#x20;	Low-dimensional latent representation learned by totalVI.

* totalvi\_umap.png

&#x09;UMAP visualization created from the totalVI latent space.

* totalvi\_hyperparameter\_tuning\_results.csv

&#x09;Table comparing different hyperparameter settings.

* best\_totalvi\_hyperparameters.csv

&#x09;Best selected hyperparameter configuration.



**Hyperparameter Tuning**



The following hyperparameters were tuned:



n\_latent

n\_layers\_encoder

n\_layers\_decoder

n\_hidden

learning\_rate

batch\_size

max\_epochs



The best model was selected based on:



validation performance

training stability

runtime

quality of the latent representation

UMAP visualization quality



Hyperparameter tuning is important because totalVI is a deep generative model. Poor hyperparameters can cause underfitting, overfitting, unstable training, or weak latent representations.



**Data Note**



The dataset is not manually uploaded to this repository.



The scripts use the dataset-loading functionality from scvi-tools, which downloads the required CITE-seq data automatically when the script is run.



The Data/ folder is included only as a placeholder so that the project structure is clear.



**Requirements**



The required Python packages are listed in:



requirements.txt



The main dependency is:



scvi-tools



which provides the totalVI implementation.



**Important Reproducibility Note**



For best results, run this project inside a fresh virtual environment.



Recommended Python version:



Python 3.10, 3.11, or 3.12



**Final Interpretation**



This project demonstrates how totalVI can be used for autoencoder-based biological representation learning. The model learns a low-dimensional joint representation from high-dimensional RNA and protein data and supports downstream visualization and analysis.



Although totalVI does not perform strict hard feature selection, it is useful for understanding biological structure in multi-omics data. It represents an important modern example of variational autoencoder-based analysis in computational biology.





\# 2. Final `requirements.txt`



Create or replace your requirements file:



```powershell

notepad requirements.txt



scvi-tools

scanpy

anndata

mudata

torch

pandas

numpy

matplotlib

scikit-learn

ipykernel

notebook













