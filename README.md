# PhysDual-GCN
PhysDual-GCN: Physics-Informed Graph Neural Network for DYRK2 Binding Affinity Prediction
This repository contains the source code, trained model parameters, and input datasets used in the study:
"Physics-Informed Graph Neural Network for Predicting DYRK2 Binding Affinities of FDA-Approved Alzheimer’s Drugs."

✅ Publication: 

📋 Overview
DYRK2 is an emerging therapeutic target in Alzheimer’s disease (AD). This work introduces PhysDual-GCN, a physics-informed Graph Neural Network (GNN) that integrates physical energy terms (Coulomb and Lennard-Jones potentials) into a GNN architecture to predict binding affinities between DYRK2 and FDA-approved AD drugs (Donepezil, Brexpiprazole, Galantamine, and Rivastigmine).

📁 Repository Contents
src/ — Source code for training and inference

models/ — Pre-trained model weights

data/ — Input datasets (SMILES strings & DYRK2 sequence)

notebooks/ — Example Jupyter notebooks for running experiments

results/ — Sample output files (predictions, figures)

🚀 Getting Started
Prerequisites
Python ≥ 3.8

PyTorch ≥ 1.10

RDKit

NetworkX

Other dependencies listed in requirements.txt

Installation
bash
Kopyala
Düzenle
git clone https://github.com/StarNNT/PhysDual-GCN.git
cd PhysDual-GCN
pip install -r requirements.txt
Running Experiments
To reproduce the results:

bash
python src/train.py --config configs/config.yaml
python src/evaluate.py --weights models/best_model.pt
or open the example notebook in notebooks/.

📊 Results
The model achieves competitive or superior predictive performance compared to classical docking tools and AI-based baselines, with improved interpretability thanks to the integration of physical interaction terms.

Metric	Donepezil	Brexpiprazole	Galantamine	Rivastigmine
MAE	0.35	0.31	0.41	0.47
RMSE	0.49	0.44	0.57	0.61
R²	0.975	0.985	0.962	0.953

See results/ folder for full outputs.

🔗 Citation
If you use this work in your research, please cite:

graphql
@article{your2025paper,
  author = {Veysel Gider et al.},
  title = {Physics-Informed Graph Neural Network for Predicting DYRK2 Binding Affinities of FDA-Approved Alzheimer’s Drugs},
  journal = {…},
  year = {2025},
  doi = {…}
}
📬 Contact
For any questions or contributions, please open an issue or contact [your email here].

📝 License
This project is licensed under the MIT License — see the LICENSE file for details.
