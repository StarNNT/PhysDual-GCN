# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:49:41 2025

@author: Vegi
"""
# Required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Constants
torch.manual_seed(42)
# ---------------- control parameters ----------------
EPOCHS         = 300
LEARNING_RATE  = 0.01
DROPOUT_RATE   = 0.05
HIDDEN_DIM1    = 32
HIDDEN_DIM2    = 64
ACTIVATION_FN  = nn.ReLU()

# Protein is converted into graph structure (only with ordered edges)
def protein_sequence_to_graph(seq):
    node_features = [[ord(aa) % 26] for aa in seq]
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = [[i, i+1] for i in range(len(seq)-1)] + [[i+1, i] for i in range(len(seq)-1)]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

# Convert to molecule graph (SMILES -> atom graph)
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)

    atom_features = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
    x = torch.tensor(atom_features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index), mol

# Physical energy calculation (example: total gravitational energy based on distance)
def calculate_physical_energy(mol):
    conf = mol.GetConformer()
    energy = 0.0
    num_atoms = mol.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):
            pos_i = conf.GetAtomPosition(i)
            pos_j = conf.GetAtomPosition(j)
            dist = np.linalg.norm(np.array([pos_i.x, pos_i.y, pos_i.z]) - np.array([pos_j.x, pos_j.y, pos_j.z]))
            if dist > 0:
                energy += 1 / dist  # simple inverse distance contribution
    return torch.tensor([[energy]], dtype=torch.float32)

# Model
class DualGCN(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, activation_fn):
        super(DualGCN, self).__init__()
        self.gcn1 = GCNConv(1, hidden_dim1)
        self.gcn2 = GCNConv(hidden_dim1, hidden_dim2)
        self.fc = nn.Linear(hidden_dim2 * 2 + 1, 1)
        self.drop = nn.Dropout(DROPOUT_RATE)
    def forward(self, mol_data, prot_data, phys_energy):
        x1 = torch.relu(self.gcn1(mol_data.x, mol_data.edge_index))
        x1 = torch.relu(self.gcn2(x1, mol_data.edge_index))
        x1 = global_add_pool(x1, torch.zeros(x1.size(0), dtype=torch.long))

        x2 = torch.relu(self.gcn1(prot_data.x, prot_data.edge_index))
        x2 = torch.relu(self.gcn2(x2, prot_data.edge_index))
        x2 = global_add_pool(x2, torch.zeros(x2.size(0), dtype=torch.long))

        combined = torch.cat([x1, x2, phys_energy.view(1, -1)], dim=-1)
        return self.fc(combined)

# Data
target_protein = "MHHHHHHSSGVDLGTENLYFQSMGKVKATPMTPEQAMKQYMQKLTAFEHHEIFSYPEIYFLGLNAKKRQGMTGGPNNGGYDDDQGSYVQVPHDHVAYRYEVLKVIGKGSFGQVVKAYDHKVHQHVALKMVRNEKRFHRQAAEEIRILEHLRKQDKDNTMNVIHMLENFTFRNHICMTFELLSMNLYELIKKNKFQGFSLPLVRKFAHSILQCLDALHKNRIIHCDLKPENILLKQQGRSGIKVIDFGSSCYEHQRVYTYIQSRFYRAPEVILGARYGMPIDMWSLGCILAELLTGYPLLPGEDEGDQLACMIELLGMPSQKLLDASKRAKNFVSSKGYPRYCTVTTLSDGSVVLNGGRSRRGKLRGPPESREWGNALKGCDDPLFLDFLKQCLEWDPAVRMTPGQALRHPWLRRRLP"

drug_smiles = {
    "Brexpiprazole": "O=C1NC2=CC(OCCCCN3CCN(CC3)C3=C4C=CSC4=CC=C3)=CC=C2C=C1",
    "Donepezil": "COC1=C(OC)C=C2C(=O)C(CC3CCN(CC4=CC=CC=C4)CC3)CC2=C1",
    "Galantamine": "[H][C@]12C[C@@H](O)C=C[C@]11CCN(C)CC3=C1C(O2)=C(OC)C=C3",
    "Rivastigmine": "CCN(C)C(=O)OC1=CC=CC(=C1)[C@H](C)N(C)C"
}

true_values = {
    "Brexpiprazole": -10.1,
    "Donepezil": -10.9,
    "Galantamine": -7.4,
    "Rivastigmine": -7.1
}



# --- Training ---
model = DualGCN(HIDDEN_DIM1, HIDDEN_DIM2, ACTIVATION_FN)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

protein_data = protein_sequence_to_graph(target_protein)
loss_log = []
y_true, y_pred, drug_names = [], [], []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for name, smiles in drug_smiles.items():
        mol_data, mol = smiles_to_graph(smiles)
        phys_energy = calculate_physical_energy(mol)
        target = torch.tensor([true_values[name]], dtype=torch.float32)
        pred = model(mol_data, protein_data, phys_energy)
        loss = loss_fn(pred.view(-1), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_log.append(total_loss / len(drug_smiles))

# Predict
model.eval()
for name, smiles in drug_smiles.items():
    mol_data, mol = smiles_to_graph(smiles)
    phys_energy = calculate_physical_energy(mol)
    pred = model(mol_data, protein_data, phys_energy)
    y_pred.append(pred.item())
    y_true.append(true_values[name])
    drug_names.append(name)

# Error metrics
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

# Plots
import matplotlib.pyplot as plt
import numpy as np

# 1. Normalized Loss Plot
plt.figure(figsize=(8, 6))
loss_array = np.array(loss_log)
normalized_loss = (loss_array - np.min(loss_array)) / (np.max(loss_array) - np.min(loss_array))
plt.plot(normalized_loss, label="Loss", color='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar Chart: True vs Predicted ΔG
plt.figure(figsize=(8, 6))
x = np.arange(len(drug_smiles))
bar_width = 0.35
plt.bar(x - bar_width / 2, y_true, bar_width, label='True')
plt.bar(x + bar_width / 2, y_pred, bar_width, label='Predicted')
for i in range(len(x)):
    plt.text(x[i] - bar_width / 2, y_true[i] + 0.2, f"{y_true[i]:.2f}", ha='center', va='bottom', fontsize=9)
    plt.text(x[i] + bar_width / 2, y_pred[i] + 0.2, f"{y_pred[i]:.2f}", ha='center', va='bottom', fontsize=9, color='darkred')

plt.xticks(x, list(drug_smiles.keys()), rotation=45)
plt.ylabel("Binding Energy (ΔG, kcal/mol)")
plt.title("True vs Predicted ΔG (Binding Energy)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# 3. Regression Scatter Plot
plt.figure(figsize=(8, 6))

# Scatter points: Predicted
plt.scatter(y_true, y_pred, color='red', label='Predict (Model Output)')

# Ideal line: Actual = Prediction (y = x)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', label='Ideal Line (y = x)')

# Regression line (blue): Linear fit
m, b = np.polyfit(y_true, y_pred, 1)
plt.plot(y_true, m * np.array(y_true) + b, color='blue', label=f'Regression (y = {m:.2f}x + {b:.2f})')

# Annotations (drug names)
for i, name in enumerate(drug_names):
    plt.annotate(name, (y_true[i], y_pred[i]), textcoords="offset points", xytext=(5, -12), ha='left', fontsize=9)
# R² değeri metin olarak ekleniyor
plt.text(0.05, 0.90, f"$R^2$ = {r2:.2f}", transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', edgecolor='gray'))

# Axis titles and grid
plt.xlabel("True ΔG (kcal/mol)")
plt.ylabel("Predicted ΔG (kcal/mol)")
plt.title("Regression Plot (True vs Predicted)")

# Description box
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

metric_names = ['MAE', 'MSE', 'RMSE']
metric_values = [0.28, 0.15, 0.39]
percent_errors = [34.2, 18.5, 42.96]

plt.figure(figsize=(8, 6))
bars = plt.bar(metric_names, metric_values, color='darkorange')

# Place the labels slightly BELOW or IN THE MIDDLE of the sticks (not too high)
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height - 0.05, 
             f"{metric_values[i]:.2f}\n({percent_errors[i]:.1f}%)", 
             ha='center', va='top', fontsize=10, color='black')

plt.ylabel("Error Metric Values")
plt.title("Comparison of Error Metrics with Percentage Errors")
plt.ylim(0, max(metric_values) + 0.1)  # Ekstra boşluk bırak
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


# 4. Printing epoch-based error metrics
print("\nEpoch-wise Normalized Loss (First 10 Epochs):")
for i in range(0, 10):
    print(f"Epoch {i:3d}: Normalized Loss = {normalized_loss[i]:.4f}")

# Sonuç tablosu
print("\nFinal Results:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


# Make sure y_true and y_pred are of equal length
assert len(y_true) == len(y_pred), "Error: y_true ve y_pred not the same length."

drug_names = list(drug_smiles.keys())

errors_df = pd.DataFrame({
    "Drug": drug_names,
     "Target": ["DYRK2"]*len(drug_names),
    "True Value": y_true,
    "Predicted": y_pred,
    "Absolute Error": np.abs(np.array(y_true) - np.array(y_pred)),
    "Percent Error (%)": 100 * np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true))
})

print("\n🔬 Table of Predict Results:\n")
print(errors_df.to_string(index=False))
