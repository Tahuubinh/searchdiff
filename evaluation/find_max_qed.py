from rdkit import Chem
from rdkit.Chem import QED
from tqdm import tqdm

# Read SMILES from file
smiles_file = "data/QM9/qm9_canonical_smiles.txt"
with open(smiles_file, 'r') as f:
    smiles_list = [line.strip() for line in f if line.strip()]

# Calculate QED for each molecule
max_qed = -1
max_qed_smiles = None

for smiles in tqdm(smiles_list, desc="Processing molecules"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        qed_value = QED.qed(mol)
        if qed_value > max_qed:
            max_qed = qed_value
            max_qed_smiles = smiles

# Print results
print(f"Molecule with highest QED: {max_qed_smiles}")
print(f"QED value: {max_qed:.4f}")
