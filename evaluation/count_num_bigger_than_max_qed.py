from rdkit import Chem
from rdkit.Chem import Crippen, QED
from rdkit import RDLogger
import json
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem import RDConfig
import sys
import os
import argparse
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer  # type: ignore

# Function to calculate logP
def calc_logp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Crippen.MolLogP(mol)

# Function to calculate QED (Quantitative Estimate of Drug-likeness)
def calc_qed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return QED.qed(mol)

def calc_sa(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return sascorer.calculateScore(mol)

def read_smiles(file_path):
    """Read SMILES strings from a file."""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def canonicalize_smiles(smiles_list):
    """Convert a list of SMILES strings to canonical SMILES."""
    canonical_smiles = set()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical_smiles.add(Chem.MolToSmiles(mol))
    return canonical_smiles

def calc_validity(smiles):
    if not smiles:
        return -1
    mol = Chem.MolFromSmiles(smiles)
    return 1 if mol else 0


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Calculate statistics for molecules.")
    parser.add_argument('--folder_path', type=str, required=True, help="Path to the folder containing molecule data.")

    args = parser.parse_args()
    folder_path = args.folder_path
    
    qm9_file = 'data/QM9/qm9_canonical_smiles.txt'

    qm9_smiles = read_smiles(qm9_file)
    max_qed_smiles = "CCCNc1ccco1"
    # Calculate max qed
    max_qed = calc_qed(max_qed_smiles)

    print(f"Max QED from QM9: {max_qed}")

    # Process molecules from seed_0 to seed_4
    counts = []
    for seed in range(5):
        file_path = os.path.join(folder_path, f"seed_{seed}/molecules.txt")
        smiles_list = read_smiles(file_path)
    
        # Filter valid and novel molecules
        valid_smiles = [smiles for smiles in smiles_list if calc_validity(smiles) == 1]
        valid_set = canonicalize_smiles(valid_smiles)
        novel_smiles = valid_set - set(qm9_smiles)

        # Count molecules with QED > max_qed
        count = sum(1 for smiles in novel_smiles if calc_qed(smiles) > max_qed)
        counts.append(count)

    # Calculate mean and std
    mean_count = sum(counts) / len(counts)
    std_count = (sum((x - mean_count) ** 2 for x in counts) / len(counts)) ** 0.5

    # Save results to JSON
    result = {
        "counts": counts,
        "mean": mean_count,
        "std": std_count
    }
    output_file = os.path.join(folder_path, "qed_statistics.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Counts of molecules with QED > {max_qed} for each seed: {counts}")
    print(f"Mean count: {mean_count}")
    print(f"Standard deviation: {std_count}")

    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()