import argparse
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import TanimotoSimilarity
import statistics
import sys
from tqdm import tqdm
import os
from rdkit import RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

def read_valid_molecules(file_path):
    """Read and filter valid molecules from a file."""
    valid_molecules = []
    with open(file_path, 'r') as file:
        for line in file:
            smiles = line.strip()
            mol = Chem.MolFromSmiles(smiles)
            if mol:  # Ensure valid molecule
                valid_molecules.append(mol)
    return valid_molecules

def generate_fingerprints(molecules):
    """Generate fingerprints for a list of molecules."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [generator.GetFingerprint(mol) for mol in molecules]
    return fingerprints

def main():
    parser = argparse.ArgumentParser(description="Calculate Tanimoto Similarity for molecules in a file.")
    parser.add_argument('--input_file', type=str, help="Path to the input file containing SMILES strings.")
    args = parser.parse_args()

    # Read and filter valid molecules
    valid_molecules = read_valid_molecules(args.input_file)

    # Print the number of valid molecules
    print(f"Number of valid molecules: {len(valid_molecules)}")

    # Generate fingerprints
    fingerprints = generate_fingerprints(valid_molecules)

    # Calculate Tanimoto Similarity for all pairs
    similarities = []
    print("Tanimoto Similarity between valid molecules:")
    for i in tqdm(range(len(fingerprints)), desc="Processing molecules"):
        for j in range(i + 1, len(fingerprints)):
            similarity = TanimotoSimilarity(fingerprints[i], fingerprints[j])
            similarities.append(similarity)

    # Calculate and print mean and standard deviation
    if similarities:
        mean_similarity = statistics.mean(similarities)
        std_similarity = statistics.stdev(similarities)
        print(f"\nMean Tanimoto Similarity: {mean_similarity:.4f}")
        print(f"Standard Deviation: {std_similarity:.4f}")

        # Save results to a file
        output_file = os.path.join(os.path.dirname(args.input_file), "tanimoto_results.txt")
        with open(output_file, 'w') as f:
            f.write(f"Number of valid molecules: {len(valid_molecules)}\n")
            f.write(f"Mean Tanimoto Similarity: {mean_similarity:.4f}\n")
            f.write(f"Standard Deviation: {std_similarity:.4f}\n")
        print(f"Results saved to {output_file}")
    else:
        print("\nNo valid similarities calculated.")

if __name__ == "__main__":
    main()