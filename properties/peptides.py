from Bio.SeqUtils.ProtParam import ProteinAnalysis
import os
import torch

# Nelson, David L., Albert L. Lehninger, and Michael M. Cox. Lehninger principles of biochemistry. Macmillan, 2008.
STRICT_HYDROPHOBIC_RESIDUES = set(['A', 'V', 'L', 'I', 'M', 'F', 'P', 'G'])

aa_candidates = ["A", "C", "E", "D", "G", "F", "I", "H", "K", "M", "L", "N", "Q", "P", "S", "R", "T", "W", "V", "Y"]
aa_candidates_with_pad = aa_candidates + ['<pad>']

def calculate_hydrophobic_ratio(sequence):
    """
    Calculates the percentage of core hydrophobic residues in a protein sequence.
    Args:
        sequence (str): Amino acid sequence (e.g., "GIGKFL...")
    Returns:
        float: Hydrophobic ratio in percentage (0.0 to 100.0)
    """
    if not sequence:
        return None
    
    # Count only the strict hydrophobic residues
    count = sum(1 for aa in sequence if aa in STRICT_HYDROPHOBIC_RESIDUES)
    
    # Calculate percentage
    ratio = (count / len(sequence))
    return ratio

def calculate_net_charge(sequence, pH=7.0):
    """
    Calculates the net charge of a peptide sequence at a specific pH.
    """
    if not sequence:
        return None
    # Create analysis object
    analyser = ProteinAnalysis(sequence.upper())
    
    # Calculate charge
    charge = analyser.charge_at_pH(pH)
    
    return charge

def calculate_peptide_length(sequence):
    """
    Calculates the length of a peptide sequence.
    Args:
        sequence (str): Amino acid sequence (e.g., "GIGKFL...")
    Returns:
        int: Length of the peptide sequence
    """
    return len(sequence)

def calculate_hydrophobic_ratio_normalize(sequence):
    """
    Calculates the percentage of core hydrophobic residues in a protein sequence.
    Args:
        sequence (str): Amino acid sequence (e.g., "GIGKFL...")
    Returns:
        float: Hydrophobic ratio in percentage (0.0 to 100.0)
    """
    if not sequence:
        return None
    
    # Count only the strict hydrophobic residues
    count = sum(1 for aa in sequence if aa in STRICT_HYDROPHOBIC_RESIDUES)
    
    # Calculate percentage
    ratio = (count / len(sequence))
    normalized_ratio = ratio / 0.3
    if normalized_ratio < 0:
        return -normalized_ratio
    return normalized_ratio - 1

def calculate_net_charge_normalize(sequence, pH=7.0):
    """
    Calculates the net charge of a peptide sequence at a specific pH.
    """
    if not sequence:
        return None
    # Create analysis object
    analyser = ProteinAnalysis(sequence.upper())
    
    # Calculate charge
    charge = analyser.charge_at_pH(pH)

    normalized_charge = (charge - 2) / 7
    
    if normalized_charge < 0:
        return -normalized_charge
    return normalized_charge - 1

def calculate_peptide_length_normalize(sequence):
    """
    Calculates the length of a peptide sequence.
    Args:
        sequence (str): Amino acid sequence (e.g., "GIGKFL...")
    Returns:
        int: Length of the peptide sequence
    """
    normalized_length = (len(sequence) - 10) / 40
    if normalized_length < 0:
        return -normalized_length
    return normalized_length - 1

def calculate_peptide_normalize(sequence):
    if not sequence:
        return None
    return calculate_peptide_length_normalize(sequence) + calculate_hydrophobic_ratio_normalize(sequence) + calculate_net_charge_normalize(sequence)


def calculate_peptide_length_10_50_score(sequence):
    """Constraint score for 10 <= length <= 50.

    Returns 0.0 when satisfied and negative values outside range.
    """
    if not sequence:
        return None
    length = len(sequence)
    violation = max(0.0, 10.0 - float(length)) + max(0.0, float(length) - 50.0)
    return -violation / 40.0


def calculate_hydrophobic_ratio_gte_03_score(sequence):
    """Constraint score for hydrophobic_ratio >= 0.3.

    Returns 0.0 when satisfied and negative values below threshold.
    """
    ratio = calculate_hydrophobic_ratio(sequence)
    if ratio is None:
        return None
    violation = max(0.0, 0.3 - float(ratio))
    return -violation / 0.3


def calculate_net_charge_2_9_score(sequence):
    """Constraint score for 2 <= net_charge <= 9.

    Returns 0.0 when satisfied and negative values outside range.
    """
    charge = calculate_net_charge(sequence)
    if charge is None:
        return None
    violation = max(0.0, 2.0 - float(charge)) + max(0.0, float(charge) - 9.0)
    return -violation / 7.0

def calc_hydrophobic_ratio_parallel(sequence_list, batch_size, device):
    rewards = torch.zeros(batch_size, device=device)
    for i, sequence in enumerate(sequence_list):
        rewards[i] = calculate_hydrophobic_ratio(sequence)

    return rewards

def calc_net_charge_parallel(sequence_list, batch_size, device):
    rewards = torch.zeros(batch_size, device=device)
    for i, sequence in enumerate(sequence_list):
        rewards[i] = calculate_net_charge(sequence)

    return rewards

def calc_peptide_length_parallel(sequence_list, batch_size, device):
    rewards = torch.zeros(batch_size, device=device)
    for i, sequence in enumerate(sequence_list):
        rewards[i] = calculate_peptide_length(sequence)

    return rewards

def calculate_peptide_normalize_parallel(sequence_list, batch_size, device):
    rewards = torch.zeros(batch_size, device=device)
    for i, sequence in enumerate(sequence_list):
        rewards[i] = calculate_peptide_normalize(sequence)

    return rewards


def calc_peptide_length_10_50_score_parallel(sequence_list, batch_size, device):
    rewards = torch.zeros(batch_size, device=device)
    for i, sequence in enumerate(sequence_list):
        value = calculate_peptide_length_10_50_score(sequence)
        rewards[i] = float('nan') if value is None else value
    return rewards


def calc_hydrophobic_ratio_gte_03_score_parallel(sequence_list, batch_size, device):
    rewards = torch.zeros(batch_size, device=device)
    for i, sequence in enumerate(sequence_list):
        value = calculate_hydrophobic_ratio_gte_03_score(sequence)
        rewards[i] = float('nan') if value is None else value
    return rewards


def calc_net_charge_2_9_score_parallel(sequence_list, batch_size, device):
    rewards = torch.zeros(batch_size, device=device)
    for i, sequence in enumerate(sequence_list):
        value = calculate_net_charge_2_9_score(sequence)
        rewards[i] = float('nan') if value is None else value
    return rewards

if __name__ == "__main__":
    # Example usage
    # File path
    file_path = "sample_results/grampa/topk_all_nucleus_all/standard/standard/['valid']_lb[0.5]_ub[1000.0]/test/molecules.txt"

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
    else:
        # Read the first 10 lines
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()[:10]]

        # Calculate metrics for each sequence
        for i, sequence in enumerate(lines, start=1):
            hydrophobic_ratio = calculate_hydrophobic_ratio(sequence)
            net_charge = calculate_net_charge(sequence)
            peptide_length = calculate_peptide_length(sequence)

            print(f"Sequence {i}:")
            print(f"  Hydrophobic Ratio: {hydrophobic_ratio:.2f}%")
            print(f"  Net Charge: {net_charge:.2f}")
            print(f"  Length: {peptide_length}")