import os
import sys
import json
import pandas as pd
import argparse
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from properties.peptides import *
from properties.molecules import *
from properties.trna import *

def parse_arguments():
    """Parse command-line arguments for molecule generation."""
    parser = argparse.ArgumentParser(description="Generate molecules with optional x_theta modifications")
    parser.add_argument('--data', type=str, required=True,
                           choices=['qm9', 'grampa', 'trna'],
                           help='Type of dataset to use')
    # Basic generation parameters
    parser.add_argument('--num_samples', type=int, default=1024,
                       help='Number of molecules to generate (default: 100)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for generation')
    parser.add_argument('--version', type=str, default='1',
                       help='Version of the model or configuration (default: None)')
    
    # Posterior sampling method parameters
    parser.add_argument("--property_type", type=str, nargs='+', default=["qed"], help="List of property types")
    parser.add_argument("--lower_bound", type=float, nargs='+', default=None, help="List of lower bounds for the property values")
    parser.add_argument("--upper_bound", type=float, nargs='+', default=None, help="List of upper bounds for the property values")

    return parser.parse_args()

def check_property_satisfied(value, lower_bound, upper_bound):
    """Check if a property value satisfies the bounds."""
    if value is None:
        return False
    return lower_bound <= value <= upper_bound


def calculate_label(properties, lower_bounds, upper_bounds):
    """
    Calculate label based on the number of properties satisfied.

    Args:
        properties (list): List of property values.
        lower_bounds (list): List of lower bounds for each property.
        upper_bounds (list): List of upper bounds for each property.

    Returns:
        int: Label representing the number of satisfied constraints.
    """
    num_satisfied = 0
    for value, lower, upper in zip(properties, lower_bounds, upper_bounds):
        if check_property_satisfied(value, lower, upper):
            num_satisfied += 1
    return num_satisfied


def process_sequences(input_file, output_csv, output_json, property_calcs,property_type, lower_bounds, upper_bounds):
    """
    Process sequences and calculate labels based on property constraints.

    Args:
        input_file (str): Path to the input file containing sequences.
        output_csv (str): Path to save the output CSV file.
        output_json (str): Path to save the output JSON file.
        property_calcs (list): List of property calculation functions.
        lower_bounds (list): List of lower bounds for each property.
        upper_bounds (list): List of upper bounds for each property.

    Returns:
        pd.DataFrame: DataFrame containing processed sequences and labels.
    """
    print(f"Reading sequences from: {input_file}")

    # Read sequences
    with open(input_file, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]

    print(f"Found {len(sequences)} sequences")

    # Process each sequence
    data = []
    label_counts = {}

    for sequence in tqdm(sequences, desc="Processing sequences"):
        try:
            # Calculate properties dynamically
            properties = [calc(sequence) for calc in property_calcs]

            # Calculate label
            label = calculate_label(properties, lower_bounds, upper_bounds)

            # Store data
            data.append({
                'sequence': sequence,
                **{
                    prop: value for prop, value in zip(property_type, properties)
                },
                **{
                    f"{prop}_satisfied": int(check_property_satisfied(value, lower, upper))
                    for prop, value, lower, upper in zip(property_type, properties, lower_bounds, upper_bounds)
                },
                'label': label
            })

            label_counts[label] = label_counts.get(label, 0) + 1

        except Exception as e:
            print(f"Error processing sequence '{sequence}': {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(data)

    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    print(f"Total sequences: {len(df)}")
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"Label {label}: {count}")

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved CSV to: {output_csv}")

    # Save to JSON
    data_dict = {
        'metadata': {
            'num_sequences': len(df),
            'label_counts': label_counts,
            'constraints': {
                prop: {'min': lower, 'max': upper}
                for prop, lower, upper in zip(property_type, lower_bounds, upper_bounds)
            }
        },
        'data': data
    }

    with open(output_json, 'w') as f:
        json.dump(data_dict, f, indent=2)
    print(f"Saved JSON to: {output_json}")

    return df


def main():
    # Parse command-line arguments
    args = parse_arguments()
    # Print all arguments
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    data = args.data
    property_type = args.property_type  
    lower_bound = args.lower_bound
    upper_bound = args.upper_bound

    property_string = "_".join(property_type)
    lower_bound_string = "_".join(map(str, lower_bound))
    upper_bound_string = "_".join(map(str, upper_bound))

    # File paths
    if data == 'grampa':
        input_file = 'data/peptides/filtered_peptides_46.txt'
        output_csv = f'data/peptides/labels/{property_string}/lb[{lower_bound_string}]/ub[{upper_bound_string}]/data.csv'
        output_json = f'data/peptides/labels/{property_string}/lb[{lower_bound_string}]/ub[{upper_bound_string}]/data.json'
    elif data == 'trna':
        input_file = 'data/trna/final_sequences.txt'
        output_csv = f'data/trna/labels/{property_string}/lb[{lower_bound_string}]/ub[{upper_bound_string}]/data.csv'
        output_json = f'data/trna/labels/{property_string}/lb[{lower_bound_string}]/ub[{upper_bound_string}]/data.json'
    else:
        raise ValueError(f"Unsupported data type: {data}")

    

    property_calcs = []
    for prop in property_type:
        calc_func = {
            "qed": calc_qed,
            "sa": calc_sa,
            "sa_qed_weighted": calc_sa_qed_weighted,
            "sa3_qed06_weighted": calc_sa3_qed06_weighted,
            "valid": calc_validity,
            "peptide_length": calculate_peptide_length,
            "peptide_length_10_50_score": calculate_peptide_length_10_50_score,
            "hydrophobic_ratio": calculate_hydrophobic_ratio,
            "hydrophobic_ratio_gte_03_score": calculate_hydrophobic_ratio_gte_03_score,
            "net_charge": calculate_net_charge,
            "net_charge_2_9_score": calculate_net_charge_2_9_score,
            "trna_length": calc_trna_length,
            "acceptor_stem": count_acceptor_stem,
            "t_loop": get_t_loop_score,
            "fold": check_fold_structure_constraints,
        }.get(prop)
        if calc_func is None:
            raise ValueError(f"Unsupported property type: {prop}")
        property_calcs.append(calc_func)
    
    # Process sequences
    df = process_sequences(
        input_file=input_file,
        output_csv=output_csv,
        output_json=output_json,
        property_calcs=property_calcs,
        property_type=property_type,
        lower_bounds=lower_bound,
        upper_bounds=upper_bound
    )
    


if __name__ == "__main__":
    main()
