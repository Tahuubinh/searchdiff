"""
Script to prepare peptides dataset with labels for classifier training.

This script:
1. Reads peptides from filtered_peptides_46.txt
2. Calculates properties: peptide_length, hydrophobic_ratio, net_charge
3. Assigns labels (0-3) based on number of properties satisfied
4. Saves to CSV and JSON formats

Usage:
    python dataset/classifier/prepare_peptides_labels.py

Output:
    - data/peptides/peptides_with_labels.csv
    - data/peptides/peptides_with_labels.json
"""

import os
import sys
import json
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from properties.peptides import (
    calculate_peptide_length,
    calculate_hydrophobic_ratio,
    calculate_net_charge
)


def check_property_satisfied(value, lower_bound, upper_bound):
    """Check if a property value satisfies the bounds."""
    if value is None:
        return False
    return lower_bound <= value <= upper_bound


def calculate_label(length, hydrophobic_ratio, net_charge, 
                    length_bounds=(10, 50),
                    hydrophobic_bounds=(0.3, float('inf')),
                    charge_bounds=(2, 9)):
    """
    Calculate label (0-3) based on number of properties satisfied.
    
    Args:
        length: Peptide length
        hydrophobic_ratio: Hydrophobic ratio (percentage)
        net_charge: Net charge at pH 7.0
        length_bounds: (min, max) for length
        hydrophobic_bounds: (min, max) for hydrophobic ratio (in percentage)
        charge_bounds: (min, max) for net charge
    
    Returns:
        int: Label from 0 to 3
    """
    num_satisfied = 0
    
    # Check each property
    if check_property_satisfied(length, length_bounds[0], length_bounds[1]):
        num_satisfied += 1
    
    # Hydrophobic ratio is in percentage (0-100), so multiply bound by 100
    if check_property_satisfied(hydrophobic_ratio, hydrophobic_bounds[0] * 100, hydrophobic_bounds[1] * 100):
        num_satisfied += 1
    
    if check_property_satisfied(net_charge, charge_bounds[0], charge_bounds[1]):
        num_satisfied += 1
    
    return num_satisfied


def process_peptides(input_file, output_csv, output_json,
                     length_bounds=(10, 50),
                     hydrophobic_bounds=(0.3, float('inf')),
                     charge_bounds=(2, 9)):
    """
    Process peptides file and generate labeled dataset.
    
    Args:
        input_file: Path to input file with peptides (one per line)
        output_csv: Path to output CSV file
        output_json: Path to output JSON file
        length_bounds: (min, max) for peptide length
        hydrophobic_bounds: (min, max) for hydrophobic ratio (as fraction, will be converted to %)
        charge_bounds: (min, max) for net charge
    """
    print(f"Reading peptides from: {input_file}")
    
    # Read peptides
    with open(input_file, 'r') as f:
        peptides = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(peptides)} peptides")
    
    # Process each peptide
    data = []
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for peptide in tqdm(peptides, desc="Processing peptides"):
        try:
            # Calculate properties
            length = calculate_peptide_length(peptide)
            hydrophobic_ratio = calculate_hydrophobic_ratio(peptide)  # Returns percentage
            net_charge = calculate_net_charge(peptide, pH=7.0)
            
            # Calculate label
            label = calculate_label(
                length, hydrophobic_ratio, net_charge,
                length_bounds, hydrophobic_bounds, charge_bounds
            )
            
            # Check individual properties
            length_satisfied = check_property_satisfied(length, length_bounds[0], length_bounds[1])
            hydrophobic_satisfied = check_property_satisfied(hydrophobic_ratio, 
                                                             hydrophobic_bounds[0] * 100, 
                                                             hydrophobic_bounds[1] * 100)
            charge_satisfied = check_property_satisfied(net_charge, charge_bounds[0], charge_bounds[1])
            
            # Store data
            data.append({
                'sequence': peptide,
                'length': length,
                'hydrophobic_ratio': hydrophobic_ratio / 100.0,  # Convert back to fraction for storage
                'net_charge': net_charge,
                'length_satisfied': int(length_satisfied),
                'hydrophobic_satisfied': int(hydrophobic_satisfied),
                'charge_satisfied': int(charge_satisfied),
                'label': label
            })
            
            label_counts[label] += 1
            
        except Exception as e:
            print(f"Error processing peptide '{peptide}': {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    print(f"Total peptides: {len(df)}")
    print(f"\nLabel distribution:")
    for label in range(4):
        count = label_counts[label]
        percentage = (count / len(df)) * 100 if len(df) > 0 else 0
        print(f"  Label {label}: {count:5d} ({percentage:5.2f}%)")
    
    print(f"\nProperty satisfaction rates:")
    print(f"  Length ({length_bounds[0]}-{length_bounds[1]}): "
          f"{df['length_satisfied'].sum()} ({df['length_satisfied'].mean()*100:.2f}%)")
    print(f"  Hydrophobic ratio (>={hydrophobic_bounds[0]}): "
          f"{df['hydrophobic_satisfied'].sum()} ({df['hydrophobic_satisfied'].mean()*100:.2f}%)")
    print(f"  Net charge ({charge_bounds[0]}-{charge_bounds[1]}): "
          f"{df['charge_satisfied'].sum()} ({df['charge_satisfied'].mean()*100:.2f}%)")
    
    print(f"\nProperty value ranges:")
    print(f"  Length: {df['length'].min():.0f} - {df['length'].max():.0f}")
    print(f"  Hydrophobic ratio: {df['hydrophobic_ratio'].min():.3f} - {df['hydrophobic_ratio'].max():.3f}")
    print(f"  Net charge: {df['net_charge'].min():.2f} - {df['net_charge'].max():.2f}")
    print("=" * 60)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved CSV to: {output_csv}")
    
    # Save to JSON
    data_dict = {
        'metadata': {
            'num_peptides': len(df),
            'label_counts': label_counts,
            'constraints': {
                'length': {'min': length_bounds[0], 'max': length_bounds[1]},
                'hydrophobic_ratio': {'min': hydrophobic_bounds[0], 'max': hydrophobic_bounds[1]},
                'net_charge': {'min': charge_bounds[0], 'max': charge_bounds[1]}
            }
        },
        'data': data
    }
    
    with open(output_json, 'w') as f:
        json.dump(data_dict, f, indent=2)
    print(f"Saved JSON to: {output_json}")
    
    return df


def main():
    # File paths
    input_file = 'data/peptides/filtered_peptides_46.txt'
    output_csv = 'data/peptides/peptides_with_labels.csv'
    output_json = 'data/peptides/peptides_with_labels.json'
    
    # Property constraints
    # 10 <= length <= 50
    # hydrophobic_ratio >= 0.3 (30%)
    # 2 <= net_charge <= 9
    length_bounds = (10, 50)
    hydrophobic_bounds = (0.3, float('inf'))  # >= 0.3 (30%)
    charge_bounds = (2, 9)
    
    # Process peptides
    df = process_peptides(
        input_file=input_file,
        output_csv=output_csv,
        output_json=output_json,
        length_bounds=length_bounds,
        hydrophobic_bounds=hydrophobic_bounds,
        charge_bounds=charge_bounds
    )
    
    # Print sample data
    print("\n" + "=" * 60)
    print("Sample data (first 10 rows):")
    print("=" * 60)
    print(df.head(10).to_string(index=False))
    print("\n" + "=" * 60)
    print("Sample data (peptides with label=3):")
    print("=" * 60)
    label_3_samples = df[df['label'] == 3].head(10)
    if len(label_3_samples) > 0:
        print(label_3_samples.to_string(index=False))
    else:
        print("No peptides with label=3 found")
    print("=" * 60)


if __name__ == "__main__":
    main()
