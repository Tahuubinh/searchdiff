"""
Helper functions to load tRNA dataset for classifier training.

Usage:
    from dataset.classifier.load_trna_stem_loop_fold import load_trna_dataset

    train_dataset, val_dataset = load_trna_dataset(
        data_path='data/trna/labels/acceptor_stem_t_loop_fold/lb[7.0_6.0_3.0]/ub[9.0_1000.0_1000.0]/data.csv',
        val_split=0.1,
        seed=42
    )
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_trna_csv(data_path='data/trna/labels/acceptor_stem_t_loop_fold/lb[7.0_6.0_3.0]/ub[9.0_1000.0_1000.0]/data.csv'):
    """
    Load tRNA dataset from CSV file.

    Args:
        data_path: Path to CSV file

    Returns:
        pd.DataFrame: DataFrame with tRNA sequences and labels
    """
    df = pd.read_csv(data_path)
    return df


def load_trna_dataset(data_path='data/trna/labels/acceptor_stem_t_loop_fold/lb[7.0_6.0_3.0]/ub[9.0_1000.0_1000.0]/data.csv',
                       val_split=0.1,
                       test_split=0.0,
                       seed=42,
                       stratify=True):
    """
    Load tRNA dataset and split into train/val/test sets.

    Args:
        data_path: Path to CSV file
        val_split: Validation split ratio (0.0 to 1.0)
        test_split: Test split ratio (0.0 to 1.0)
        seed: Random seed for reproducibility
        stratify: Whether to stratify split by label

    Returns:
        tuple: (train_df, val_df, test_df) or (train_df, val_df) if test_split=0
    """
    df = load_trna_csv(data_path)

    if val_split == 0 and test_split == 0:
        return df

    # First split: train + (val + test)
    if stratify:
        stratify_col = df['label']
    else:
        stratify_col = None

    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    if test_split > 0:
        # Split train and temp (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_split + test_split),
            random_state=seed,
            stratify=stratify_col
        )

        # Split temp into val and test
        val_ratio = val_split / (val_split + test_split)
        if stratify:
            stratify_col_temp = temp_df['label']
        else:
            stratify_col_temp = None

        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio),
            random_state=seed,
            stratify=stratify_col_temp
        )

        return train_df, val_df, test_df
    else:
        # Only split train and val
        train_df, val_df = train_test_split(
            df,
            test_size=val_split,
            random_state=seed,
            stratify=stratify_col
        )

        return train_df, val_df


def get_label_distribution(df):
    """
    Get label distribution statistics.

    Args:
        df: DataFrame with 'label' column

    Returns:
        dict: Label distribution statistics
    """
    label_counts = df['label'].value_counts().sort_index()
    total = len(df)

    distribution = {}
    for label in range(4):
        count = label_counts.get(label, 0)
        distribution[label] = {
            'count': int(count),
            'percentage': float(count / total * 100) if total > 0 else 0.0
        }

    return distribution


def filter_by_label(df, min_label=0, max_label=3):
    """
    Filter dataset by label range.

    Args:
        df: DataFrame with 'label' column
        min_label: Minimum label (inclusive)
        max_label: Maximum label (inclusive)

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    return df[(df['label'] >= min_label) & (df['label'] <= max_label)]


def print_dataset_stats(df, name='Dataset'):
    """
    Print dataset statistics.

    Args:
        df: DataFrame with tRNA data
        name: Name of the dataset (for printing)
    """
    print(f"\n{'='*60}")
    print(f"{name} Statistics:")
    print(f"{'='*60}")
    print(f"Total tRNA sequences: {len(df)}")

    # Label distribution
    print(f"\nLabel distribution:")
    dist = get_label_distribution(df)
    for label in range(4):
        stats = dist[label]
        print(f"  Label {label}: {stats['count']:5d} ({stats['percentage']:5.2f}%)")

    # Property statistics
    print(f"\nProperty value ranges:")
    print(f"  Length: {df['length'].min():.0f} - {df['length'].max():.0f}")
    print(f"  Hydrophobic ratio: {df['hydrophobic_ratio'].min():.3f} - {df['hydrophobic_ratio'].max():.3f}")
    print(f"  Net charge: {df['net_charge'].min():.2f} - {df['net_charge'].max():.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    print("Loading tRNA dataset...")

    # Load full dataset
    df = load_trna_csv()
    print_dataset_stats(df, "Full Dataset")

    # Split into train/val
    train_df, val_df = load_trna_dataset(val_split=0.1, seed=42)
    print_dataset_stats(train_df, "Training Set")
    print_dataset_stats(val_df, "Validation Set")

    # Filter for high-quality tRNA sequences (label=3)
    label_3_df = filter_by_label(df, min_label=3, max_label=3)
    print_dataset_stats(label_3_df, "Label=3 tRNA Sequences")
