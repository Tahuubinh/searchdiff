#!/usr/bin/env python3
"""Download and preprocess the GRAMPA peptide dataset."""

import csv
from pathlib import Path
from urllib.request import urlretrieve


GRAMPA_URL = (
    "https://raw.githubusercontent.com/zswitten/"
    "Antimicrobial-Peptides/master/data/grampa.csv"
)
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Witten and Witten (2019) selected 46 residues as the 95th percentile of
# peptide length: https://doi.org/10.1101/692681
MAX_PEPTIDE_LENGTH = 46

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "peptides"
RAW_DATA_PATH = DATA_DIR / "grampa.csv"
OUTPUT_PATH = DATA_DIR / "filtered_peptides_46.txt"


def download_grampa(output_path=RAW_DATA_PATH):
    """Download the public GRAMPA CSV unless it is already available."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        print(f"Downloading GRAMPA to {output_path}")
        urlretrieve(GRAMPA_URL, output_path)
    else:
        print(f"Using existing GRAMPA file: {output_path}")
    return output_path


def prepare_peptides(input_path=RAW_DATA_PATH, output_path=OUTPUT_PATH):
    """Filter GRAMPA and write one unique peptide sequence per line."""
    sequences = set()

    with Path(input_path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            sequence = row["sequence"].strip().upper()
            if (
                sequence
                and len(sequence) <= MAX_PEPTIDE_LENGTH
                and set(sequence) <= STANDARD_AMINO_ACIDS
            ):
                sequences.add(sequence)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for sequence in sorted(sequences):
            file.write(f"{sequence}\n")

    print(f"Saved {len(sequences):,} unique peptide sequences to {output_path}")
    return sequences


def main():
    raw_data_path = download_grampa()
    prepare_peptides(raw_data_path)


if __name__ == "__main__":
    main()
