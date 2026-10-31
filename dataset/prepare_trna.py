#!/usr/bin/env python3
"""Prepare the mammalian tRNA training file from GtRNAdb FASTA files."""

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "trna" / "mature"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "trna" / "final_sequences.txt"
FASTA_PATTERN = "*-mature-tRNAs.fa"


def read_fasta(path):
    """Yield sequences from a FASTA file, including multiline records."""
    sequence_parts = []

    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sequence_parts:
                    yield "".join(sequence_parts)
                    sequence_parts = []
            else:
                sequence_parts.append(line)

    if sequence_parts:
        yield "".join(sequence_parts)


def prepare_trna(input_dir=DEFAULT_INPUT_DIR, output_path=DEFAULT_OUTPUT_PATH):
    """Merge mature tRNA FASTA records and convert DNA thymine to RNA uracil."""
    input_dir = Path(input_dir)
    fasta_paths = sorted(input_dir.glob(FASTA_PATTERN))
    if not fasta_paths:
        raise FileNotFoundError(
            f"No files matching {FASTA_PATTERN!r} were found in {input_dir}"
        )

    sequences = []
    for fasta_path in fasta_paths:
        sequences.extend(
            sequence.upper().replace("T", "U")
            for sequence in read_fasta(fasta_path)
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for sequence in sequences:
            file.write(f"{sequence}\n")

    print(
        f"Saved {len(sequences):,} sequences from {len(fasta_paths)} "
        f"FASTA files to {output_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing {FASTA_PATTERN} files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the combined one-sequence-per-line output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    prepare_trna(args.input_dir, args.output)


if __name__ == "__main__":
    main()
