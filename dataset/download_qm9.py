#!/usr/bin/env python3
"""Download QM9 and export the canonical SMILES training split."""

from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "QM9" / "qm9_canonical_smiles.txt"


def prepare_qm9(output_path=OUTPUT_PATH):
    """Write one canonical SMILES string per line for the QM9 training split."""
    dataset = load_dataset("yairschiff/qm9", split="train")
    train_dataset = dataset.train_test_split(test_size=0.05, seed=42)["train"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for smiles in train_dataset["canonical_smiles"]:
            file.write(f"{smiles}\n")

    print(f"Saved {len(train_dataset):,} canonical SMILES to {output_path}")


if __name__ == "__main__":
    prepare_qm9()
