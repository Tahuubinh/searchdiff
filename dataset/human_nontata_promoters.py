from genomic_benchmarks.loc2seq import download_dataset
from pathlib import Path

dataset_path = download_dataset(
    "human_nontata_promoters", 
    version=0, 
    dest_path="data/human_nontata_promoters"
)

print(f"Data downloaded to: {dataset_path}")