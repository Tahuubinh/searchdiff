import os
import sys
import argparse
from itertools import combinations
import json
import numpy as np

# Add the project root directory to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate mean, std properties")
    parser.add_argument("--folder_path", type=str, required=True, help="Path to the input folder")
    return parser.parse_args()

def calculate_mean_std(folder_path, considerred_folder_list):
    results = {}
    
    for folder in considerred_folder_list:
        json_path = os.path.join(folder_path, folder, "molecules_statistic.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                for key, value in data.items():
                    if isinstance(value, (int, float)) and value is not None:
                        if key not in results:
                            results[key] = []
                        results[key].append(value)

    mean_std_results = {}
    for key, values in results.items():
        mean_std_results[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }

    # Save results to mean_std folder
    output_folder = os.path.join(folder_path, "mean_std")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "mean_std_molecules_results.json")
    with open(output_path, "w") as f:
        json.dump(mean_std_results, f, indent=4)

if __name__ == "__main__":
    args = parse_arguments()
    folder_path = args.folder_path
    considerred_folder_list = ['seed_0', 'seed_1', 'seed_2', 'seed_3', 'seed_4']

    calculate_mean_std(folder_path, considerred_folder_list)
