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
    parser.add_argument("--result_path", type=str, default="molecules_prop_calc/['peptide_length', 'hydrophobic_ratio', 'net_charge']/[10.0, 0.3, 2.0]/[50.0, 1000.0, 9.0]/properties.json", help="Path to save the result file")
    return parser.parse_args()

def calculate_mean_std(folder_path, considerred_folder_list, result_path):
    results = {}

    for folder in considerred_folder_list:
        json_path = os.path.join(folder_path, folder, result_path)
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)

                # Process property_satisfy_counts
                # if "property_satisfy_counts" in data:
                #     for key, value in data["property_satisfy_counts"].items():
                #         if isinstance(value, (int, float)) and value is not None:
                #             if key not in results:
                #                 results[key] = []
                #             results[key].append(value)

                # Process subset_satisfy_counts
                if "subset_satisfy_counts" in data:
                    for key, value in data["subset_satisfy_counts"].items():
                        if isinstance(value, (int, float)) and value is not None:
                            if key not in results:
                                results[key] = []
                            results[key].append(value)

                # Process novel_sequences_count
                if "novel_sequences_count" in data:
                    if "novel_sequences_count" not in results:
                        results["novel_sequences_count"] = []
                    results["novel_sequences_count"].append(data["novel_sequences_count"])

    mean_std_results = {}
    for key, values in results.items():
        mean_std_results[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values))
        }

    # Save results to mean_std folder
    output_folder = os.path.join(folder_path, "mean_std")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "mean_std_property_results.json")
    with open(output_path, "w") as f:
        json.dump(mean_std_results, f, indent=4)

if __name__ == "__main__":
    args = parse_arguments()
    folder_path = args.folder_path
    result_path = args.result_path
    considerred_folder_list = ['seed_0', 'seed_1', 'seed_2', 'seed_3', 'seed_4']

    calculate_mean_std(folder_path, considerred_folder_list, result_path)
