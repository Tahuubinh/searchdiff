import os
import requests
import pandas as pd

# URL of the CSV file
url = "https://raw.githubusercontent.com/zswitten/Antimicrobial-Peptides/master/data/grampa.csv"

# Local directory and filename
save_dir = "./data"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "grampa.csv")

# Download and save
response = requests.get(url)
response.raise_for_status()  # raise error if download failed

with open(save_path, "wb") as f:
    f.write(response.content)

print(f"File downloaded and saved to {save_path}")

# Load the CSV file into a DataFrame
df = pd.read_csv(save_path)

# Extract the 'sequence' column
sequences = df['sequence'].tolist()
print(f"Number of sequences: {len(sequences)}")
# First, filter unique sequences
unique_sequences = set(sequences)
# Then, filter by length <= 46
filtered_sequences = [seq for seq in unique_sequences if len(seq) <= 46]

# Save the filtered sequences to a file
output_path = os.path.join(save_dir, "filtered_peptides_46.txt")
with open(output_path, "w") as f:
    for seq in filtered_sequences:
        f.write(seq + "\n")

print(f"Number of unique sequences: {len(unique_sequences)}")
print(f"Number of unique sequences with length <= 46: {len(filtered_sequences)}")
print(f"Filtered sequences saved to {output_path}")

