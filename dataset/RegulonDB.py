import os
import requests
import torch
import random
from torch.utils.data import Dataset, DataLoader

class RegulonDBDataset(Dataset):
    """
    PyTorch Dataset for E. coli Sigma-70 Promoters.
    
    Source: UCI Machine Learning Repository (Molecular Biology - Promoter Gene Sequences).
    Note: Since the original academic GitHub repositories (iPromoter-2L) are 404/down,
    we use the stable UCI archive. The UCI dataset contains 53 positive promoters 
    (57bp length). We perform data augmentation (duplication) to simulate a larger 
    dataset for testing the training pipeline.
    """
    def __init__(self, root_dir='data/regulondb', max_length=57):
        """
        Args:
            root_dir (str): Directory to save/load the dataset.
            max_length (int): Fixed sequence length (UCI standard is 57bp).
        """
        self.max_length = max_length
        # Discrete Diffusion requires integer inputs [0, 1, 2, 3]
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        # 1. Setup Directory
        os.makedirs(root_dir, exist_ok=True)
        self.file_path = os.path.join(root_dir, "promoters.data")
        
        # Stable URL from UCI Archive
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data"

        # 2. Download Data
        if not os.path.exists(self.file_path):
            print(f"[INFO] File not found. Downloading from UCI Repository: {url}")
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(self.file_path, 'w') as f:
                        f.write(response.text)
                    print(f"[INFO] Download and save successful.")
                else:
                    raise Exception(f"UCI Server returned status code: {response.status_code}")
            except Exception as e:
                print(f"[WARN] Network download failed: {e}")
                print(f"[WARN] Creating dummy synthetic data for debugging purposes.")
                self._create_dummy_data()

        # 3. Load and Process Data
        self.sequences = []
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                # UCI Format: "+, S1, AATGCC..."
                parts = line.strip().split(',')
                if len(parts) < 3: 
                    continue
                
                label = parts[0].strip()   # '+' (Promoter) or '-' (Non-promoter)
                seq = parts[2].strip().upper()
                
                # Filter: Keep only Positive class (Promoters) for generative modeling
                if label == '+':
                    # Remove any non-standard characters (only keep ACGT)
                    seq_clean = "".join([c for c in seq if c in self.vocab])
                    
                    if len(seq_clean) >= self.max_length:
                        # Truncate to fixed length if necessary
                        self.sequences.append(seq_clean[:self.max_length])

        # 4. Data Augmentation (Crucial for small datasets)
        # UCI only has ~53 positive samples. We duplicate them 50x to 
        # allow the DataLoader to function with larger batch sizes (e.g., 64).
        if len(self.sequences) > 0:
            print(f"[INFO] Original positive samples found: {len(self.sequences)}")
            self.sequences = self.sequences * 50
            print(f"[INFO] Augmented dataset size (x50 duplication): {len(self.sequences)}")
        else:
            # Fallback if file parsing failed completely
            print("[WARN] No valid sequences found. Generating random synthetic DNA.")
            bases = ['A', 'C', 'G', 'T']
            self.sequences = ["".join(random.choices(bases, k=self.max_length)) for _ in range(1000)]
            
        print(f"[INFO] Dataset ready. Sequence Length: {self.max_length} bp")

    def _create_dummy_data(self):
        """Creates a local dummy file to prevent code crashes during network issues."""
        with open(self.file_path, 'w') as f:
            # Write a few dummy lines in UCI format
            f.write("+, Dummy1, " + "A" * 57 + "\n")
            f.write("+, Dummy2, " + "C" * 57 + "\n")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_str = self.sequences[idx]
        
        # Convert String -> List of Integers -> Tensor
        token_ids = [self.vocab[c] for c in seq_str]
        return torch.tensor(token_ids, dtype=torch.long)

# --- EXECUTION CHECK ---
if __name__ == "__main__":
    # Initialize dataset
    # UCI data is typically 57bp long
    dataset = RegulonDBDataset(root_dir='data/regulondb', max_length=57)
    
    # Create DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Inspect one batch
    try:
        batch = next(iter(loader))
        print(f"\n[TEST] Batch Shape: {batch.shape}")  # Expected: [64, 57]
        print(f"[TEST] Sample Tensor (First 10 tokens): {batch[0][:10]}")
    except StopIteration:
        print("[ERROR] DataLoader is empty.")