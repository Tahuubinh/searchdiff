import os
import requests
import torch
from torch.utils.data import Dataset, DataLoader

class SpliceJunctionDataset(Dataset):
    """
    Dataset: Primate Splice-Junction Gene Sequences (UCI).
    - Sequence Length: 60 bp
    - Vocabulary: 4 (A, C, G, T)
    - Total Samples: ~3,190
    - Task: Ideally suited for debugging discrete diffusion models due to short length.
    """
    def __init__(self, root_dir='data/splice', split='train', train_ratio=0.9):
        """
        Args:
            root_dir (str): Directory to save the dataset. Defaults to 'data/splice'.
            split (str): 'train' or 'test'.
            train_ratio (float): Ratio to split training and testing data.
        """
        self.max_length = 60
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        # 1. Setup Directory
        os.makedirs(root_dir, exist_ok=True)
        self.file_path = os.path.join(root_dir, "splice.data")
        
        # Stable URL from UCI Machine Learning Repository
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/splice-junction-gene-sequences/splice.data"
        
        # 2. Download Data if not exists
        if not os.path.exists(self.file_path):
            print(f"[INFO] Dataset not found locally. Downloading to {self.file_path}...")
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    with open(self.file_path, 'w') as f:
                        f.write(response.text)
                    print("[INFO] Download and save successful.")
                else:
                    raise Exception(f"Server returned status code: {response.status_code}")
            except Exception as e:
                print(f"[ERROR] Failed to download data: {e}")
                # Create dummy data to prevent crash during debugging if network fails
                self._create_dummy_data()

        # 3. Load and Process Data
        self.sequences = []
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                lines = f.readlines()
            
            print(f"[INFO] Processing {len(lines)} raw lines...")
            
            for line in lines:
                # UCI Format: Class, Name, Sequence
                # Example: EI, ATRINS-DONOR-521, CCAGCTGCATCACAGG...
                parts = line.strip().split(',')
                if len(parts) < 3: 
                    continue
                
                raw_seq = parts[2].strip().upper()
                
                # Cleaning: The dataset contains noise characters (D, N, S, R).
                # We filter to keep only A, C, G, T.
                clean_seq = "".join([c for c in raw_seq if c in self.vocab])
                
                # Ensure fixed length of 60bp
                if len(clean_seq) == self.max_length:
                    self.sequences.append(clean_seq)
                elif len(clean_seq) > self.max_length:
                    self.sequences.append(clean_seq[:self.max_length])
        
        # 4. Train/Test Split
        # Since UCI provides a single file, we manually split it.
        total_count = len(self.sequences)
        split_idx = int(total_count * train_ratio)
        
        if split == 'train':
            self.sequences = self.sequences[:split_idx]
        else:
            self.sequences = self.sequences[split_idx:]

        print(f"[INFO] Dataset Ready ({split}). Samples: {len(self.sequences)} | Length: {self.max_length} bp")

    def _create_dummy_data(self):
        """Creates a dummy file to prevent code crashes if download fails."""
        print("[WARN] Creating dummy data for debugging.")
        with open(self.file_path, 'w') as f:
            f.write("EI, DUMMY_1, " + "A" * 60 + "\n")
            f.write("IE, DUMMY_2, " + "C" * 60 + "\n")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_str = self.sequences[idx]
        # Convert String -> List of Integers -> Tensor
        token_ids = [self.vocab[c] for c in seq_str]
        return torch.tensor(token_ids, dtype=torch.long)

# --- EXECUTION CHECK ---
if __name__ == "__main__":
    # This will save data to 'data/splice'
    dataset = SpliceJunctionDataset(root_dir='data/splice', split='train')
    
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    try:
        batch = next(iter(loader))
        print(f"\n[TEST] Batch Shape: {batch.shape}")  # Expected: [64, 60]
        print(f"[TEST] Sample Tensor (First 10 tokens): {batch[0][:10]}")
    except StopIteration:
        print("[ERROR] DataLoader is empty.")