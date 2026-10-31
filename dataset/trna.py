import os
import gzip
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO  # Requires: pip install biopython

class TRNADataset(Dataset):
    """
    PyTorch Dataset for tRNA sequences (Rfam ID: RF00005).
    
    Attributes:
        - Source: Rfam Database (RF00005.fa.gz)
        - Vocabulary: 4 (A, C, G, U) + 1 Pad token
        - Sequence Length: Filtered to 70-80 bp (Standard tRNA length)
        - Data Size: ~60k - 80k unique sequences after filtering
    """
    def __init__(self, root_dir='data/tRNA', min_len=70, max_len=80, pad_token_id=4):
        """
        Args:
            root_dir (str): Directory to save the dataset.
            min_len (int): Minimum sequence length to keep.
            max_len (int): Maximum sequence length to keep (and pad to).
            pad_token_id (int): Integer ID used for padding sequences shorter than max_len.
        """
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 3} # Map DNA 'T' to RNA 'U'
        self.inv_vocab = {0: 'A', 1: 'C', 2: 'G', 3: 'U', 4: '<PAD>'}
        
        self.min_len = min_len
        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.sequences = []
        
        # 1. Setup Directory
        os.makedirs(root_dir, exist_ok=True)
        
        # URL for the specific tRNA family (RF00005)
        url = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/fasta/RF00005.fa.gz"
        gz_path = os.path.join(root_dir, "RF00005.fa.gz")
        
        # 2. Download Data
        if not os.path.exists(gz_path):
            print(f"[INFO] Downloading tRNA dataset from Rfam: {url}")
            try:
                urllib.request.urlretrieve(url, gz_path)
                print("[INFO] Download successful.")
            except Exception as e:
                raise RuntimeError(f"[ERROR] Failed to download data: {e}")

        # 3. Process and Filter Data
        print(f"[INFO] Processing and filtering sequences (Length: {min_len}-{max_len})...")
        
        unique_seqs = set()
        
        try:
            with gzip.open(gz_path, "rt") as handle:
                for record in SeqIO.parse(handle, "fasta"):
                    # Convert to upper case and normalize T -> U (if DNA format)
                    seq = str(record.seq).upper().replace('T', 'U')
                    
                    # FILTER 1: Length Constraint
                    if self.min_len <= len(seq) <= self.max_len:
                        # FILTER 2: Vocabulary Constraint (Remove sequences with 'N', 'R', etc.)
                        if all(c in self.vocab for c in seq):
                            unique_seqs.add(seq)
                            
        except Exception as e:
            print(f"[ERROR] Error parsing FASTA file: {e}")

        self.sequences = list(unique_seqs)
        
        print(f"[INFO] Dataset Ready.")
        print(f"      - Total Unique Samples: {len(self.sequences)}")
        print(f"      - Sequence Length Range: {min_len} - {max_len} bp")
        print(f"      - Vocabulary: {{'A': 0, 'C': 1, 'G': 2, 'U': 3, '<PAD>': {pad_token_id}}}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Tokenize
        tokens = [self.vocab[c] for c in seq]
        
        # Padding
        # Pad with `pad_token_id` to ensure all tensors have length `max_len`
        pad_len = self.max_len - len(tokens)
        if pad_len > 0:
            tokens += [self.pad_token_id] * pad_len
            
        return torch.tensor(tokens, dtype=torch.long)

# --- EXECUTION CHECK ---
if __name__ == "__main__":
    # Initialize Dataset
    dataset = TRNADataset()
    
    # Create DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # Fetch one batch
    batch = next(iter(loader))
    
    print(f"\n[TEST] Batch Shape: {batch.shape}") # Expected: [64, 80]
    print(f"[TEST] Sample Tensor (First 10 tokens): {batch[0][:10]}")
    
    # Check for padding in the batch
    # If any value is 4, it means padding was applied
    if (batch == 4).any():
        print("[TEST] Padding detected (Token ID 4 present).")
    else:
        print("[TEST] No padding detected in this specific batch (all sequences were max_len).")