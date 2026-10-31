"""
Custom dataset for Grampa (Peptides) with labels for classifier training.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class GrampaClassifierDataset(Dataset):
    """
    Dataset for Grampa peptides with labels (0-3) for classifier training.
    
    Labels indicate how many properties are satisfied:
    - Label 0: No properties satisfied
    - Label 1: 1 property satisfied
    - Label 2: 2 properties satisfied  
    - Label 3: All 3 properties satisfied
    """
    
    def __init__(self, split='train', tokenizer=None, max_length=256, 
                 add_special_tokens=True, data_path='data/peptides/labels/peptide_length_hydrophobic_ratio_net_charge/lb[10.0_0.3_2.0]/ub[50.0_1000.0_9.0]/data.csv',
                 val_split=0.1, seed=42):
        """
        Args:
            split: 'train' or 'test' (validation)
            tokenizer: Tokenizer for encoding sequences
            max_length: Maximum sequence length
            add_special_tokens: Whether to add special tokens (BOS, EOS)
            data_path: Path to CSV file with peptides and labels (relative to repo root)
            val_split: Validation split ratio
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Convert relative path to absolute path based on repo root
        if not os.path.isabs(data_path):
            # Get the repo root (parent of custom_datasets/)
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_path = os.path.join(repo_root, data_path)
        
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        
        # Load data
        df = pd.read_csv(data_path)
        
        # Split into train/val
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df,
            test_size=val_split,
            random_state=seed,
            stratify=df['label']
        )
        
        if split == 'train':
            self.data = train_df.reset_index(drop=True)
        else:
            self.data = val_df.reset_index(drop=True)
        
        print(f"Loaded {len(self.data)} peptides for {split} split")
        print(f"Label distribution: {self.data['label'].value_counts().sort_index().to_dict()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['sequence']
        label = int(row['label'])
        
        # Tokenize
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                sequence,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                add_special_tokens=self.add_special_tokens,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)
            }
        else:
            # Return raw sequence if no tokenizer
            return {
                'sequence': sequence,
                'label': torch.tensor(label, dtype=torch.long)
            }
