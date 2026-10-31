"""tRNA Mammal Dataset Loader."""

import os

import datasets
import torch


def load_trna_dataset(
    data_file=None,
    split='train',
    test_size=0.05,
    seed=42):
    """Load tRNA mammal dataset from file.
    
    Args:
        data_file: Path to the tRNA sequences file.
        split: Dataset split ('train' or 'test').
        test_size: Fraction of data to use for testing.
        seed: Random seed for train/test split.
    
    Returns:
        datasets.Dataset containing tRNA sequences.
    """
    if data_file is None:
        # Path to the mammal selection file in data/trna/
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_file = os.path.join(
            base_dir,
            'data', 'trna', 'final_sequences.txt')
    
    # Read tRNA sequences from file
    with open(data_file, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    
    # Create dataset from sequences
    dataset = datasets.Dataset.from_dict({'sequence': sequences})
    
    # Split into train and test
    dataset_dict = dataset.train_test_split(
        test_size=test_size,
        seed=seed)
    
    return dataset_dict[split]


class TRNADataset(torch.utils.data.Dataset):
    """tRNA mammal dataset with on-the-fly tokenization.
    
    Similar to GrampaDataset but for tRNA sequences.
    """
    
    def __init__(
        self,
        split: str,
        tokenizer,
        max_length: int = 128,
        add_special_tokens: bool = False,
        data_file=None,
        dataset=None):
        """Initialize tRNA dataset.
        
        Args:
            split: Dataset split ('train' or 'test').
            tokenizer: Tokenizer for tRNA sequences.
            max_length: Maximum sequence length.
            add_special_tokens: Whether to add BOS/EOS tokens.
            data_file: Path to tRNA sequences file.
            dataset: Pre-loaded dataset (optional).
        """
        if dataset is None:
            self.dataset = load_trna_dataset(
                data_file=data_file,
                split=split)
        else:
            self.dataset = dataset
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Returns a tokenized tRNA sequence."""
        seq = self.dataset[idx]['sequence']
        
        # Tokenize sequence
        seq = self.tokenizer(
            seq,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=self.add_special_tokens,
            return_attention_mask=True)
        
        input_ids = torch.LongTensor(seq['input_ids'])
        attention_mask = torch.LongTensor(seq['attention_mask'])
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask}
