"""Tokenizer for tRNA sequences."""

import json
import os
import typing

import torch
import transformers


class TRNATokenizer(transformers.PreTrainedTokenizer):
    """Tokenizer for tRNA sequences.
    
    Uses a vocabulary of 4 nucleotides (A, C, G, U) with special tokens.
    """
    
    def __init__(
        self,
        vocab_file=None,
        bos_token='<bos>',
        eos_token='<eos>',
        sep_token='<sep>',
        cls_token='<cls>',
        pad_token='<pad>',
        mask_token='<mask>',
        unk_token='<unk>',
        **kwargs):
        
        # Load vocabulary from file
        if vocab_file is None:
            # Path to trna_vocab.txt in custom_datasets/
            base_dir = os.path.dirname(__file__)
            vocab_file = os.path.join(base_dir, 'trna_vocab.txt')
        
        with open(vocab_file, 'r') as f:
            nucleotide_vocab = json.load(f)
        
        # Create full vocabulary with special tokens
        self._vocab_str_to_int = {
            '<cls>': 0,
            '<sep>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<mask>': 4,
            '<pad>': 5,
            '<reserved>': 6,
            '<unk>': 7,
        }
        
        # Add nucleotides to vocabulary (offset by 8 for special tokens)
        for nuc, idx in nucleotide_vocab.items():
            self._vocab_str_to_int[nuc] = idx + 8
        
        self._vocab_int_to_str = {
            v: k for k, v in self._vocab_str_to_int.items()}
        
        # Cache frequently used IDs for faster lookup
        self._unk_token_id = self._vocab_str_to_int['<unk>']
        
        # Cache special tokens for faster tokenization
        # Sort by length (descending) to match longer tokens first
        self._special_tokens_list = sorted(
            ['<cls>', '<sep>', '<bos>', '<eos>', '<mask>', '<pad>', '<reserved>', '<unk>'],
            key=len,
            reverse=True
        )
        
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs)
    
    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)
    
    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        """Tokenize a tRNA sequence into individual nucleotides.
        
        Handles special tokens properly to avoid splitting them into characters.
        """
        if not text:
            return []
        
        tokens = []
        i = 0
        text_len = len(text)
        
        while i < text_len:
            # Check for special tokens (already sorted by length descending)
            matched = False
            if text[i] == '<':  # Quick check before trying special tokens
                for special_token in self._special_tokens_list:
                    token_len = len(special_token)
                    if i + token_len <= text_len and text[i:i+token_len] == special_token:
                        tokens.append(special_token)
                        i += token_len
                        matched = True
                        break
            
            if not matched:
                # Regular nucleotide - convert to uppercase
                tokens.append(text[i].upper())
                i += 1
        
        return tokens
    
    def _convert_token_to_id(self, token: str) -> int:
        """Convert a nucleotide token to its ID."""
        return self._vocab_str_to_int.get(token, self._unk_token_id)
    
    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its nucleotide token."""
        return self._vocab_int_to_str.get(index, '<unk>')
    
    def convert_tokens_to_string(self, tokens):
        """Convert a list of tokens back to a string."""
        return ''.join(tokens)
    
    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int
    
    def encode(self, text):
        """Convert a tRNA sequence into token IDs."""
        tokens = self._tokenize(text)
        return [self._convert_token_to_id(token) for token in tokens]

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        """Convert token IDs back into a tRNA sequence.
        
        Args:
            token_ids: List or tensor of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in output
            **kwargs: Additional arguments (for compatibility)
        
        Returns:
            Decoded string sequence
        """
        # Convert tensor to list if needed
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Define special token IDs to skip if requested
        special_token_ids = {
            self._vocab_str_to_int['<cls>'],
            self._vocab_str_to_int['<sep>'],
            self._vocab_str_to_int['<bos>'],
            self._vocab_str_to_int['<eos>'],
            self._vocab_str_to_int['<mask>'],
            self._vocab_str_to_int['<pad>'],
            self._vocab_str_to_int['<reserved>'],
            self._vocab_str_to_int['<unk>'],
        }
        
        # Filter out special tokens if requested
        if skip_special_tokens:
            token_ids = [tid for tid in token_ids if tid not in special_token_ids]
        
        return ''.join([self._convert_id_to_token(token_id) for token_id in token_ids])
    
    def batch_decode(self, batch_token_ids, skip_special_tokens=False, **kwargs):
        """Batch decode token IDs back into tRNA sequences.
        
        Args:
            batch_token_ids: Tensor or list of lists of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            **kwargs: Additional arguments (for compatibility)
        
        Returns:
            List of decoded string sequences
        """
        # batch_token_ids: torch.Tensor or list of lists
        if isinstance(batch_token_ids, torch.Tensor):
            batch_token_ids = batch_token_ids.tolist()
        return [self.decode(token_ids, skip_special_tokens=skip_special_tokens) 
                for token_ids in batch_token_ids]
