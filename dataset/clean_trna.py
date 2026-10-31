import gzip
import os
import random

def create_trna_benchmark_subset(
    input_path='data/trna/RF00005.fa.gz', 
    output_path='data/trna/tRNA_benchmark_30k.txt',
    target_size=30000,
    min_len=73,
    max_len=93
):
    print(f"[INFO] Input file: {input_path}")

    if not os.path.exists(input_path):
        print(f"[ERROR] File not found: {input_path}")
        return
    
    valid_sequences = set()
    current_seq_parts = []
    
    # ---- READ & MERGE FASTA SEQUENCES ----
    with gzip.open(input_path, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # save previous seq
                if current_seq_parts:
                    seq = "".join(current_seq_parts)
                    if min_len <= len(seq) <= max_len:
                        valid_sequences.add(seq)
                    current_seq_parts = []
            else:
                current_seq_parts.append(line)
        
        # handle last sequence
        if current_seq_parts:
            seq = "".join(current_seq_parts).upper()
            if min_len <= len(seq) <= max_len:
                valid_sequences.add(seq)

    valid_sequences = list(valid_sequences)
    print(f"[INFO] Total full-length (unique) seqs = {len(valid_sequences)}")
    
    # ---- RANDOM SAMPLE ----
    if len(valid_sequences) < target_size:
        print("[WARN] Not enough sequences to sample, saving all available.")
        subset = valid_sequences
    else:
        subset = random.sample(valid_sequences, target_size)
    
    # ---- SAVE OUTPUT ----
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as out:
        for seq in subset:
            out.write(seq + "\n")
    
    print(f"[INFO] Saved {len(subset)} sequences → {output_path}")

if __name__ == "__main__":
    create_trna_benchmark_subset()