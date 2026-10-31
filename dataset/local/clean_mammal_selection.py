import glob
import os
from Bio import SeqIO

def process_tRNA_sequences():
    """
    Reads all '*-mature-tRNAs.fa' files in the current directory,
    extracts sequences (handling multi-line FASTA), converts T to U,
    and saves them to a single text file.
    """
    
    # Define input pattern and output filename
    input_pattern = "*-mature-tRNAs.fa"
    output_filename = "final_sequences.txt"
    
    # Find all matching files
    files = glob.glob(input_pattern)
    
    if not files:
        print(f"Error: No files matching '{input_pattern}' found in current directory.")
        return

    print(f"Found {len(files)} files. Processing...")
    
    count = 0
    with open(output_filename, "w") as f_out:
        for file_path in files:
            try:
                # SeqIO.parse automatically handles multi-line sequences
                # It reads the header (>...) and joins the sequence lines below it
                for record in SeqIO.parse(file_path, "fasta"):
                    
                    # Convert to string and uppercase
                    seq = str(record.seq).upper()
                    
                    # Convert DNA (T) to RNA (U) if necessary
                    seq = seq.replace("T", "U")
                    
                    # Write to file (one sequence per line)
                    f_out.write(seq + "\n")
                    count += 1
                    
            except Exception as e:
                print(f"Warning: Could not parse {file_path}. Error: {e}")

    print("-" * 30)
    print(f"Success! Processed {count} sequences.")
    print(f"Saved to: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    process_tRNA_sequences()