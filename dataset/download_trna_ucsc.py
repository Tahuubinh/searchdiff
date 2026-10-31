import requests
import time

def download_tRNA_ucsc(genome_db, filename):
    """
    Download tRNA from UCSC Table Browser.
    genome_db: 'hg38' (Human), 'mm10' (Mouse), 'dm6' (Fruitfly)
    """
    url = "https://genome.ucsc.edu/cgi-bin/hgTables"
    
    params = {
        'clade': 'mammal',       # Or 'insect' for dm6
        'db': genome_db,
        'hgta_group': 'genes',
        'hgta_track': 'tRNAs',   # This is the GtRNAdb track
        'hgta_table': 'tRNAs',
        'hgta_outputType': 'sequence', # Get sequence
        'hgta_doTopSubmit': 'get output',
        'hgSeq.casing': 'upper', # Uppercase
    }
    
    print(f"Downloading data for {genome_db}...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        # UCSC returns FASTA file with long headers, save it for later processing
        with open(filename, 'w') as f:
            f.write(response.text)
        print(f"-> Saved to {filename}")
    else:
        print(f"-> Error downloading {genome_db}: {response.status_code}")

# Download the 3 largest datasets
download_tRNA_ucsc('hg38', 'human_tRNA.fasta')
download_tRNA_ucsc('mm10', 'mouse_tRNA.fasta')
# download_tRNA_ucsc('dm6', 'fruitfly_tRNA.fasta') # Need to change clade='insect'