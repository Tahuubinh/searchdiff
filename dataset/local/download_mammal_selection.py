import os
import urllib.request
# https://gtrnadb.org/browse.html#Mammalia_atod

urls = [
"https://gtrnadb.org/genomes/eukaryota/Hsapi38/hg38-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Ptrog6/panTro6-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Mmula8/rheMac8-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Ppygm3/ponAbe3-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Sscro11/susScr11-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Ecaba2/equCab2-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Oarie4/oviAri4-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Fcatu9/felCat9-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Cfami4/canFam4-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Ocuni2/oryCun2-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Cporc3/cavPor3-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Lafri3/loxAfr3-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Mluci2/myoLuc2-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Ttrun2/turTru2-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Bacut1/balAcu1-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Mdome5.1/monDom5.1-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Sharr1/sarHar1-tRNAs.tar.gz",
"https://gtrnadb.org/genomes/eukaryota/Oanat2/ornAna2-tRNAs.tar.gz"
]

def download(url):
    filename = url.split("/")[-1]
    print(f"Downloading {filename} ...")

    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Saved: {filename}")
    except Exception as e:
        print(f"✗ Failed to download {url}")
        print(e)

for url in urls:
    download(url)

print("\nDone.")
