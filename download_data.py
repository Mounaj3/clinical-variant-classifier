"""
download_data.py
----------------
Download ClinVar variant_summary file from NCBI FTP
and decompress it into data/raw/.

Usage:
    python download_data.py
"""

import os
import gzip
import shutil
import urllib.request
from tqdm import tqdm


URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
RAW_DIR = "data/raw"
GZ_PATH = os.path.join(RAW_DIR, "variant_summary.txt.gz")
TXT_PATH = os.path.join(RAW_DIR, "variant_summary.txt")


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file():
    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(TXT_PATH):
        print("File already exists, skipping download.")
        return

    if not os.path.exists(GZ_PATH):
        print("Downloading ClinVar variant_summary from NCBI FTP...")
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="variant_summary.txt.gz") as t:
            urllib.request.urlretrieve(URL, filename=GZ_PATH, reporthook=t.update_to)
        print(f"Downloaded: {GZ_PATH}")

    print("Decompressing archive...")
    with gzip.open(GZ_PATH, "rb") as f_in:
        with open(TXT_PATH, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Saved to: {TXT_PATH}")

    # Quick sanity check
    with open(TXT_PATH, "r") as f:
        header = f.readline().strip().split("\t")
    print(f"\n{len(header)} columns detected.")
    print(f"First few: {header[:6]}")


if __name__ == "__main__":
    download_file()