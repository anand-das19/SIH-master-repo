"""
AgriNode AI - Environmental Risk Intelligence
Phase 1: Data Ingestion & Dataset Management

Handles automated downloading, extraction, and validation of INDmet and IMD historical weather datasets.
"""

import os
import sys
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path

ZENODO_RECORD_API = "https://zenodo.org/api/records/15430548"
RAW_INDMET_DIR = Path("data/raw/indmet")
RAW_IMD_DIR = Path("data/raw/imd")
PROCESSED_DIR = Path("data/processed")

ZENODO_FILES = {
    "India_Districts.csv": "https://zenodo.org/api/records/15430548/files/India_Districts.csv/content",
    "readme.txt": "https://zenodo.org/api/records/15430548/files/readme.txt/content",
    "INDmet_District_Data.zip": "https://zenodo.org/api/records/15430548/files/INDmet_District_Data.zip/content"
}


def download_file(url: str, dest_path: Path, expected_size: int = None) -> Path:
    """Download a file with streaming progress display."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        actual_size = dest_path.stat().st_size
        if expected_size is None or actual_size == expected_size or actual_size > 1000:
            print(f"[CACHE] {dest_path.name} already exists ({actual_size / (1024*1024):.2f} MB). Skipping download.")
            return dest_path

    print(f"[DOWNLOADING] {dest_path.name} from {url}...")
    headers = {"User-Agent": "AgriNode-AI-Ingestion/1.0"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_length = response.headers.get("content-length")
        total_size = int(total_length) if total_length else None
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size:
                percent = (downloaded / total_size) * 100
                mb_down = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent:.1f}% ({mb_down:.1f} / {mb_tot:.1f} MB)", end="", flush=True)
            else:
                mb_down = downloaded / (1024 * 1024)
                print(f"\r  Downloaded: {mb_down:.1f} MB", end="", flush=True)

    print("\n[DONE] Download complete.")
    return dest_path


def extract_zip(zip_path: Path, target_dir: Path) -> Path:
    """Extract zip archive if not already extracted."""
    target_dir.mkdir(parents=True, exist_ok=True)
    marker = target_dir / ".extracted_marker"
    if marker.exists():
        print(f"[CACHE] {zip_path.name} already extracted in {target_dir}. Skipping extraction.")
        return target_dir

    print(f"[EXTRACTING] {zip_path.name} to {target_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dir)

    with open(marker, "w") as f:
        f.write("extracted")
    print(f"[DONE] Extraction complete: {target_dir}")
    return target_dir


def download_indmet_datasets():
    """Download all INDmet dataset components from Zenodo."""
    RAW_INDMET_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Download metadata files
    districts_csv = RAW_INDMET_DIR / "India_Districts.csv"
    download_file(ZENODO_FILES["India_Districts.csv"], districts_csv)

    readme_file = RAW_INDMET_DIR / "readme.txt"
    download_file(ZENODO_FILES["readme.txt"], readme_file)

    # 2. Download district zip
    district_zip = RAW_INDMET_DIR / "INDmet_District_Data.zip"
    download_file(ZENODO_FILES["INDmet_District_Data.zip"], district_zip, expected_size=153159782)

    # 3. Extract district zip
    extract_dir = RAW_INDMET_DIR / "district_data"
    extract_zip(district_zip, extract_dir)
    return extract_dir


if __name__ == "__main__":
    download_indmet_datasets()
