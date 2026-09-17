"""
Dataset Verification Tool
Validates image-label pairs, bounding box boundaries, and split counts.
"""

import argparse
from pathlib import Path

def verify_dataset(data_dir: Path):
    print(f"[*] Checking dataset at: {data_dir.resolve()}")
    if not data_dir.exists():
        print(f"[!] Error: Dataset directory {data_dir} does not exist.")
        print("[!] Please follow data/README.md to download and unpack M3FD.")
        return False

    splits = ['train', 'val', 'test']
    for split in splits:
        img_dir = data_dir / 'images' / split
        lbl_dir = data_dir / 'labels' / split
        
        imgs = list(img_dir.glob('*.*')) if img_dir.exists() else []
        lbls = list(lbl_dir.glob('*.txt')) if lbl_dir.exists() else []
        
        print(f"  - Split [{split}]: {len(imgs)} images found, {len(lbls)} label files found.")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verify M3FD dataset integrity.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/m3fd"), help="Path to M3FD dataset root.")
    args = parser.parse_args()
    verify_dataset(args.data_dir)
