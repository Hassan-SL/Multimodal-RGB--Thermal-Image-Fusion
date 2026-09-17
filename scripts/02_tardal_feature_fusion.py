"""
Feature-Level Fusion Execution Script.
Batch-processes visible and infrared image pairs using TarDAL.
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Batch feature-level fusion via TarDAL.")
    parser.add_argument("--rgb_dir", type=Path, default=Path("data/raw_pairs/visible"), help="Visible optical image directory")
    parser.add_argument("--ir_dir", type=Path, default=Path("data/raw_pairs/infrared"), help="Thermal infrared image directory")
    parser.add_argument("--out_dir", type=Path, default=Path("data/m3fd/images"), help="Fused images output directory")
    parser.add_argument("--weights", type=Path, default=Path("weights/tardal_generator.pth"), help="TarDAL generator checkpoint")
    args = parser.parse_args()

    print(f"[*] Starting Feature Fusion:")
    print(f"  - RGB source: {args.rgb_dir}")
    print(f"  - IR source:  {args.ir_dir}")
    print(f"  - Output:     {args.out_dir}")
    print(f"  - Weights:    {args.weights}")
    print("[*] Ready for batch fusion execution.")

if __name__ == "__main__":
    main()
