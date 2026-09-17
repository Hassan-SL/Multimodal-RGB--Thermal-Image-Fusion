"""
02_run_tardal_inference.py (Supports dt, ct, tt checkpoints)
Runs TarDAL inference dynamically using portable paths and auto-device selection.
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Import portable path resolver

import sys
from pathlib import Path

# Add scripts directory to sys.path so path_config is always discoverable
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import path_config
from path_config import *



def resolve_model_config(model_name: str) -> tuple[Path, Path]:
    """Resolves model config and output directory based on model name alias (dt, ct, tt)."""
    model_key = model_name.lower().replace('.pt', '').replace('.pth', '').strip()
    if model_key in ['dt', 'tardal-dt', 'tardal_dt', 'tardal']:
        cfg_name = 'tardal-dt.yaml'
        run_folder = 'tardal-dt'
    elif model_key in ['ct', 'tardal-ct', 'tardal_ct', 'tardal+']:
        cfg_name = 'tardal-ct.yaml'
        run_folder = 'tardal-ct'
    else:  # default to tt
        cfg_name = 'tardal-tt.yaml'
        run_folder = 'tardal-tt'

    cfg_path = TARDAL_ROOT / "config" / "official" / "infer" / cfg_name
    if not cfg_path.exists():
        cfg_path = TARDAL_ROOT / "config" / "official" / "infer" / "tardal-tt.yaml"

    output_dir = RUNS_DIR / "inference" / run_folder
    return cfg_path, output_dir


def run_inference(model: str = "dt", output_dir_override: Path | None = None):
    print(f"=== TarDAL Inference Runner (Model: {model}) ===")
    
    if not TARDAL_ROOT.exists():
        print(f"[Error] TarDAL repository not found at {TARDAL_ROOT}")
        sys.exit(1)

    cfg_path, default_output_dir = resolve_model_config(model)
    output_dir = output_dir_override if output_dir_override else default_output_dir

    if not cfg_path.exists():
        print(f"[Error] Config file not found at {cfg_path}")
        sys.exit(1)

    python_exe = sys.executable
    infer_script = TARDAL_ROOT / "infer.py"

    cmd = [
        python_exe,
        str(infer_script),
        "--cfg", str(cfg_path),
        "--save_dir", str(output_dir)
    ]

    print(f"[Run] Executing: {' '.join(cmd)}")
    print(f"[Run] Output location: {output_dir}\n")

    res = subprocess.run(cmd, cwd=str(TARDAL_ROOT))
    if res.returncode == 0:
        print(f"\n[Success] TarDAL Inference completed successfully!")
        print(f"[Output] Results saved to: {output_dir}")
    else:
        print(f"\n[Error] Inference finished with exit code {res.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run TarDAL Inference for dt, ct, or tt checkpoints")
    parser.add_argument("--model", "--weights", type=str, default="dt",
                        help="Model checkpoint alias: 'dt' (Direct Training), 'ct' (Cooperative Training), 'tt' (Task-oriented Training)")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional custom output directory")
    args = parser.parse_args()

    out_p = Path(args.output_dir) if args.output_dir else None
    run_inference(model=args.model, output_dir_override=out_p)


if __name__ == "__main__":
    main()