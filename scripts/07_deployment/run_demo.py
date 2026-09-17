"""
Streamlit Interactive MultiSensorFusion Demonstrator Launcher
"""

import subprocess
import sys
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent.parent
    app_path = root / "interactive_demo" / "app.py"
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=false",
        "--server.enableCORS=false",
        "--server.enableXsrfProtection=false"
    ]
    print(f"[*] Launching MultiSensorFusion Interactive Demonstrator...")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
