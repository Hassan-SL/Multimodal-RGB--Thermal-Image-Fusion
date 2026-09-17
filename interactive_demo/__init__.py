"""
FYP Multimodal RGB-IR Interactive Perception & Demonstration Package.
"""

import sys
from pathlib import Path

# Automatically ensure interactive_demo directory is on sys.path
_PACKAGE_DIR = Path(__file__).resolve().parent
if str(_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_DIR))
