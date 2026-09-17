"""

import sys
from pathlib import Path

# Add scripts directory to sys.path so path_config is always discoverable
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import path_config
from path_config import *

M3FD XML (Pascal VOC) to YOLO Label Converter (scripts_AG/convert_m3fd_xml_to_yolo.py)
Converts XML annotation files into normalized YOLO format (.txt) files required by TarDAL.
Format per line: <class_id> <cx> <cy> <w> <h>
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

M3FD_CLASS_MAP = {
    'people': 0, 'person': 0, 'human': 0,
    'car': 1,
    'bus': 2,
    'lamp': 3,
    'motorcycle': 4, 'motor': 4,
    'truck': 5
}

def convert_xml_file(xml_path: Path, output_txt_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find('size')
    if size_elem is None:
        return 0

    img_w = float(size_elem.find('width').text)
    img_h = float(size_elem.find('height').text)

    if img_w <= 0 or img_h <= 0:
        return 0

    yolo_lines = []

    for obj in root.findall('object'):
        cls_name = obj.find('name').text.strip().lower()
        if cls_name not in M3FD_CLASS_MAP:
            continue

        cls_id = M3FD_CLASS_MAP[cls_name]
        bndbox = obj.find('bndbox')
        
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        # Convert to YOLO normalized (cx, cy, w, h)
        box_w = max(0.0, xmax - xmin)
        box_h = max(0.0, ymax - ymin)
        cx = xmin + box_w / 2.0
        cy = ymin + box_h / 2.0

        norm_cx = cx / img_w
        norm_cy = cy / img_h
        norm_w = box_w / img_w
        norm_h = box_h / img_h

        # Clamp values between 0.0 and 1.0
        norm_cx = min(max(norm_cx, 0.0), 1.0)
        norm_cy = min(max(norm_cy, 0.0), 1.0)
        norm_w = min(max(norm_w, 0.0), 1.0)
        norm_h = min(max(norm_h, 0.0), 1.0)

        yolo_lines.append(f"{cls_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")

    output_txt_path.write_text("\n".join(yolo_lines) + "\n", encoding='utf-8')
    return len(yolo_lines)


def batch_convert_xml_dir(xml_dir: Path, labels_dir: Path):
    labels_dir.mkdir(parents=True, exist_ok=True)
    xml_files = list(xml_dir.glob('*.xml'))

    if not xml_files:
        print(f"[Warning] No .xml files found in {xml_dir}")
        return

    print(f"[XML Converter] Converting {len(xml_files)} XML files from {xml_dir} -> {labels_dir} ...")
    converted_count = 0
    for xml_p in tqdm(xml_files, desc="Converting XML to YOLO"):
        txt_name = xml_p.stem + ".txt"
        txt_p = labels_dir / txt_name
        converted_count += convert_xml_file(xml_p, txt_p)

    print(f"[XML Converter] Successfully generated {len(xml_files)} YOLO label files in {labels_dir}!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        xml_dir = Path(sys.argv[1])
        labels_dir = Path(sys.argv[2])
        batch_convert_xml_dir(xml_dir, labels_dir)
    else:
        print("Usage: python convert_m3fd_xml_to_yolo.py <xml_dir> <labels_dir>")