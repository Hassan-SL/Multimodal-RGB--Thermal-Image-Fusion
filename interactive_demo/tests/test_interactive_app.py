"""
Automated Smoke Test Suite for FYP Interactive Multimodal Perception Engine.
Verifies all 5 pipelines, preprocessing, model loading, WBF, and sensor dropout.
Can be executed completely headless: python tests/test_interactive_app.py
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch
import yaml

# Set up paths
APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR.parent

for p in [APP_DIR, PROJECT_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ROOT = APP_DIR

from models.loader import get_model_manager
from preprocessing.tardal_preprocessor import TarDALPreprocessor
from preprocessing.image_loader import load_and_align_pair
from pipelines.rgb_pipeline import RGBPipeline
from pipelines.ir_pipeline import IRPipeline
from pipelines.stage3_pipeline import Stage3Pipeline
from pipelines.stage5_pipeline import Stage5Pipeline
from pipelines.stage6_pipeline import Stage6Pipeline


class TestInteractiveApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manager = get_model_manager(ROOT)
        cls.config_path = ROOT / "config" / "app_config.yaml"
        # Load sample pair
        sample_rgb = ROOT / "assets" / "sample_pairs" / "rgb_00000.png"
        sample_ir = ROOT / "assets" / "sample_pairs" / "ir_00000.png"
        if sample_rgb.exists() and sample_ir.exists():
            cls.rgb_img, cls.ir_img, err = load_and_align_pair(sample_rgb, sample_ir)
            assert err is None
        else:
            # Fallback to synthetic if sample not found
            cls.rgb_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            cls.ir_img = np.random.randint(0, 255, (640, 640), dtype=np.uint8)

    def test_01_config_loading(self):
        """Verifies app_config.yaml loads and contains necessary keys."""
        self.assertTrue(self.config_path.exists())
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertIn("models", cfg)
        self.assertIn("late_fusion", cfg)
        self.assertIn("protocols", cfg)
        self.assertIn("dataset", cfg)

    def test_02_checkpoint_resolution(self):
        """Verifies that all required model checkpoints exist and resolve."""
        ckpts = [
            "checkpoints/yolov5su_rgb_best.pt",
            "checkpoints/yolov5su_ir_best.pt",
            "checkpoints/best.pt",
            "checkpoints/stage3_gen_best.pt",
            "checkpoints/stage6/stage6_yolo11s_best.pt"
        ]
        for c in ckpts:
            resolved = self.manager.resolve_checkpoint(c)
            self.assertIsNotNone(resolved, f"Checkpoint failed to resolve: {c}")
            self.assertTrue(resolved.exists())

    def test_03_tardal_preprocessing_and_reconstruction(self):
        """Tests mathematical YCrCb split, Tanh remapping, and reconstruction."""
        prep = TarDALPreprocessor(target_size=640)
        ir_t, vis_t, meta = prep.prepare_tensors(self.rgb_img, self.ir_img, self.manager.device)

        self.assertEqual(ir_t.shape, (1, 1, 640, 640))
        self.assertEqual(vis_t.shape, (1, 1, 640, 640))
        self.assertTrue(0.0 <= ir_t.min() and ir_t.max() <= 1.0)
        self.assertTrue(0.0 <= vis_t.min() and vis_t.max() <= 1.0)

        # Mock generator output in [-1, 1]
        mock_fused = torch.tanh(torch.randn(1, 1, 640, 640, device=self.manager.device))
        fused_rgb = prep.reconstruct_fused_image(mock_fused, meta)
        self.assertEqual(fused_rgb.shape, (meta["orig_h"], meta["orig_w"], 3))
        self.assertEqual(fused_rgb.dtype, np.uint8)

    def test_04_mode_a_rgb_pipeline(self):
        """Tests Mode A Direct RGB inference."""
        pipeline = RGBPipeline()
        res = pipeline.run(self.rgb_img, conf=0.25, iou=0.50)
        self.assertIsNotNone(res.annotated_image)
        self.assertIn("detector_ms", res.timing)
        self.assertIn("total_e2e_ms", res.timing)
        self.assertTrue(res.timing["fps"] > 0)

    def test_05_mode_b_ir_pipeline(self):
        """Tests Mode B Direct IR inference."""
        pipeline = IRPipeline()
        res = pipeline.run(ir_img=self.ir_img, conf=0.25, iou=0.50)
        self.assertIsNotNone(res.annotated_image)
        self.assertIn("detector_ms", res.timing)
        self.assertTrue(res.timing["fps"] > 0)

    def test_06_mode_c_stage3_pipeline(self):
        """Tests Mode C TarDAL + YOLOv5su feature fusion."""
        pipeline = Stage3Pipeline()
        res = pipeline.run(self.rgb_img, self.ir_img, conf=0.25, iou=0.50)
        self.assertIsNotNone(res.annotated_image)
        self.assertIsNotNone(res.fused_image)
        self.assertIn("generator_ms", res.timing)
        self.assertIn("reconstruction_ms", res.timing)
        self.assertIn("detector_ms", res.timing)

    def test_07_mode_d_stage6_pipeline(self):
        """Tests Mode D TarDAL + YOLO11s modern detector feature fusion."""
        pipeline = Stage6Pipeline()
        res = pipeline.run(self.rgb_img, self.ir_img, conf=0.25, iou=0.50)
        self.assertIsNotNone(res.annotated_image)
        self.assertIsNotNone(res.fused_image)
        self.assertIn("detector_ms", res.timing)

    def test_08_mode_e_stage5_wbf_pipeline(self):
        """Tests Mode E Decision-Level Late Fusion (both sensors online)."""
        pipeline = Stage5Pipeline()
        res = pipeline.run(self.rgb_img, self.ir_img, conf=0.25, iou=0.50, w_rgb=0.6, w_ir=0.4)
        self.assertIsNotNone(res.annotated_image)
        self.assertIn("wbf_merge_ms", res.timing)
        self.assertIn("rgb_detections", res.intermediate_data)
        self.assertIn("ir_detections", res.intermediate_data)

    def test_09_mode_e_sensor_dropout(self):
        """Tests sensor failure handling in Stage 5 (Graceful Degradation)."""
        pipeline = Stage5Pipeline()
        # Simulate IR Failure
        res_no_ir = pipeline.run(self.rgb_img, self.ir_img, conf=0.25, rgb_active=True, ir_active=False)
        self.assertIn("IR OFFLINE", res_no_ir.intermediate_data["status_message"])
        self.assertIsNotNone(res_no_ir.annotated_image)

        # Simulate RGB Failure
        res_no_rgb = pipeline.run(self.rgb_img, self.ir_img, conf=0.25, rgb_active=False, ir_active=True)
        self.assertIn("RGB OFFLINE", res_no_rgb.intermediate_data["status_message"])
        self.assertIsNotNone(res_no_rgb.annotated_image)


if __name__ == "__main__":
    unittest.main()
