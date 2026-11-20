import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "gpu_formats.py"
SPEC = importlib.util.spec_from_file_location("gpu_formats", SCRIPT)
GPU_FORMATS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GPU_FORMATS)


class ScoringMetricsTest(unittest.TestCase):
    def test_computes_accuracy_macro_recall_and_confusion_matrix(self):
        metrics = GPU_FORMATS.scoring_metrics(
            [0, 0, 1, 1, 2, 3],
            [0, 1, 1, 1, 2, 0],
        )

        self.assertEqual(metrics["confusion_matrix"], [
            [1, 1, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
        ])
        self.assertEqual(metrics["accuracy"], 4 / 6)
        self.assertEqual(metrics["per_class"], [0.5, 1.0, 1.0, 0.0])
        self.assertEqual(metrics["score"], 0.625)

    def test_empty_class_has_zero_recall(self):
        metrics = GPU_FORMATS.scoring_metrics(np.array([0]), np.array([0]))

        self.assertEqual(metrics["per_class"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(metrics["score"], 0.25)

    def test_rejects_different_vector_shapes(self):
        with self.assertRaises(ValueError):
            GPU_FORMATS.scoring_metrics([0], [0, 1])


if __name__ == "__main__":
    unittest.main()
