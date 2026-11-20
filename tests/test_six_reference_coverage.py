import unittest

import numpy as np

import six_references


class SixReferenceCoverageTests(unittest.TestCase):
    def test_resolves_only_failed_anchor_without_changing_queries(self):
        clip = {
            "x": np.zeros((9, 96, 48, 3), dtype=np.uint8),
            "y": np.asarray([0, 0, 0, 1, 2, 3, 0, 1, 2]),
            "t": np.asarray([10, 20, 100, 10, 10, 10, 100, 100, 100]),
            "track": np.asarray([1, 1, 2, 3, 4, 5, 2, 6, 4]),
            "early": np.asarray([True, True, False, True, True, True, False, False, False]),
        }
        bank_times = np.asarray([10, 30, 40, 50, 60, *([30, 40, 50, 60, 70] * 3)])
        bank_labels = np.repeat(np.arange(4), 5)
        bank = {
            "t": bank_times,
            "y": bank_labels,
            "track": np.repeat([1, 3, 4, 5], 5),
            "desc": np.tile(np.asarray([[np.nan, np.nan, 0.5, 0.0, 0.5]], dtype=np.float32), (len(bank_times), 1)),
        }
        original = ([0, 3, 4, 5], [2, 6, 7, 8])

        resolved = six_references.resolve_episode(clip, bank, original, "synthetic-session")

        self.assertEqual(resolved[0], [1, 3, 4, 5])
        self.assertEqual(resolved[1], original[1])
        self.assertEqual(original[0], [0, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
