import unittest

import torch

from experiments.sparse24 import build_2_4_mask


class Sparse24MaskTests(unittest.TestCase):
    def test_keeps_two_largest_magnitudes_in_each_input_group(self):
        weight = torch.tensor(
            [[[[1.0]], [[-4.0]], [[2.0]], [[3.0]], [[-8.0]], [[6.0]], [[5.0]], [[1.0]]]]
        )

        mask = build_2_4_mask(weight)

        self.assertEqual(mask.flatten().tolist(), [False, True, False, True, True, True, False, False])
        self.assertEqual(int(mask.sum()), 4)


if __name__ == "__main__":
    unittest.main()
