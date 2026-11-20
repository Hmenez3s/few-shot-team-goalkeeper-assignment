import unittest

from experiments.analyze_errors import box_edges, maximum_overlap


class GeometryTests(unittest.TestCase):
    def test_overlap_uses_intersection_over_query_area_and_clips_to_frame(self):
        query = box_edges({"x_center": 5, "y_center": 5, "w": 6, "h": 6}, 10, 10)
        self.assertEqual(query, (2.0, 2.0, 8.0, 8.0))
        self.assertAlmostEqual(maximum_overlap(query, [(5.0, 2.0, 9.0, 8.0)]), 0.5)
        clipped = box_edges({"x_center": 1, "y_center": 5, "w": 4, "h": 4}, 10, 10)
        self.assertEqual(clipped, (0.0, 3.0, 3.0, 7.0))
        self.assertEqual(maximum_overlap(clipped, []), 0.0)


if __name__ == "__main__":
    unittest.main()
