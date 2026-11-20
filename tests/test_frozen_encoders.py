import unittest

import numpy as np

from experiments.frozen_encoders import assign_goalkeepers, map_player_clusters, score_predictions


class FrozenEncoderMappingTest(unittest.TestCase):
    def test_cluster_side_goalkeeper_and_scoring_mapping(self):
        team_ids, centroids = map_player_clusters([1, 1, 0, 0], [8.0, 10.0, -7.0, -5.0])
        self.assertEqual(team_ids.tolist(), [1, 1, 0, 0])
        goalkeeper_ids = assign_goalkeepers([-6.0, 9.0, np.nan], centroids)
        prediction = np.asarray([*team_ids, *[2 + value if value >= 0 else -1 for value in goalkeeper_ids]])
        result = score_predictions([1, 1, 0, 0, 2, 3, 2], prediction)
        self.assertEqual(result["score"], 0.875)
        self.assertEqual(result["truth_counts"], {"team_a": 2, "team_b": 2, "goalkeeper_a": 2, "goalkeeper_b": 1})
        self.assertEqual(result["unassigned_boxes"], 1)
        self.assertEqual(result["confusion_matrix"][2], [0, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
