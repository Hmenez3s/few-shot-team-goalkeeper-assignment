import unittest

import numpy as np

import prtreid_comparison


class TeamMappingTest(unittest.TestCase):
    def test_track_votes_cluster_sides_and_oracle_permutation(self):
        rows = [
            {"clip": "c", "track_id": "a", "role": "player", "pitch_x": -4.0, "class": 0, "game": "7", "session": "1"},
            {"clip": "c", "track_id": "a", "role": "player", "pitch_x": -3.0, "class": 0, "game": "7", "session": "1"},
            {"clip": "c", "track_id": "b", "role": "player", "pitch_x": 5.0, "class": 1, "game": "7", "session": "1"},
            {"clip": "c", "track_id": "b", "role": "player", "pitch_x": 4.0, "class": 1, "game": "7", "session": "1"},
            {"clip": "c", "track_id": "g", "role": "goalkeeper", "pitch_x": -2.0, "class": 2, "game": "7", "session": "1"},
        ]
        vectors = np.array([[0, 0], [0, 0], [9, 9], [9, 9], [0, 0]], dtype=float)
        roles = np.array([3, 3, 3, 3, 1])
        v1, roles_v1 = prtreid_comparison.cluster_tracks(rows, vectors, roles)
        v2, _ = prtreid_comparison.cluster_tracks(rows, vectors, roles, ground_truth=True)
        v3, _ = prtreid_comparison.cluster_tracks(rows, vectors, roles, ground_truth=True, oracle=True)
        self.assertEqual(v1[0], 0)
        self.assertEqual(roles_v1[2], "player")
        self.assertEqual(v2[0], 0)
        self.assertEqual(v2[2], 1)
        self.assertEqual(v2[4], 0)
        self.assertEqual(v3, v2)


if __name__ == "__main__":
    unittest.main()
