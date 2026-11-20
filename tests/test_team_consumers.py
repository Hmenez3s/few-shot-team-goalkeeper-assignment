import unittest

import prepare
import reference_bank


class TeamConsumerTests(unittest.TestCase):
    def test_assignments_preserve_or_swap_both_roles_and_reject_missing(self):
        manifest = {
            "audit": {
                "team_normalization": {"schema": prepare.TEAM_NORMALIZATION_SCHEMA},
                "team_normalization_assignments": [
                    {"split": "test", "game": "7", "half": "2", "clip": "128", "orientation": "unchanged"},
                    {"split": "test", "game": "7", "half": "2", "clip": "129", "orientation": "swapped"},
                    {"split": "train", "game": "7", "half": "2", "clip": "129", "orientation": "unchanged"},
                ],
            }
        }
        player = {"attributes": {"role": "player", "team": "left"}}
        goalkeeper = {"attributes": {"role": "goalkeeper", "team": "left"}}

        self.assertFalse(reference_bank.team_swap_for_clip(manifest, "test", "7", "2", "128"))
        self.assertTrue(reference_bank.team_swap_for_clip(manifest, "test", "7", "2", "129"))
        self.assertEqual(prepare.label_class(player, False), 0)
        self.assertEqual(prepare.label_class(player, True), 1)
        self.assertEqual(prepare.label_class(goalkeeper, False), 2)
        self.assertEqual(prepare.label_class(goalkeeper, True), 2)
        with self.assertRaisesRegex(ValueError, "no team normalization assignment"):
            reference_bank.team_swap_for_clip(manifest, "test", "7", "2", "130")
        duplicate = {"audit": {**manifest["audit"], "team_normalization_assignments":
                               manifest["audit"]["team_normalization_assignments"] + [manifest["audit"]["team_normalization_assignments"][0]]}}
        with self.assertRaisesRegex(ValueError, "duplicate team normalization assignments"):
            reference_bank.team_swap_for_clip(duplicate, "test", "7", "2", "128")
        unresolved = {"audit": {**manifest["audit"], "team_normalization_assignments": [
            {**manifest["audit"]["team_normalization_assignments"][0], "orientation": "unresolved"}
        ]}}
        with self.assertRaisesRegex(ValueError, "unresolved team normalization assignment"):
            reference_bank.team_swap_for_clip(unresolved, "test", "7", "2", "128")
        with self.assertRaisesRegex(ValueError, "rerun prepare.py"):
            reference_bank.team_swap_for_clip({"audit": {}}, "test", "7", "2", "128")


if __name__ == "__main__":
    unittest.main()
