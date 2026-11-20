import unittest

import prepare


def entry(clip, left, right, goalkeeper_team="left"):
    annotations = [
        {"attributes": {"role": "player", "team": "left", "jersey": str(number)}}
        for number in left
    ]
    annotations.extend(
        {"attributes": {"role": "player", "team": "right", "jersey": str(number)}}
        for number in right
    )
    annotations.append({"attributes": {"role": "goalkeeper", "team": goalkeeper_team}})
    return {
        "split": "train",
        "game": "game-1",
        "half": "1",
        "clip": clip,
        "member": f"{clip}/Labels-GameState.json",
        "archive": "synthetic.zip",
        "record": {"annotations": annotations},
    }


def audit():
    return {"team_normalization_assignments": [], "team_normalization_errors": []}


class TeamNormalizationTests(unittest.TestCase):
    def test_camera_swap_normalizes_player_and_goalkeeper_slots(self):
        clips = [
            entry("clip-a", [1, 2], [3, 4]),
            entry("clip-b", [3, 4], [1, 2]),
            entry("clip-c", [1, 2], [3, 4]),
        ]
        source_team = clips[1]["record"]["annotations"][-1]["attributes"]["team"]
        grouped = {("train", "game-1", "1"): clips}
        report = audit()

        failures = prepare.normalize_team_sides(grouped, report)

        self.assertEqual(failures, [])
        self.assertFalse(clips[0]["swap_teams"])
        self.assertTrue(clips[1]["swap_teams"])
        self.assertFalse(clips[2]["swap_teams"])
        goalie = clips[1]["record"]["annotations"][-1]
        self.assertEqual(prepare.label_class(goalie, clips[1]["swap_teams"]), prepare.CLASS_INDEX["goalkeeper_right"])
        self.assertEqual(goalie["attributes"]["team"], source_team)
        self.assertEqual(len(report["team_normalization_assignments"]), 3)
        self.assertEqual(report["team_normalization_assignments"][1]["orientation"], "swapped")

    def test_unshared_jerseys_reject_ambiguous_orientation(self):
        clips = [entry("clip-a", [1], [2]), entry("clip-b", [3], [4])]
        grouped = {("train", "game-1", "1"): clips}
        report = audit()

        failures = prepare.normalize_team_sides(grouped, report)

        self.assertEqual(failures[0]["reason"], "ambiguous_optimum")
        self.assertTrue(all(item["orientation"] == "unresolved" for item in report["team_normalization_assignments"]))
        self.assertEqual(report["team_normalization_assignments"][0]["support"][0]["score"], 0)


if __name__ == "__main__":
    unittest.main()
