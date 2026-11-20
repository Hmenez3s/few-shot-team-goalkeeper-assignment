import unittest

from experiments.prospective_evaluation import choose_support, query_frame_indices, timeline_cutoff


class ProspectivePolicyTests(unittest.TestCase):
    def test_late_labels_cannot_change_early_support_or_scored_frames(self):
        frames = [{"time": time} for time in (0, 100, 200, 300, 400, 500, 600, 700, 800, 900)]
        early = [
            {"class_id": class_id, "time": time, "clip": "early", "image_id": f"{time}-{class_id}",
             "annotation_index": 0, "quality_valid": True}
            for class_id in range(4) for time in (0, 100, 200)
        ]
        late = [
            {"class_id": class_id, "time": time, "clip": "late", "image_id": f"late-{time}-{class_id}",
             "annotation_index": 0, "quality_valid": True}
            for class_id in range(4) for time in (700, 800, 900)
        ]
        cutoff_before = timeline_cutoff(frames)
        support_before, missing_before = choose_support(early + late, cutoff_before, 1)
        query_before = query_frame_indices(frames, cutoff_before)

        altered_late = [{**row, "class_id": 3 - row["class_id"]} for row in late]
        support_after, missing_after = choose_support(early + altered_late, cutoff_before, 1)
        query_after = query_frame_indices(frames, cutoff_before)

        self.assertEqual(missing_before, [])
        self.assertEqual(missing_after, [])
        self.assertEqual(cutoff_before, 200)
        self.assertEqual(query_before, [3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(query_after, query_before)
        self.assertEqual(
            {class_id: [(row["time"], row["image_id"]) for row in rows] for class_id, rows in support_before.items()},
            {class_id: [(row["time"], row["image_id"]) for row in rows] for class_id, rows in support_after.items()},
        )


if __name__ == "__main__":
    unittest.main()
