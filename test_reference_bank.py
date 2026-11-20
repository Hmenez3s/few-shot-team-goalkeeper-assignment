import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import reference_bank
import six_references


class ReferenceBankTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "work" / "clips" / "prepared.npz"
        self.prepared.parent.mkdir(parents=True)
        np.savez(self.prepared, marker=np.asarray([1], dtype=np.int64))
        self.entry = {"split": "train", "game": "7", "clip": "game-7-half-1",
                      "path": "work/clips/prepared.npz"}
        self.manifest = {
            "classes": ["player_left", "player_right", "goalkeeper_left", "goalkeeper_right"],
            "clips": [{**self.entry}],
            "audit": {
                "team_normalization": {"schema": 1, "protocol": " Per-half   jersey agreement "},
                "units": [{"split": "train", "game": "7", "key": self.entry["clip"], "cutoff_ms": 100}],
            },
        }
        self.path = self.root / "work" / "references" / f"{self.entry['clip']}.npz"
        self.path.parent.mkdir(parents=True)
        self.write_bank(np.asarray([99], dtype=np.int64))

    def tearDown(self):
        self.temporary.cleanup()

    def write_bank(self, times, identity=None, include_identity=True):
        identity = identity or reference_bank.bank_identity(self.root, self.manifest, self.entry, self.path)
        arrays = {
            "x": np.zeros((len(times), 96, 48, 3), dtype=np.uint8),
            "t": times,
            "y": np.zeros(len(times), dtype=np.int64),
            "track": np.zeros(len(times), dtype=np.int64),
            "desc": np.tile(np.asarray([[np.nan, np.nan, 0.5, 0.0, 0.5]], dtype=np.float32), (len(times), 1)),
        }
        if include_identity:
            arrays.update({
                "bank_schema": np.asarray(identity["bank_schema"], dtype=np.int64),
                "protocol_id": np.asarray(identity["protocol_id"], dtype="U64"),
                "prepared_sha256": np.asarray(identity["prepared_sha256"], dtype="U64"),
                "cutoff_ms": np.asarray(identity["cutoff_ms"], dtype=np.int64),
            })
        np.savez(self.path, **arrays)

    def test_identity_bound_bank_loads_with_optional_pitch(self):
        bank = six_references.load_bank(self.path, self.root, self.manifest, self.entry)
        self.assertTrue(np.isnan(bank["desc"][0, :2]).all())

    def test_bank_rejects_stale_timestamps_and_missing_identity(self):
        self.write_bank(np.asarray([100], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "unit cutoff"):
            six_references.load_bank(self.path, self.root, self.manifest, self.entry)
        self.write_bank(np.asarray([99], dtype=np.int64), include_identity=False)
        with self.assertRaisesRegex(ValueError, "missing bank identity"):
            six_references.load_bank(self.path, self.root, self.manifest, self.entry)

    def test_bank_rejects_cutoff_identity_mismatch(self):
        identity = reference_bank.bank_identity(self.root, self.manifest, self.entry, self.path)
        self.write_bank(np.asarray([99], dtype=np.int64), {**identity, "cutoff_ms": 101})
        with self.assertRaisesRegex(ValueError, "cutoff_ms"):
            six_references.load_bank(self.path, self.root, self.manifest, self.entry)

    def test_bank_rejects_prepared_content_or_class_mapping_changes(self):
        self.prepared.write_bytes(self.prepared.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "prepared_sha256"):
            six_references.load_bank(self.path, self.root, self.manifest, self.entry)
        self.prepared.write_bytes(b"restored")
        changed = {**self.manifest, "classes": ["right", "left", "keeper-right", "keeper-left"]}
        with self.assertRaisesRegex(ValueError, "protocol_id"):
            six_references.load_bank(self.path, self.root, changed, self.entry)

    def test_descriptor_survives_missing_pitch_with_valid_geometry(self):
        image = Image.new("RGB", (100, 100), "white")
        descriptor = reference_bank.descriptor(
            {"bbox_image": {"x_center": 50, "y_center": 50, "w": 20, "h": 40}},
            [], image, {"x_center": 50, "y_center": 50, "w": 20, "h": 40},
        )
        self.assertIsNotNone(descriptor)
        self.assertTrue(np.isnan(descriptor[:2]).all())
        self.assertTrue(np.isfinite(descriptor[2:]).all())
        image.close()


if __name__ == "__main__":
    unittest.main()
