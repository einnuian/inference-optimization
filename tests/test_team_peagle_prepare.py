"""Protect preparation ownership, completion provenance, and CPU accounting."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import team_peagle_prepare as prepare
import team_peagle_sequence as sequence


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_active_preparation_blocks_another_preparer_and_the_gpu_controller(self):
        with prepare.preparation_lock(self.root):
            with self.assertRaises(FileExistsError):
                with prepare.preparation_lock(self.root):
                    self.fail("A second preparer acquired the active lock.")
            with self.assertRaisesRegex(ValueError, "CPU preparation lock"):
                sequence.require_preparation_idle(self.root)
        sequence.require_preparation_idle(self.root)

    def test_failed_preparation_releases_its_own_lock(self):
        with self.assertRaisesRegex(RuntimeError, "import failed"):
            with prepare.preparation_lock(self.root):
                raise RuntimeError("import failed")
        self.assertFalse((self.root / ".preparation.lock").exists())

    def test_cpu_phase_accounts_for_zero_gpus(self):
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "123"}):
            ledger = sequence.PhaseLog(self.root / "phases.jsonl", None)
            with ledger.phase("cpu_checkpoint_verification"):
                pass
            ledger.close()
        events = [json.loads(line) for line in (self.root / "phases.jsonl").read_text().splitlines()]
        self.assertTrue(all(event["allocated_gpu_count"] == 0 for event in events))
        self.assertTrue(all(event["gpu_uuid"] is None for event in events))

    def completed_import(self):
        directory = self.root / "checkpoints"
        directory.mkdir()
        prepared = self.root / "prepared.json"
        prepared.write_text('{}')
        record = {
            "artifact_kind": "team_original_checkpoint_import", "status": "complete",
            "source_directory": str((self.root / "source").resolve()),
            "evaluation_prepared_sha256": prepare.sha256(prepared),
            "evaluation_input_sha256": prepare.INPUT_SHA256,
            "evaluation_target_revision": prepare.TARGET_REVISION,
            "evaluation_notebook_sha256": prepare.NOTEBOOK_SHA256,
            "overlap_match_count": 0,
        }
        for field, filename in (("training_data_sha256", "train_data.jsonl"),
                                ("vocabulary_sha256", "draft_vocab.json"),
                                ("overlap_report_sha256", "evaluation-overlap.json")):
            (directory / filename).write_text('{}')
            record[field] = prepare.sha256(directory / filename)
        (directory / "import-manifest.json").write_text(json.dumps(record))
        return directory, prepared, record

    def test_reuse_requires_completed_matching_provenance(self):
        directory, prepared, record = self.completed_import()
        prepare.validate_import(directory, self.root / "source", prepared)
        record["evaluation_target_revision"] = "another-target"
        (directory / "import-manifest.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "evaluation_target_revision"):
            prepare.validate_import(directory, self.root / "source", prepared)

    def test_reuse_detects_changed_vocabulary(self):
        directory, prepared, _ = self.completed_import()
        (directory / "draft_vocab.json").write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, "draft_vocab.json"):
            prepare.validate_import(directory, self.root / "source", prepared)


if __name__ == "__main__":
    unittest.main()
