"""Check the team controller's phase evidence and same-GPU result reuse."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import team_peagle_sequence as sequence


class TeamSequenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_phase_records_preserve_gpu_job_and_monotonic_interval(self):
        path = self.root / "phases.jsonl"
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "42"}):
            ledger = sequence.PhaseLog(path, "GPU-one")
            with ledger.phase("original_checkpoint_import", "team-ffn-75") as phase:
                phase["process_pid"] = 123
            ledger.close()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([row["event"] for row in rows], ["started", "ended"])
        self.assertEqual(rows[1]["gpu_uuid"], "GPU-one")
        self.assertEqual(rows[1]["slurm_job_id"], "42")
        self.assertEqual(rows[1]["process_pid"], 123)
        self.assertGreaterEqual(rows[1]["ended_ns"], rows[1]["started_ns"])
        self.assertEqual(rows[1]["status"], "completed")

    def test_phase_failure_is_recorded_and_propagated(self):
        path = self.root / "phases.jsonl"
        with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "42"}):
            ledger = sequence.PhaseLog(path, "GPU-one")
            with self.assertRaisesRegex(RuntimeError, "source unavailable"):
                with ledger.phase("original_checkpoint_import"):
                    raise RuntimeError("source unavailable")
            ledger.close()
        ending = json.loads(path.read_text().splitlines()[-1])
        self.assertEqual(ending["status"], "failed")
        self.assertEqual(ending["error_type"], "RuntimeError")

    def complete_run(self):
        path = self.root / "team-ar"
        path.mkdir()
        metadata = {
            "configuration": "team-ar", "phase": "primary", "gpu_uuid": "GPU-one",
            "prepared_sha256": "prepared-sha", "input_sha256": sequence.INPUT_SHA256,
            "target_revision": sequence.TARGET_REVISION, "repetitions": 3,
        }
        (path / "run.json").write_text(json.dumps(metadata))
        (path / "summary.json").write_text(json.dumps({name: {} for name in sequence.BENCHMARKS}))
        for repetition in range(1, 4):
            for benchmark in sequence.BENCHMARKS:
                (path / f"{repetition:02d}-{benchmark}.json").write_text(json.dumps({
                    "repetition": repetition, "benchmark": benchmark,
                    "requests": [{"server_metadata": {"team_rounds": []}} for _ in range(3)],
                }))
        return path

    def test_completed_runs_require_the_same_physical_gpu(self):
        path = self.complete_run()
        sequence.validate_completed(path, "team-ar", "primary", "GPU-one", "prepared-sha")
        with self.assertRaisesRegex(ValueError, "gpu_uuid"):
            sequence.validate_completed(path, "team-ar", "primary", "GPU-two", "prepared-sha")

    def test_missing_blocks_remain_incomplete(self):
        path = self.complete_run()
        next(path.glob("01-*.json")).unlink()
        with self.assertRaises(FileNotFoundError):
            sequence.validate_completed(path, "team-ar", "primary", "GPU-one", "prepared-sha")

    def test_explicit_policy_selects_primary_without_changing_requests(self):
        (self.root / "sequence-policy.json").write_text(json.dumps({
            "phases": ["primary"],
            "reason": "Run all primary measurements in this allocation; collect diagnostics separately.",
        }))
        phases, record = sequence.execution_phases(self.root, True)
        self.assertEqual(phases, ["primary"])
        self.assertEqual(record["sha256"], sequence.sha256(self.root / "sequence-policy.json"))
        self.assertEqual(set(sequence.CONFIGURATIONS), {"team-ar", "team-ffn-100", "team-ffn-75", "team-ffn-50"})

    def test_policy_rejects_unknown_or_reordered_phases(self):
        for phases in ([], ["diagnostic", "primary"], ["training"], ["primary", "primary"]):
            with self.subTest(phases=phases):
                (self.root / "sequence-policy.json").write_text(json.dumps({"phases": phases, "reason": "Test."}))
                with self.assertRaises(ValueError):
                    sequence.execution_phases(self.root, True)

    def test_default_preserves_requested_phase_selection(self):
        self.assertEqual(sequence.execution_phases(self.root, False), (["primary"], None))
        self.assertEqual(sequence.execution_phases(self.root, True), (["primary", "diagnostic"], None))

    def test_reference_policy_preserves_the_explicit_storage_choice(self):
        path = self.root / "sequence-policy.json"
        policy = {"phases": ["primary"], "reason": "Read the verified shared artifacts.",
                  "checkpoint_storage": "verified_references"}
        path.write_text(json.dumps(policy))
        phases, record = sequence.execution_phases(self.root, True)
        self.assertEqual(phases, ["primary"])
        self.assertEqual(record["checkpoint_storage"], "verified_references")
        policy["checkpoint_storage"] = "unchecked"
        path.write_text(json.dumps(policy))
        with self.assertRaises(ValueError):
            sequence.execution_phases(self.root, True)

    def test_policy_preserves_pilot_priority_and_rejects_duplicates(self):
        path = self.root / "sequence-policy.json"
        order = ["team-ffn-75", "team-ffn-50", "team-ar", "team-ffn-100"]
        policy = {"phases": ["primary"], "reason": "Measure the remaining pilot widths first.",
                  "configurations": order}
        path.write_text(json.dumps(policy))
        self.assertEqual(sequence.execution_phases(self.root, False)[1]["configurations"], order)
        policy["configurations"] = ["team-ffn-75", "team-ffn-75"]
        path.write_text(json.dumps(policy))
        with self.assertRaisesRegex(ValueError, "configuration"):
            sequence.execution_phases(self.root, False)

    def test_reuse_preserves_original_metadata_on_the_same_gpu(self):
        source = self.complete_run()
        before = (source / "run.json").read_bytes()
        destination = self.root / "reused-ar"
        result = sequence.reuse_completed_primary(source, destination, "team-ar", "GPU-one", "prepared-sha")
        self.assertEqual(result["status"], "reused_completed")
        self.assertEqual((destination / "run.json").read_bytes(), before)
        self.assertEqual((source / "run.json").read_bytes(), before)
        sequence.validate_completed(destination, "team-ar", "primary", "GPU-one", "prepared-sha")
        (destination / "run.json").write_text('{}')
        self.assertEqual((source / "run.json").read_bytes(), before)

    def test_different_gpu_requires_new_measurement(self):
        source = self.complete_run()
        destination = self.root / "fresh-ar"
        result = sequence.reuse_completed_primary(source, destination, "team-ar", "GPU-two", "prepared-sha")
        self.assertEqual(result["status"], "different_gpu")
        self.assertFalse(destination.exists())

    def test_reuse_rejects_changed_prompt_provenance(self):
        source = self.complete_run()
        with self.assertRaisesRegex(ValueError, "prepared_sha256"):
            sequence.reuse_completed_primary(source, self.root / "invalid", "team-ar", "GPU-one", "new-prompts")


if __name__ == "__main__":
    unittest.main()
