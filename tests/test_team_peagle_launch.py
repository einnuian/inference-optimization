"""Keep a qualification omission separate from all inference commands."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
from team_peagle_launch import omission, save_omission


class LaunchPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = {"qualification_omissions": [{
            "slurm_job_id": "123", "configuration": "team-ffn-100", "reason": "Complete the primary trial.",
        }]}

    def test_only_selected_job_and_configuration_omit_qualification(self):
        arguments = ["qualify", "--configuration", "team-ffn-100", "--output", "/tmp/evidence.json"]
        self.assertIsNotNone(omission(arguments, "123", self.policy))
        self.assertIsNone(omission(arguments, "456", self.policy))
        arguments[2] = "team-ffn-75"
        self.assertIsNone(omission(arguments, "123", self.policy))

    def test_serving_recording_and_comparison_always_use_the_original_adapter(self):
        for command in ("serve", "record", "compare", "export-source"):
            with self.subTest(command=command):
                self.assertIsNone(omission([command, "--configuration", "team-ffn-100"], "123", self.policy))

    def test_omission_requires_a_saved_evidence_record(self):
        with self.assertRaisesRegex(ValueError, "evidence path"):
            omission(["qualify", "--configuration=team-ffn-100"], "123", self.policy)

    def test_omission_record_is_written_without_an_adapter_save_function(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qualification.json"
            save_omission(path, {"core_adapter_sha256": "source-hash"}, "team-ffn-100", "Complete the trial.")
            record = json.loads(path.read_text())
            self.assertEqual(record["status"], "omitted")
            self.assertFalse(record["source_equivalence_verified"])
            self.assertEqual(record["core_adapter_sha256"], "source-hash")


if __name__ == "__main__":
    unittest.main()
