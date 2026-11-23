import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import json
import subprocess
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
from trial_sequence import logit_probe_command, run_ancillary_probe


class SequenceProbeTests(unittest.TestCase):
    def command(self, enabled=True, runtime=Path("/runtime"), configuration="dflash"):
        args = SimpleNamespace(probe_sglang_logits=enabled, diagnostic_runtime=runtime, port=31000)
        return logit_probe_command(args, Path("/code"), Path("/inputs.json"),
            Path("/trial/diagnostics"), Path("/trial/diagnostics/attempts/one"),
            Path("/trial/diagnostics/launch/one.json"), configuration)

    def test_probe_requires_explicit_diagnostic_scope(self):
        self.assertIsNone(self.command(enabled=False, runtime=None))
        self.assertIsNone(self.command(configuration="peagle"))
        with self.assertRaisesRegex(ValueError, "diagnostic runtime"):
            self.command(runtime=None)

    def test_probe_uses_primary_trial_and_separate_ancillary_output(self):
        command = self.command()
        self.assertEqual(command[command.index("--primary-results") + 1], "/trial")
        self.assertEqual(command[command.index("--output") + 1],
            "/trial/diagnostics/attempts/one/logit-probe/humaneval-source-index-2.json")
        self.assertEqual(command[command.index("--configuration") + 1], "dflash")
        self.assertIn("--run", command)

    def test_failed_probe_preserves_status_and_permits_the_next_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            with patch("trial_sequence.run_checked", side_effect=subprocess.CalledProcessError(1, ["probe"])):
                status = run_ancillary_probe(["probe"], attempt)
            saved = json.loads((attempt / "logit-probe" / "status.json").read_text())
            self.assertEqual(status, saved)
            self.assertEqual(saved["status"], "requires_review")
            self.assertEqual(saved["returncode"], 1)
            self.assertFalse(saved["included_in_primary_timing"])

    def test_interruption_propagates_out_of_the_optional_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("trial_sequence.run_checked", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_ancillary_probe(["probe"], Path(temporary))


if __name__ == "__main__":
    unittest.main()
