"""Validate diagnostic GPU attribution through the checked host launch record."""

import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import trial_entrypoint as entrypoint


class DiagnosticAllocationTests(unittest.TestCase):
    def test_visible_gpu_uuid_is_recorded_and_conflicting_or_invalid_ids_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary).resolve() / "results"
            parent = results / "trial-first30-20261122/diagnostic-runtime/vllm"
            parent.mkdir(parents=True)
            output = results / "trial-first30-20261122/diagnostics/trace"
            record = results / "trial-first30-20261122/launch.json"
            common = {"slurm_job_id": "123", "hostname": socket.gethostname(), "uid": os.getuid()}
            manifest = parent / "diagnostic-instrumentation.json"
            manifest.write_text(json.dumps(common))
            launch = {**common, "configuration": "peagle", "launch_unix_ns": time.time_ns(),
                      "launch_perf_counter_ns": time.perf_counter_ns(),
                      "diagnostic": {"scope": "separate_diagnostic",
                          "package_results_relative": parent.relative_to(results).as_posix(),
                          "output_results_relative": output.relative_to(results).as_posix(),
                          "allocation_record_results_relative": record.relative_to(results).as_posix(),
                          "instrumentation_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                          "entrypoint_sha256": hashlib.sha256(Path(entrypoint.__file__).read_bytes()).hexdigest()},
                      "command": [str(Path(entrypoint.__file__).resolve()), "--package", str(parent),
                          "--backend", "vllm", "--configuration", "peagle", "--diagnostic-output", str(output),
                          "--allocation-record", str(record), "--"]}
            record.write_text(json.dumps(launch))
            with mock.patch.dict(os.environ, {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"}, clear=True), \
                    mock.patch.object(entrypoint.subprocess, "check_output", return_value="GPU-ab12-cd34, NVIDIA L40S") as query:
                entrypoint.validate_allocation_record(record, parent, output, "vllm", "peagle", results)
                self.assertEqual(os.environ["HPML_GPU_UUID"], "GPU-ab12-cd34")
                self.assertIn("--id=0", query.call_args.args[0])
                os.environ["HPML_GPU_UUID"] = "GPU-ffff"
                with self.assertRaisesRegex(ValueError, "differs from the visible"):
                    entrypoint.validate_allocation_record(record, parent, output, "vllm", "peagle", results)
                self.assertEqual(os.environ["HPML_GPU_UUID"], "GPU-ffff")
                del os.environ["HPML_GPU_UUID"]
                query.return_value = "unidentified, NVIDIA L40S"
                with self.assertRaisesRegex(ValueError, "physical UUID"):
                    entrypoint.validate_allocation_record(record, parent, output, "vllm", "peagle", results)
                self.assertNotIn("HPML_GPU_UUID", os.environ)


if __name__ == "__main__":
    unittest.main()
