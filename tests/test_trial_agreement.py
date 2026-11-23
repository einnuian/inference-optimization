"""Check the distinction between delivered-output agreement and native diagnostics."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
from trial_analysis import compare
from trial_recorder import BENCHMARKS


class TrialAgreementTests(unittest.TestCase):
    def compare_fixture(self, change):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method, reference = root / "eagle3", root / "ar-sglang"
            method.mkdir()
            reference.mkdir()
            metadata = {"prepared_sha256": "a" * 64, "target_revision": "b" * 40,
                        "input_sha256": "c" * 64, "repetitions": 3,
                        "temperature": 0, "max_new_tokens": 2048,
                        "gpu_record": "GPU-test, NVIDIA L40S, 580.82.07", "gpu_uuid": "GPU-test"}
            for path in (method, reference):
                (path / "run.json").write_text(json.dumps({**metadata, "configuration": path.name}))
            for benchmark in sorted(BENCHMARKS):
                for repetition in range(1, 4):
                    rows = [{"prompt_id": "%s-%d" % (benchmark, index),
                             "input_token_ids": [3, index], "output_token_ids": [10 + index, 20],
                             "output_tokens": 2, "native_output_token_ids": [10 + index, 20, 128009],
                             "finish_reason": "stop", "native_stop_reason": 128009}
                            for index in range(3)]
                    control = {"benchmark": benchmark, "repetition": repetition,
                               "requests": rows, "metrics": {"otps": 20.0}}
                    actual = copy.deepcopy(control)
                    actual["metrics"]["otps"] = 40.0
                    change(actual["requests"][0])
                    name = "%02d-%s.json" % (repetition, benchmark)
                    (reference / name).write_text(json.dumps(control))
                    (method / name).write_text(json.dumps(actual))
            return compare(method, reference)

    def test_native_token_and_stop_differences_preserve_delivered_agreement(self):
        def change(row):
            row["native_output_token_ids"][-1] = 128001
            row["native_stop_reason"] = 128001

        results = self.compare_fixture(change)
        for benchmark in results.values():
            self.assertEqual(benchmark["exact_output_agreement_percent"], 100.0)
            self.assertEqual(benchmark["speedup_mean"], 2.0)
            first = benchmark["repetitions"][0]["checks"][0]
            self.assertTrue(first["exact_output_agreement"])
            self.assertTrue(first["delivered_token_ids_match"])
            self.assertTrue(first["output_length_match"])
            self.assertTrue(first["stopping_reason_match"])
            self.assertFalse(first["native_diagnostics"]["output_token_ids_match"])
            self.assertFalse(first["native_diagnostics"]["stop_reason_match"])

    def test_delivered_token_difference_breaks_agreement(self):
        results = self.compare_fixture(lambda row: row["output_token_ids"].__setitem__(0, 999))
        self.assert_mismatch(results, "delivered_token_ids_match")

    def test_output_length_difference_breaks_agreement(self):
        results = self.compare_fixture(lambda row: row.__setitem__("output_tokens", 1))
        self.assert_mismatch(results, "output_length_match")

    def test_common_stopping_reason_difference_breaks_agreement(self):
        results = self.compare_fixture(lambda row: row.__setitem__("finish_reason", "length"))
        self.assert_mismatch(results, "stopping_reason_match")

    def assert_mismatch(self, results, field):
        for benchmark in results.values():
            self.assertAlmostEqual(benchmark["exact_output_agreement_percent"], 200 / 3)
            first = benchmark["repetitions"][0]["checks"][0]
            self.assertFalse(first["exact_output_agreement"])
            self.assertFalse(first[field])
            self.assertTrue(first["native_diagnostics"]["output_token_ids_match"])
            self.assertTrue(first["native_diagnostics"]["stop_reason_match"])


if __name__ == "__main__":
    unittest.main()
