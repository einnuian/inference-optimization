import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
from compute_usage import gpu_count, parse_jobs, totals
from trial_recorder import show_prompt


class ComputeUsageTests(unittest.TestCase):
    def test_generic_gpu_total_does_not_double_count_typed_gpus(self):
        self.assertEqual(gpu_count("cpu=8,gres/gpu:l40s=2,gres/gpu=2"), 2)
        self.assertEqual(gpu_count("gres/gpu:l40s=2,gres/gpu:a100=1"), 3)
        self.assertEqual(gpu_count("cpu=2,mem=8G"), 0)

    def test_allocation_time_and_explicit_project_membership(self):
        raw = "1|hpml-check|gpu|edu|FAILED|2026-11-22T00:00:00|2026-11-22T01:00:00|3600|8|cpu=8,gres/gpu=2|gpu01\n"
        raw += "2|env|short|edu|COMPLETED|2026-11-22T00:00:00|2026-11-22T00:01:00|60|2|cpu=2|cpu01\n"
        jobs = parse_jobs(raw, {"2"})
        self.assertTrue(all(row["project_job"] for row in jobs))
        self.assertEqual(totals(jobs)["allocated_gpu_hours"], 2)
        self.assertAlmostEqual(totals(jobs)["allocated_cpu_core_hours"], 8 + 1 / 30)
        with self.assertRaises(ValueError):
            parse_jobs(raw + raw, set())

    def test_exact_prompt_is_visible_with_method_and_identity(self):
        stream = io.StringIO()
        row = {"id": "example-1", "benchmark": "example", "messages": [{"content": "Line one.\nLine two."}]}
        with redirect_stdout(stream):
            elapsed = show_prompt("eagle3", row, "Repetition 1/3")
        self.assertIn("eagle3 | example | example-1", stream.getvalue())
        self.assertIn("Line one.\nLine two.", stream.getvalue())
        self.assertGreaterEqual(elapsed, 0)


if __name__ == "__main__":
    unittest.main()
