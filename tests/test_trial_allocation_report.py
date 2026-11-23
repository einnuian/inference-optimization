import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "baselines" / "trial_allocation_report.py"
SPEC = importlib.util.spec_from_file_location("trial_allocation_report", SCRIPT)
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class AllocationReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.allocation = {
            "job_id": "41", "start": "start", "end": "end", "state": "COMPLETED",
            "elapsed_seconds": 3600, "gpu_count": 1, "allocated_gpu_seconds": 3600,
            "alloc_tres": "cpu=8,mem=32G,node=1,gres/gpu=1",
        }

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def run_record(self, folder, job="41"):
        self.write(f"{folder}/run.json", {
            "configuration": "ar-sglang", "slurm_job_id": job, "hostname": "compute",
        })

    @staticmethod
    def request(request_id, tokens):
        return {
            "request_id": request_id, "output_tokens": tokens,
            "submitted_ns": 100, "completed_ns": 200,
        }

    def block(self, folder, filename, request):
        self.write(f"{folder}/{filename}", {
            "benchmark": "gsm8k", "repetition": 1, "requests": [request],
            "metrics": {"output_tokens": request["output_tokens"], "otps": 10.0},
        })

    def test_allocation_is_counted_once_and_request_copies_are_deduplicated(self):
        self.run_record("ar-sglang")
        self.write("ar-sglang/warmup.json", [self.request(f"w{i}", i) for i in (1, 2, 3)])
        primary = self.request("primary", 5)
        self.block("ar-sglang", "01-gsm8k.json", primary)
        self.block("ar-sglang", "02-gsm8k.json", primary)
        qualification = "qualification/ar-sglang"
        self.run_record(qualification)
        summaries = []
        for number, name in ((1, "one-token"), (2, "natural-stop")):
            filename = f"{number:02d}-{name}.json"
            record = {**self.request(f"q{number}", number), "slurm_job_id": "41"}
            self.write(f"{qualification}/{filename}", record)
            summaries.append({
                "request_id": f"q{number}", "trace_file": filename,
                "output_tokens": number, "completion_status": "completed",
            })
        self.write(f"{qualification}/qualification.json", {
            "slurm_job_id": "41", "qualified": True, "requests": summaries,
        })
        diagnostics = "diagnostics/ar-sglang"
        self.run_record(diagnostics)
        self.write(f"{diagnostics}/warmup.json", [self.request("dw", 4)])
        self.block(diagnostics, "01-gsm8k.json", self.request("db", 7))
        # Runtime instrumentation lives outside the diagnostic request folder.
        self.run_record("diagnostic-runtime/ar-sglang")
        self.write("diagnostic-runtime/ar-sglang/warmup.json", [self.request("excluded", 999)])

        report = REPORT.build_report(self.root, self.allocation)

        self.assertEqual(report["allocation"]["allocated_gpu_hours"], 1)
        self.assertEqual(report["allocation"]["allocated_cpu_hours"], 8)
        self.assertEqual(report["recorded_request_totals"]["request_count"], 8)
        self.assertEqual(report["recorded_request_totals"]["returned_output_tokens_with_evidence"], 25)
        scopes = report["requests_by_scope"]
        self.assertEqual(scopes["primary_timed_benchmarks"]["request_count"], 1)
        self.assertEqual(scopes["stream_qualification"]["request_count"], 2)
        self.assertEqual(scopes["diagnostic_timed_benchmarks"]["returned_output_tokens_with_evidence"], 7)

    def test_other_jobs_are_excluded_and_inconsistent_allocation_time_is_rejected(self):
        self.run_record("ar-sglang", job="99")
        self.write("ar-sglang/warmup.json", [self.request("other-job", 20)])
        report = REPORT.build_report(self.root, self.allocation)
        self.assertEqual(report["recorded_request_totals"]["request_count"], 0)
        self.assertTrue(any(row["code"] == "job_id_mismatch" for row in report["missing_token_evidence"]))
        with self.assertRaises(ValueError):
            REPORT.build_report(self.root, {**self.allocation, "allocated_gpu_seconds": 1})

    def test_attempt_qualification_is_counted_and_duplicate_copies_are_deduplicated(self):
        for base in ("attempts/ar-sglang-retry", "diagnostics/attempts/ar-sglang-check"):
            folder = base + "/qualification"
            self.run_record(folder)
            self.write(folder + "/qualification.json", {
                "slurm_job_id": "41", "qualified": True,
                "requests": [{"request_id": "same-q", "trace_file": "01-one-token.json", "output_tokens": 1}],
            })
            self.write(folder + "/01-one-token.json", {
                **self.request("same-q", 1), "slurm_job_id": "41",
            })
        report = REPORT.build_report(self.root, self.allocation, ["ar-sglang"])
        self.assertEqual(report["requests_by_scope"]["stream_qualification"]["request_count"], 1)
        self.assertEqual(report["requests_by_scope"]["stream_qualification"]["returned_output_tokens_with_evidence"], 1)

    def test_conflicting_copies_leave_the_request_token_count_unknown(self):
        self.run_record("ar-sglang")
        self.write("ar-sglang/warmup.json", [self.request("conflict", 5)])
        self.block("ar-sglang", "01-gsm8k.json", self.request("conflict", 7))
        report = REPORT.build_report(self.root, self.allocation)
        totals = report["recorded_request_totals"]
        self.assertEqual(totals["request_count"], 1)
        self.assertEqual(totals["returned_output_tokens_with_evidence"], 0)
        self.assertEqual(totals["requests_with_missing_token_counts"], 1)
        self.assertTrue(any(row["code"] == "conflicting_request" for row in report["missing_token_evidence"]))

    def test_qualification_summary_recovers_tokens_and_missing_tokens_stay_explicit(self):
        folder = "qualification/ar-sglang"
        self.run_record(folder)
        self.write(f"{folder}/qualification.json", {
            "slurm_job_id": "41", "qualified": False,
            "requests": [
                {"request_id": "saved-summary", "trace_file": "01-one-token.json", "output_tokens": 3},
                {"request_id": "partial", "trace_file": "02-natural-stop.json", "completion_status": "incomplete"},
            ],
        })
        self.write(f"{folder}/02-natural-stop.json", {
            "slurm_job_id": "41", "request_id": "partial", "completion_status": "incomplete", "wire_events": [],
        })
        report = REPORT.build_report(self.root, {**self.allocation, "alloc_tres": "gres/gpu=1"})
        scope = report["requests_by_scope"]["stream_qualification"]
        self.assertEqual(scope["request_count"], 2)
        self.assertEqual(scope["returned_output_tokens_with_evidence"], 3)
        self.assertEqual(scope["requests_with_missing_token_counts"], 1)
        self.assertIsNone(report["allocation"]["allocated_cpu_hours"])
        self.assertEqual(report["missing_resource_evidence"][0]["code"], "missing_cpu_count")

    def probe(self, *, partial=False, job="41", eligible=False):
        record = {"schema_version": 1, "scope": "ancillary_agreement_qualification",
            "configuration": "ar-sglang", "status": "requires_review",
            "request_payload": {"rid": "hpml-probe-" + "a" * 32},
            "scores_at_first_divergence": {"eligible": eligible, "reason": "The scores require review."},
            "output_comparison": {"preserved": False},
            "accounting": {"classification": "ancillary_qualification", "included_in_primary_timing": False,
                "slurm_job_id": job, "submitted_monotonic_ns": 100, "completed_monotonic_ns": 200,
                "observed_native_token_count": 2 if partial else 4}}
        if partial:
            record.update(status="failed", error="The response ended early.")
        else:
            record["response"] = {"output_tokens": 3, "output_token_ids": [1, 2, 3],
                                  "native_output_token_ids": [1, 2, 3, 99]}
            record["accounting"].update(output_tokens=3, native_completion_tokens=4)
        return record

    def test_ineligible_and_malformed_score_interpretations_preserve_token_accounting(self):
        for stamp in ("221003994481", "221003994482"):
            self.write(f"diagnostics/attempts/ar-sglang-20261122T{stamp}Z/logit-probe/humaneval-source-index-2.json",
                       self.probe(eligible=False if stamp.endswith("1") else "invalid"))
        report = REPORT.build_report(self.root, self.allocation, ["ar-sglang"])
        scope = report["requests_by_scope"]["ancillary_score_probes"]
        self.assertEqual(scope["request_count"], 1)
        self.assertEqual(scope["returned_output_tokens_with_evidence"], 3)
        accounting = report["ancillary_probe_token_accounting"]
        self.assertEqual(accounting["unique_request_count"], 1)
        self.assertEqual(accounting["native_completion_tokens_with_evidence"], 4)
        self.assertEqual(accounting["observed_native_token_lower_bound"], 4)
        self.assertEqual(len(accounting["requests"][0]["sources"]), 2)
        interpretations = report["ancillary_probe_interpretations"]
        self.assertFalse(interpretations[0]["score_eligible"])
        self.assertIsNone(interpretations[1]["score_eligible"])
        self.assertFalse(interpretations[1]["score_metadata_valid"])

    def test_partial_probe_preserves_observed_lower_bound_and_unknown_final_counts(self):
        self.write("diagnostics/attempts/ar-sglang-20261122T221003994481Z/logit-probe/humaneval-source-index-2.json",
                   self.probe(partial=True))
        report = REPORT.build_report(self.root, self.allocation, ["ar-sglang"])
        scope = report["requests_by_scope"]["ancillary_score_probes"]
        self.assertEqual(scope["request_count"], 1)
        self.assertEqual(scope["requests_with_missing_token_counts"], 1)
        accounting = report["ancillary_probe_token_accounting"]
        self.assertEqual(accounting["native_completion_tokens_with_evidence"], 0)
        self.assertEqual(accounting["requests_with_unknown_final_native_counts"], 1)
        self.assertEqual(accounting["observed_native_token_lower_bound"], 2)

    def test_other_job_probes_and_untrusted_attempt_names_are_excluded(self):
        self.write("diagnostics/attempts/ar-sglang-20261122T221003994481Z/logit-probe/humaneval-source-index-2.json",
                   self.probe(job="99"))
        self.write("diagnostics/attempts/ar-sglang-untrusted/logit-probe/humaneval-source-index-2.json", self.probe())
        report = REPORT.build_report(self.root, self.allocation, ["ar-sglang"])
        self.assertEqual(report["requests_by_scope"]["ancillary_score_probes"]["request_count"], 0)
        self.assertEqual(report["ancillary_probe_token_accounting"]["unique_request_count"], 0)

    def test_conflicting_probe_counts_remain_unknown(self):
        record = self.probe()
        record["accounting"]["native_completion_tokens"] = 9
        self.write("diagnostics/attempts/ar-sglang-20261122T221003994481Z/logit-probe/humaneval-source-index-2.json", record)
        report = REPORT.build_report(self.root, self.allocation, ["ar-sglang"])
        accounting = report["ancillary_probe_token_accounting"]
        self.assertEqual(accounting["native_completion_tokens_with_evidence"], 0)
        self.assertEqual(accounting["observed_native_token_lower_bound"], 0)
        self.assertTrue(accounting["requests"][0]["conflicting_evidence"])
        self.assertEqual(report["requests_by_scope"]["ancillary_score_probes"]["returned_output_tokens_with_evidence"], 3)


if __name__ == "__main__":
    unittest.main()
