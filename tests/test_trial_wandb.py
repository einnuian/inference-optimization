"""Validate W&B publication inputs without contacting an external service."""

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import trial_wandb as uploader
from trial_native_stats import delta


def save(path, value):
    path.write_text(json.dumps(value))


def snapshot(completions, tick, rounds):
    names = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
    values = (rounds, rounds * 3, rounds)
    return {"metrics_url": "http://127.0.0.1:31000/metrics", "hostname": "fixture-host",
            "gpu_uuid": "GPU-fixture", "observed_monotonic_ns": tick,
            "observed_completions": completions,
            "samples": [{"name": "vllm:spec_decode_" + name + "_total",
                         "labels": {"model_name": "hpml-target"}, "value": value}
                        for name, value in zip(names, values)]}


def configuration(root, name):
    directory = root / name
    directory.mkdir()
    backend, reference = uploader.CONFIGURATIONS[name]
    team = name in uploader.TEAM_CONFIGURATIONS
    metadata = {"configuration": name, "backend": backend, "reference": reference,
                "model": "hpml-target", "repetitions": 3, "temperature": 0, "max_new_tokens": 2048,
                "target_revision": uploader.TARGET_REVISION, "input_sha256": uploader.INPUT_SHA256,
                "prepared_sha256": "a" * 64, "recorder_sha256": ("b" if team else "c") * 64,
                "server_record_sha256": "d" * 64, "gpu_uuid": "GPU-fixture",
                "gpu_record": "GPU-fixture, NVIDIA L40S, fixture-driver"}
    if team:
        metadata.update(adapter_sha256="e" * 64, phase="primary")
        metadata["server_settings"] = {
            "configuration": name, "backend": backend, "phase": "primary",
            "target_revision": uploader.TARGET_REVISION, "adapter_sha256": "e" * 64,
            "notebook_sha256": "1" * 64, "notebook_commit": "2" * 40,
            "checkpoint_sha256": "3" * 64, "training_data_sha256": "4" * 64,
            "draft_vocabulary_sha256": "5" * 64, "training_examples": 200,
            "training_updates": 50, "training_manifest": {"status": "complete", "seed": 0},
        }
    save(directory / "run.json", metadata)
    materialized = {"team_rounds": [],
                    "team_rounds_materialization": "Expanded after measurement from the raw compact stream counts."}
    save(directory / "warmup.json", [{"prompt_id": "warmup-%d" % index, "finish_reason": "stop",
                                      "server_metadata": materialized if team else {}} for index in range(3)])
    (directory / "live.jsonl").write_text("")
    previous = snapshot(5, 1, 3)
    if name in {"peagle", "dspark"}:
        save(directory / "native-after-warmup.json", previous)
    for repetition in range(1, 4):
        for benchmark_index, benchmark in enumerate(uploader.BENCHMARKS):
            start = (repetition * 100 + benchmark_index) * 10_000_000_000
            requests = []
            for index in range(3):
                submitted = start + index * 1_000_000_000
                requests.append({
                    "request_id": "%s-%s-%d-%d" % (name, benchmark, repetition, index),
                    "prompt_id": "%s-%d" % (benchmark, index), "benchmark": benchmark,
                    "input_token_ids": [1, index + 2], "output_token_ids": [10, 11],
                    "native_output_token_ids": [10, 11, 128009], "output_tokens": 2,
                    "output_text": "Fixture response.", "finish_reason": "stop",
                    "submitted_ns": submitted, "completed_ns": submitted + 500_000_000,
                    "end_to_end_seconds": .5, "ttft_seconds": .1, "tpot_seconds": .4,
                    "server_metadata": {"spec_verify_ct": 1, "spec_num_correct_drafts": 1,
                                        "spec_correct_drafts_histogram": [0, 1],
                                        **(materialized if team else {})},
                })
            block = {"benchmark": benchmark, "repetition": repetition, "started_ns": start,
                     "ended_ns": start + 2_500_000_000, "requests": requests,
                     "metrics": {"output_tokens": 6, "elapsed_seconds": 2.5, "otps": 2.4}}
            save(directory / ("%02d-%s.json" % (repetition, benchmark)), block)
            if name in {"peagle", "dspark"}:
                after = snapshot(previous["observed_completions"] + 3,
                                 previous["observed_monotonic_ns"] + 1,
                                 previous["samples"][0]["value"] + 3)
                save(directory / ("native-%02d-%s.json" % (repetition, benchmark)),
                     {"benchmark": benchmark, "repetition": repetition,
                      "before": previous, "after": after, "metrics": delta(previous, after)})
                previous = after
    save(directory / "summary.json", {
        benchmark: {"across_repetitions": {"otps": {"n": 3, "mean": 2.4, "sample_sd": 0.0}}}
        for benchmark in uploader.BENCHMARKS})
    return directory


def catalogue(root):
    output = root / "export"
    output.mkdir()
    source = root / "eagle3" / "01-gsm8k.json"
    source.parent.mkdir(exist_ok=True)
    save(source, {"fixture": "source observation"})
    relative = source.relative_to(root).as_posix()
    source_record = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "bytes": source.stat().st_size}
    for team in (False, True):
        cell = {"metricId": "metric-otps", "variant": "", "benchmark": "GSM8K", "position": None,
                "method": uploader.METHOD_LABELS["team-ffn-100" if team else "eagle3"],
                "value": None if team else 4.0, "sd": None if team else 2.0,
                "n": 0 if team else 3, "unit": "tokens/s", "evidence": "" if team else relative}
        reason = "The diagnostic observation is unavailable."
        if team:
            cell["reason"] = reason
        key = {field: cell[field] for field in ("metricId", "variant", "method", "benchmark", "position")}
        observations = [] if team else [{"repetition": repeat, "value": repeat * 2.0, "sources": [relative]}
                                       for repeat in range(1, 4)]
        missing = [{"repetition": repeat, "reason": reason} for repeat in range(1, 4)] if team else []
        prefix = "team-" if team else ""
        save(output / (prefix + "trial-results.json"), {"trialId": root.name, "status": "partial", "cells": [cell]})
        save(output / (prefix + "measurement-evidence.json"),
             {"schema_version": 1, "trial_id": root.name, "source_files": {} if team else {relative: source_record},
              "cells": [{"key": key, "observations": observations, "missing_repetitions": missing}]})
    return output


class WandbPublicationTests(unittest.TestCase):
    def test_original_team_artifact_preserves_unknown_training_fields_and_manifest_hashes(self):
        settings = {"artifact_origin": "teammate-original", "training_target_revision": None,
                    "training_seed": None, "training_manifest": None,
                    "checkpoint_manifest": {"verified": True},
                    "import_manifest": {"training_dataset_revision": None, "overlap_match_count": 0}}
        result = uploader.source_provenance({"server_settings": settings})
        self.assertEqual(result["artifact_origin"], "teammate-original")
        self.assertIsNone(result["training_target_revision"])
        self.assertIsNone(result["training_seed"])
        self.assertIn("checkpoint_manifest_sha256", result)
        self.assertIn("import_manifest_sha256", result)
        self.assertNotIn("training_manifest_sha256", result)
        self.assertEqual(uploader.METHOD_LABELS["team-ffn-100"], "Pilot FFN 100%")

    def test_all_eleven_configurations_allow_independent_team_recorder_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in uploader.CONFIGURATIONS:
                configuration(root, name)
            _, records = uploader.validate_trial(root)
            self.assertEqual(len(records), 11)
            self.assertEqual(sum(len(record["requests"]) for record in records), 990)
            team = next(record for record in records if record["configuration"] == "team-ffn-100")
            self.assertEqual(team["source_provenance"]["training_updates"], 50)
            self.assertIn("training_manifest_sha256", team["source_provenance"])
            changed = root / "team-ffn-50" / "run.json"
            data = json.loads(changed.read_text())
            data["adapter_sha256"] = data["server_settings"]["adapter_sha256"] = "f" * 64
            save(changed, data)
            with self.assertRaisesRegex(ValueError, "recorder group differ in adapter_sha256"):
                uploader.validate_trial(root)

    def test_explicit_partial_selection_requires_the_matching_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            configuration(root, "ar-sglang")
            configuration(root, "dflash")
            _, records = uploader.validate_trial(root, configurations=["ar-sglang", "dflash"])
            self.assertEqual([record["configuration"] for record in records], ["ar-sglang", "dflash"])
            with self.assertRaisesRegex(ValueError, "matching reference"):
                uploader.validate_trial(root, configurations=["dflash"])

    def test_complete_team_run_can_publish_while_reference_is_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            configuration(root, "team-ffn-100")
            reference = configuration(root, "team-ar")
            (reference / "summary.json").unlink()
            (reference / "03-gsm8k.json").unlink()
            _, records = uploader.validate_trial(root, configurations=["team-ffn-100"],
                                                 allow_unpaired_team=True)
            self.assertEqual(len(records[0]["requests"]), 90)
            self.assertTrue(all(path.parent.name == "team-ffn-100" for path in records[0]["paths"]))
            self.assertNotIn("speedup", records[0]["summary"]["gsm8k"]["across_repetitions"])
            with self.assertRaisesRegex(ValueError, "matching reference"):
                uploader.validate_trial(root, configurations=["team-ffn-100"])
            with self.assertRaisesRegex(ValueError, "Select explicit"):
                uploader.validate_trial(root, allow_unpaired_team=True)

    def test_unpaired_team_option_preserves_complete_block_requirement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            saved = configuration(root, "team-ffn-100")
            block_path = saved / "03-gsm8k.json"
            block = json.loads(block_path.read_text())
            block["requests"].pop()
            save(block_path, block)
            with self.assertRaisesRegex(ValueError, "three completed requests"):
                uploader.validate_trial(root, configurations=["team-ffn-100"], allow_unpaired_team=True)
            block_path.unlink()
            with self.assertRaisesRegex(ValueError, "all 30 completed"):
                uploader.validate_trial(root, configurations=["team-ffn-100"], allow_unpaired_team=True)

    def test_team_publication_waits_for_post_timing_materialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            saved = configuration(root, "team-ffn-100")
            block_path = saved / "03-gsm8k.json"
            block = json.loads(block_path.read_text())
            del block["requests"][0]["server_metadata"]["team_rounds"]
            save(block_path, block)
            with self.assertRaisesRegex(ValueError, "completed post-measurement"):
                uploader.validate_trial(root, configurations=["team-ffn-100"], allow_unpaired_team=True)
            (saved / "03-gsm8k.json.expanding").write_text("pending")
            with self.assertRaisesRegex(ValueError, "still saving"):
                uploader.validate_trial(root, configurations=["team-ffn-100"], allow_unpaired_team=True)

    def test_selected_reference_requires_the_same_physical_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            configuration(root, "team-ffn-100")
            reference = configuration(root, "team-ar") / "run.json"
            metadata = json.loads(reference.read_text())
            metadata["gpu_uuid"] = "GPU-other"
            save(reference, metadata)
            with self.assertRaisesRegex(ValueError, "reference differ in gpu_uuid"):
                uploader.validate_trial(root, configurations=["team-ffn-100", "team-ar"])
            _, records = uploader.validate_trial(root, configurations=["team-ffn-100", "team-ar"],
                                                 allow_unpaired_team=True)
            self.assertIn("gpu_uuid", records[0]["unpaired_reference_reason"])
            sources = {"team-ffn-100/run.json": {}, "team-ar/run.json": {}}
            with self.assertRaisesRegex(ValueError, "incompatible gpu_uuid"):
                uploader.validate_catalogue_pair(root, "team-ffn-100", sources)
            metadata["gpu_uuid"] = "GPU-fixture"
            save(reference, metadata)
            uploader.validate_catalogue_pair(root, "team-ffn-100", sources)

    def test_nested_results_roots_and_gpu_attempts_have_separate_run_ids(self):
        first = Path("/saved/team-priority-first30-14056875/results")
        second = Path("/saved/team-original-first30-14058516/results")
        record = {"configuration": "team-ffn-100", "run": {"gpu_uuid": "GPU-one", "slurm_job_id": "14056875"}}
        uploader.assign_run_identity(first, record, "v1")
        initial_id = record["run_id"]
        uploader.assign_run_identity(second, record, "v1")
        self.assertNotEqual(record["run_id"], initial_id)
        record["run"]["gpu_uuid"] = "GPU-two"
        uploader.assign_run_identity(first, record, "v1")
        self.assertNotEqual(record["run_id"], initial_id)
        self.assertEqual(uploader.trial_identity(first), "team-priority-first30-14056875/results")

    def test_native_counter_files_are_validated_and_included(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            path = configuration(root, "peagle")
            record = uploader.validate_configuration(root, "peagle")
            native = [file.name for file in record["paths"] if file.name.startswith("native-")]
            self.assertEqual(len(native), 31)
            self.assertIn("native-after-warmup.json", native)
            save(path / "native-extra.json", {"unapproved": True})
            with self.assertRaisesRegex(ValueError, "all 30 native"):
                uploader.validate_configuration(root, "peagle")

    def test_catalogue_preserves_measured_and_unavailable_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            catalogue(root)
            record = uploader.validate_catalogue(root)
            self.assertEqual(len(record["cells"]), 2)
            self.assertEqual(record["cells"][0]["value"], 4.0)
            self.assertIsNone(record["cells"][1]["value"])
            self.assertTrue(record["cells"][1]["reason"])
            sdk = SimpleNamespace(Table=lambda **kwargs: kwargs)
            run = SimpleNamespace(summary={})
            uploader.save_catalogue_table(sdk, run, record)
            self.assertEqual(run.summary["unavailable_cell_count"], 1)
            self.assertEqual(run.summary["metric_catalogue"]["data"][1][6], None)
            self.assertEqual(run.summary["metric_catalogue"]["data"][1][8], 0)
            self.assertIn(root / "eagle3" / "01-gsm8k.json", record["paths"])

    def test_legacy_method_labels_validate_without_rewriting_immutable_exports(self):
        for label in uploader.METHOD_LABEL_ALIASES:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                output = catalogue(root)
                result_path = output / "team-trial-results.json"
                evidence_path = output / "team-measurement-evidence.json"
                result, evidence = json.loads(result_path.read_text()), json.loads(evidence_path.read_text())
                result["cells"][0]["method"] = label
                evidence["cells"][0]["key"]["method"] = label
                save(result_path, result)
                save(evidence_path, evidence)
                original = result_path.read_bytes()
                record = uploader.validate_catalogue(root)
                self.assertEqual(record["cells"][1]["method"], label)
                self.assertEqual(result_path.read_bytes(), original)

    def test_catalogue_rejects_duplicate_cells_under_old_and_new_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = catalogue(root)
            result_path = output / "team-trial-results.json"
            evidence_path = output / "team-measurement-evidence.json"
            result, evidence = json.loads(result_path.read_text()), json.loads(evidence_path.read_text())
            duplicate_cell = dict(result["cells"][0], method="Team FFN 100% (pilot)")
            duplicate_detail = dict(evidence["cells"][0])
            duplicate_detail["key"] = dict(duplicate_detail["key"], method=duplicate_cell["method"])
            result["cells"].append(duplicate_cell)
            evidence["cells"].append(duplicate_detail)
            save(result_path, result)
            save(evidence_path, evidence)
            with self.assertRaisesRegex(ValueError, "equivalent method labels"):
                uploader.validate_catalogue(root)

    def test_catalogue_rejects_changed_sources_and_stale_complete_run_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            catalogue(root)
            with self.assertRaisesRegex(ValueError, "predates complete primary results"):
                uploader.validate_catalogue(root, records=[{"configuration": "eagle3"}])
            (root / "eagle3" / "01-gsm8k.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "differs from its evidence manifest"):
                uploader.validate_catalogue(root)

    def test_final_report_and_provenance_join_catalogue_without_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            catalogue(root)
            for relative in uploader.CATALOGUE_SUPPLEMENTS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("A final report fixture.")
            secret = root / "report/credentials.json"
            secret.write_text("Excluded fixture.")
            record = uploader.validate_catalogue(root)
            self.assertTrue({root / name for name in uploader.CATALOGUE_SUPPLEMENTS} <= set(record["paths"]))
            self.assertNotIn(secret, record["paths"])
            report = root / "report/benchmark_report.pdf"
            report.unlink()
            report.symlink_to(secret)
            with self.assertRaisesRegex(ValueError, "regular paths"):
                uploader.validate_catalogue(root)

    def test_catalogue_sources_reject_credentials_and_path_traversal(self):
        probe = "diagnostics/attempts/eagle3-20261122T221003994481Z/logit-probe/"
        self.assertTrue(uploader.approved_catalogue_source(Path(probe + "humaneval-source-index-2.json")))
        self.assertTrue(uploader.approved_catalogue_source(Path(probe + "status.json")))
        self.assertFalse(uploader.approved_catalogue_source(Path(probe + "credentials.json")))
        self.assertFalse(uploader.approved_catalogue_source(Path(probe.replace("eagle3-", "peagle-") + "status.json")))
        self.assertFalse(uploader.approved_catalogue_source(Path(probe.replace("221003994481Z", "untrusted") + "status.json")))
        self.assertTrue(uploader.approved_catalogue_source(Path("gpu-memory-snapshot-20261122T2137.jsonl")))
        self.assertTrue(uploader.approved_catalogue_source(Path("gpu-memory-snapshot-20261122T213700Z.jsonl")))
        self.assertFalse(uploader.approved_catalogue_source(Path("gpu-memory-snapshot-credentials.jsonl")))
        self.assertTrue(uploader.approved_catalogue_source(Path("allocations/14056875.json")))
        self.assertFalse(uploader.approved_catalogue_source(Path("allocations/credentials.json")))
        self.assertFalse(uploader.approved_catalogue_source(Path("allocations/14056875/credentials.json")))
        self.assertTrue(uploader.approved_catalogue_source(Path("diagnostics/attempts/peagle-20261122T180530123456Z/trace/phase-123-abc.jsonl")))
        self.assertTrue(uploader.approved_catalogue_source(Path("diagnostics/team-ffn-100/team-rounds-123.jsonl")))
        self.assertFalse(uploader.approved_catalogue_source(Path("diagnostic-runtime/vllm/phase-123-abc.jsonl")))
        self.assertFalse(uploader.approved_catalogue_source(Path("unrelated/phase-123-abc.jsonl")))
        self.assertFalse(uploader.approved_catalogue_source(Path("diagnostics/attempts/untrusted/trace/phase-123-abc.jsonl")))
        self.assertFalse(uploader.approved_catalogue_source(Path("eagle3/credentials.json")))
        self.assertFalse(uploader.approved_catalogue_source(Path("../eagle3/run.json")))
        self.assertFalse(uploader.approved_catalogue_source(Path("/tmp/run.json")))

    def test_changed_evidence_requires_a_separate_version_and_preserves_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            record = {"configuration": "dflash", "evidence_sha256": "a" * 64}
            uploader.assign_run_identity(root, record, "v1")
            old_id = record["run_id"]
            receipt = uploader.read_receipt(root, [record])
            receipt["runs"]["dflash"] = {"run_id": old_id, "evidence_sha256": "a" * 64, "status": "uploaded"}
            uploader.save_receipt(root, receipt)
            original = (root / "wandb-upload.json").read_bytes()
            record["evidence_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "new --evidence-version"):
                uploader.read_receipt(root, [record])
            uploader.assign_run_identity(root, record, "final")
            self.assertNotEqual(record["run_id"], old_id)
            revised = uploader.read_receipt(root, [record], "final")
            uploader.save_receipt(root, revised)
            self.assertTrue((root / "wandb-upload-final.json").is_file())
            self.assertEqual((root / "wandb-upload.json").read_bytes(), original)

    def test_sdk_settings_disable_automatic_capture_in_the_pinned_version(self):
        sdk = SimpleNamespace(__version__="0.30.0", Settings=mock.Mock())
        uploader.upload_settings(sdk)
        options = sdk.Settings.call_args.kwargs
        for name in ("disable_code", "disable_git", "x_disable_meta", "x_disable_stats", "x_disable_machine_info"):
            self.assertTrue(options[name])
        self.assertFalse(options["save_code"])
        self.assertEqual(options["console"], "off")
        self.assertEqual(options["capture_loggers"], {})
        sdk.__version__ = "0.29.0"
        with self.assertRaisesRegex(ValueError, "0.30.0"):
            uploader.upload_settings(sdk)


if __name__ == "__main__":
    unittest.main()
