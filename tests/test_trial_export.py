"""Check evidence-based Site cells and publication table exports."""

import json
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import trial_export as export
from trial_recorder import block_metrics


class TrialExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write(self, relative, value, lines=False):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(json.dumps(row) for row in value) + "\n" if lines else json.dumps(value)
        path.write_text(content, encoding="utf-8")

    def metadata(self, configuration, diagnostic=False):
        backend, reference = (export.TEAM_CONFIGURATIONS if configuration.startswith("team-") else export.CONFIGURATIONS)[configuration]
        meta = {
            "configuration": configuration, "backend": backend, "reference": reference,
            "prepared_sha256": "a" * 64, "target_revision": export.TARGET_REVISION,
            "input_sha256": export.INPUT_SHA256, "repetitions": 3, "temperature": 0,
            "max_new_tokens": 2048, "model": "hpml-target", "gpu_uuid": "GPU-fixture",
            "gpu_record": "GPU-fixture, NVIDIA L40S, driver", "slurm_job_id": "41", "hostname": "compute",
            "server_settings": {},
        }
        if configuration.startswith("team-"):
            meta["phase"] = "diagnostic" if diagnostic else "primary"
        directory = f"diagnostics/{configuration}" if diagnostic else configuration
        if diagnostic:
            meta["server_settings"]["diagnostic"] = {
                "scope": "separate_diagnostic",
                "output_results_relative": f"{self.root.name}/traces/{configuration}",
            }
        self.write(f"{directory}/run.json", meta)
        return meta

    def block(self, configuration, repetition, elapsed=3, diagnostic=False):
        directory = f"diagnostics/{configuration}" if diagnostic else configuration
        records = []
        duration = round(elapsed / 3 * 1e9)
        first = repetition * 100_000_000_000
        for index in range(3):
            start = first + index * duration
            end = start + duration
            content = [start + duration // 4, start + duration // 2]
            latency = duration / 1e9
            ttft = (content[0] - start) / 1e9
            gap = (content[1] - content[0]) / 1e9
            records.append({
                "prompt_id": f"gsm8k-{index}", "benchmark": "gsm8k",
                "request_id": f"{directory}-{repetition}-{index}",
                "input_token_ids": [10, index], "output_token_ids": [100 + index, 200 + index],
                "native_output_token_ids": [100 + index, 200 + index, 128009],
                "output_tokens": 2, "output_text": "answer", "finish_reason": "stop",
                "native_stop_reason": 128009, "submitted_ns": start, "completed_ns": end,
                "first_content_ns": content[0], "events": [
                    {"received_ns": content[0], "text": "ans"},
                    {"received_ns": content[1], "text": "wer"},
                ],
                "end_to_end_seconds": latency, "ttft_seconds": ttft,
                "tpot_seconds": latency - ttft, "output_rate": 2 / latency,
                "inter_chunk_seconds": [gap], "max_inter_chunk_seconds": gap,
                "server_metadata": None,
            })
        block = {"benchmark": "gsm8k", "repetition": repetition, "requests": records,
                 "started_ns": first, "ended_ns": records[-1]["completed_ns"],
                 "metrics": block_metrics(records, (records[-1]["completed_ns"] - first) / 1e9)}
        self.write(f"{directory}/{repetition:02d}-gsm8k.json", block)
        return block

    @staticmethod
    def cell(result, metric, method="EAGLE-3", variant="", position=None):
        return next(row for row in result["cells"] if row["metricId"] == metric
                    and row["method"] == method and row["benchmark"] == "GSM8K"
                    and row["variant"] == variant and row["position"] == position)

    def test_speedup_pairs_repetitions_and_timing_uses_milliseconds(self):
        for configuration in ("eagle3", "ar-sglang"):
            self.metadata(configuration)
        self.block("eagle3", 1, elapsed=3)
        self.block("eagle3", 2, elapsed=3)
        self.block("ar-sglang", 1, elapsed=6)
        self.block("ar-sglang", 2, elapsed=3)
        result, manifest = export.Exporter(self.root).export()
        speedup = self.cell(result, "metric-speedup")
        self.assertEqual(speedup["value"], 1.5)
        self.assertEqual(speedup["sd"], statistics.stdev([2, 1]))
        self.assertEqual(speedup["n"], 2)
        self.assertAlmostEqual(self.cell(result, "metric-ttft", variant="Median")["value"], 250)
        self.assertEqual(self.cell(result, "metric-quality")["value"], 100)
        self.assertIn("ar-sglang/01-gsm8k.json", speedup["evidence"])
        self.assertEqual(manifest["primary_completed_blocks"], 4)

    def test_current_scope_defers_two_configurations_without_a_failure(self):
        self.metadata("dspark")
        self.block("dspark", 1)
        result, manifest = export.Exporter(self.root).export()
        self.assertFalse(any(cell["method"] in {"DSpark", "AR · vLLM ModelRunnerV2"} for cell in result["cells"]))
        self.assertEqual(manifest["primary_expected_blocks"], 150)
        self.assertEqual(manifest["primary_expected_timed_requests"], 450)
        self.assertEqual(manifest["deferred_expected_timed_requests"], 180)
        self.assertEqual(manifest["primary_completed_blocks"], 0)
        resumed, resumed_manifest = export.Exporter(self.root, include_deferred=True).export()
        self.assertEqual(self.cell(resumed, "metric-otps", method="DSpark")["value"], 2)
        self.assertEqual(resumed_manifest["primary_expected_blocks"], 210)
        self.assertEqual(resumed_manifest["deferred_configurations"], [])

    def test_one_repetition_has_no_fabricated_standard_deviation(self):
        self.metadata("ar-sglang")
        self.block("ar-sglang", 1)
        result, _ = export.Exporter(self.root).export()
        cell = self.cell(result, "metric-otps", method="AR · SGLang")
        self.assertEqual(cell["value"], 2)
        self.assertEqual(cell["n"], 1)
        self.assertIsNone(cell["sd"])
        keys = [(row["metricId"], row["variant"], row["method"], row["benchmark"], row["position"]) for row in result["cells"]]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(row["method"] in export.METHODS.values() for row in result["cells"]))
        tables = export.latex_tables(result, export.catalogue(export.CATALOGUE))
        self.assertEqual(tables.count(r"\begin{table*}"), 4)
        self.assertNotIn(r"\documentclass", tables)
        self.assertNotIn(r"\resizebox", tables)
        team_rows = [line for line in tables.splitlines() if line.startswith("Pilot FFN")]
        self.assertEqual(len(team_rows), 36)
        self.assertTrue(all(not field.strip() for line in team_rows for field in line.removesuffix(r" \\").split(" & ")[1:]))

    def test_appendix_preserves_zero_values_and_omits_unmeasured_variants(self):
        for configuration in ("eagle3", "ar-sglang"):
            self.metadata(configuration)
            block = self.block(configuration, 1)
            if configuration == "eagle3":
                for request in block["requests"]:
                    request["output_token_ids"][0] += 10
                    request["native_output_token_ids"][0] += 10
                self.write("eagle3/01-gsm8k.json", block)
        exporter = export.Exporter(self.root)
        result, _ = exporter.export()
        appendix = export.latex_appendix_tables(result, exporter.metrics)
        self.assertIn(r"\(0\;[1]\)", appendix)
        self.assertIn("EAGLE-3", appendix)
        self.assertNotIn("Framework-allocated memory", appendix)
        self.assertNotIn("Pilot FFN", appendix)
        self.assertNotIn(r"\pm", appendix)

    def test_findings_distinguish_unique_prompts_and_repeated_executions(self):
        for configuration in ("eagle3", "ar-sglang"):
            self.metadata(configuration)
            for repetition in (1, 2, 3):
                block = self.block(configuration, repetition)
                if configuration == "eagle3":
                    request = block["requests"][0]
                    request["output_token_ids"][0] += 10
                    request["native_output_token_ids"][0] += 10
                    request["output_token_ids"].append(300)
                    request["native_output_token_ids"].insert(-1, 300)
                    request["output_tokens"] = 3
                    request["output_rate"] = 3 / request["end_to_end_seconds"]
                    request["tpot_seconds"] = (request["end_to_end_seconds"] - request["ttft_seconds"]) / 2
                    block["metrics"] = block_metrics(block["requests"], (block["ended_ns"] - block["started_ns"]) / 1e9)
                    self.write(f"eagle3/{repetition:02d}-gsm8k.json", block)
        result, manifest = export.Exporter(self.root).export()
        team, team_manifest = export.Exporter(self.root, team=True).export()
        findings = export.latex_findings(result, manifest, team, team_manifest)
        self.assertIn("6 of 9 checked request executions", findings)
        self.assertIn("2 of 3 distinct prompts", findings)
        self.assertIn("returned token 1", findings)
        self.assertIn("Their cause remains unresolved", findings)
        self.assertIn("Output lengths match in every observed repetition for 2 of 3 distinct prompts", findings)
        self.assertIn("the method delivers 7 tokens and its AR reference delivers 6", findings)
        quality = next(cell for cell in manifest["cells"] if cell["key"]["metricId"] == "metric-quality"
                       and cell["key"]["method"] == "EAGLE-3" and cell["observations"])
        check = quality["observations"][0]["agreement_checks"][0]
        self.assertEqual((check["method_output_tokens"], check["reference_output_tokens"]), (3, 2))

    def test_acceptance_caption_states_tree_and_linear_denominators(self):
        cell = {"metricId": "metric-draft-fraction", "variant": "", "method": "EAGLE-3",
                "benchmark": "GSM8K", "position": None, "value": 20, "sd": None, "n": 1, "unit": "%"}
        appendix = export.latex_appendix_tables({"cells": [cell]}, export.catalogue(export.CATALOGUE))
        self.assertIn("15 candidate tree nodes", appendix)
        self.assertIn("nine linear draft tokens", appendix)
        self.assertIn("seven linear draft tokens", appendix)
        self.assertIn("candidates on alternative branches", appendix)

    def test_best_means_use_direction_displayed_ties_and_denominator_scope(self):
        def cell(method, metric, value):
            return {"metricId": metric, "variant": "", "method": method, "benchmark": "GSM8K",
                    "position": None, "value": value, "sd": 0.02, "n": 3, "unit": "unit"}
        throughput = [cell("EAGLE-3", "metric-otps", 100), cell("DFlash", "metric-otps", 110.011),
                      cell("P-EAGLE", "metric-otps", 110.014)]
        winners = export.best_observed_cells(throughput)
        self.assertEqual(winners, {export.cell_identity(row) for row in throughput[1:]})
        latency = [cell("EAGLE-3", "metric-ttft", 20), cell("DFlash", "metric-ttft", 25),
                   cell("P-EAGLE", "metric-ttft", 15)]
        self.assertEqual(export.best_observed_cells(latency), {export.cell_identity(latency[2])})
        acceptance = [cell("EAGLE-3", "metric-draft-fraction", 99), cell("DFlash", "metric-draft-fraction", 50),
                      cell("P-EAGLE", "metric-draft-fraction", 60)]
        self.assertEqual(export.best_observed_cells(acceptance), {export.cell_identity(acceptance[2])})
        self.assertEqual(export.best_observed_cells([throughput[0]]), set())
        component = [cell("EAGLE-3", "metric-component-time", 10), cell("DFlash", "metric-component-time", 5)]
        self.assertEqual(export.best_observed_cells(component), set())
        self.assertIn(r"\mathbf{110.01}", export.latex_value(throughput[1], True))
        self.assertNotIn(r"\mathbf{0.02}", export.latex_value(throughput[1], True))
        self.assertEqual(export.mean_display(cell("EAGLE-3", "metric-otps", 110.293)), "110.29")
        self.assertEqual(export.mean_display(cell("DFlash", "metric-otps", 109.731)), "109.73")

    def test_team_timing_and_memory_survive_an_unavailable_ar_reference(self):
        self.metadata("team-ffn-100")
        block = self.block("team-ffn-100", 1)
        self.write("gpu-memory.jsonl", [{"slurm_job_id": "41", "hostname": "compute", "gpu_uuid": "GPU-fixture",
                                        "monotonic_ns": tick, "memory_used_mib": 8192}
                                       for tick in (block["started_ns"] - 1, block["started_ns"] + 1, block["ended_ns"] + 1)], lines=True)
        result, manifest = export.Exporter(self.root, team=True).export()
        method = export.TEAM_METHODS["team-ffn-100"]
        self.assertEqual(self.cell(result, "metric-otps", method)["value"], 2)
        self.assertEqual(self.cell(result, "metric-memory", method, "Sampled whole-device memory")["value"], 8)
        for metric in ("metric-speedup", "metric-quality"):
            item = self.cell(result, metric, method)
            self.assertIsNone(item["value"])
            self.assertIn("requires both recorded model configurations", item["reason"])
        primary, primary_manifest = export.Exporter(self.root).export()
        findings = export.latex_findings(primary, primary_manifest, result, manifest)
        self.assertIn("Team pilot observations", findings)
        self.assertIn("0 of these 1 benchmarks", findings)
        self.assertIn("measurements for 1 of the three team widths", findings)

    def test_generated_report_refresh_preserves_authored_source(self):
        self.metadata("ar-sglang")
        self.block("ar-sglang", 1)
        report = self.root / "report"
        report.mkdir()
        (report / "report.tex").write_text("Authored source.")
        (report / "tables.tex").write_text("An earlier generated table.")
        export.write_export(export.Exporter(self.root), self.root / "export", report)
        self.assertEqual((report / "report.tex").read_text(), "Authored source.")
        self.assertIn(r"\begin{table*}", (report / "tables.tex").read_text())
        self.assertTrue((report / "results-summary.tex").is_file())

    def test_paper_number_format_preserves_significant_digits(self):
        self.assertEqual(export.latex_number(1374.8, 3), "1370")
        self.assertEqual(export.latex_number(0.003649, 2), "0.0036")
        self.assertEqual(export.latex_number(0, 2), "0")
        self.assertEqual(export.latex_number(0.00000123, 2), r"1.2\times 10^{-6}")

    def test_team_origin_comes_from_saved_metadata(self):
        metadata = self.metadata("team-ffn-100")
        metadata["artifact_origin"] = "teammate-original"
        self.write("team-ffn-100/run.json", metadata)
        team, team_manifest = export.Exporter(self.root, team=True).export()
        primary, manifest = export.Exporter(self.root).export()
        origin = team_manifest["configuration_provenance"][0]
        self.assertEqual(origin["artifact_origin"], "teammate-original")
        self.assertEqual(origin["source"], "team-ffn-100/run.json")
        findings = export.latex_findings(primary, manifest, team, team_manifest)
        self.assertIn("uses an imported original team checkpoint", findings)
        self.assertNotIn("newly trained local reproduction checkpoint", findings)

    def test_corrupt_counts_and_mismatched_gpu_leave_cells_empty(self):
        for configuration in ("eagle3", "ar-sglang"):
            self.metadata(configuration)
            self.block(configuration, 1)
        meta = json.loads((self.root / "ar-sglang/run.json").read_text())
        meta["gpu_uuid"] = "GPU-other"
        meta["gpu_record"] = "GPU-other, NVIDIA L40S, driver"
        self.write("ar-sglang/run.json", meta)
        result, _ = export.Exporter(self.root).export()
        self.assertIsNone(self.cell(result, "metric-speedup")["value"])
        self.assertIn("gpu_uuid", self.cell(result, "metric-speedup")["reason"])
        block = json.loads((self.root / "eagle3/01-gsm8k.json").read_text())
        block["metrics"]["otps"] = 999
        self.write("eagle3/01-gsm8k.json", block)
        result, _ = export.Exporter(self.root).export()
        cell = self.cell(result, "metric-otps")
        self.assertIsNone(cell["value"])
        self.assertEqual(cell["n"], 0)
        self.assertIn("differs", cell["reason"])

    def test_tau_pools_rounds_and_preserves_tokens_before_trimming(self):
        self.metadata("eagle3")
        self.block("eagle3", 1)
        self.metadata("eagle3", diagnostic=True)
        block = self.block("eagle3", 1, diagnostic=True)
        rounds = []
        for index, request in enumerate(block["requests"]):
            native = request["native_output_token_ids"]
            committed = [native] if index != 1 else [native[:1], native[1:]]
            request["server_metadata"] = {"spec_verify_ct": len(committed)}
            offset = 0
            for number, tokens in enumerate(committed, 1):
                rounds.append({
                    "scope": "separate_diagnostic", "configuration": "eagle3",
                    "request_id": request["request_id"], "round": number,
                    "proposed_draft_tokens": 3, "accepted_draft_tokens": len(tokens) - 1,
                    "retained_target_tokens": 1, "committed_tokens": len(tokens),
                    "committed_token_ids": tokens, "previous_output_count": offset,
                    "depth_metrics_available": False, "proposed_depth": None, "accepted_prefix_depth": None,
                })
                offset += len(tokens)
        self.write("diagnostics/eagle3/01-gsm8k.json", block)
        self.write("traces/eagle3/sglang-rounds-1.jsonl", rounds, lines=True)
        result, manifest = export.Exporter(self.root).export()
        tau = self.cell(result, "metric-acceptance-length")
        self.assertEqual(tau["value"], 9 / 4)
        self.assertEqual(self.cell(result, "metric-acceptance-length-request")["value"], (3 + 1.5 + 3) / 3)
        self.assertEqual(self.cell(result, "metric-returned-round")["value"], 6 / 4)
        self.assertIsNone(self.cell(result, "metric-prefix-survival", position=1)["value"])
        detail = next(row for row in manifest["cells"] if row["key"]["metricId"] == "metric-acceptance-length"
                      and row["key"]["method"] == "EAGLE-3" and row["key"]["benchmark"] == "GSM8K")
        self.assertEqual(detail["observations"][0]["verification_rounds"], 4)
        self.assertEqual(detail["observations"][0]["committed_tokens"], 9)

    @staticmethod
    def tree_capture(root=50, first=100, bonus=200, leaf=True):
        tokens = [root, 77, first, 88] if leaf else [root, first, 77, 88]
        raw = lambda value: {"value": value}
        before = {"raw_tree": {
            "draft_token_num": raw(4), "draft_token": raw(tokens),
            "positions": raw([8, 9, 9, 10]), "retrieve_index": raw([[0, 1, 2, 3]]),
            "retrieve_next_token": raw([[1, 3, -1, -1]]),
            "retrieve_next_sibling": raw([[-1, 2, -1, -1]]),
        }}
        after = {"result_type": "GenerationBatchResult", "raw_result": {
            "next_token_ids": raw([first, bonus, 0, 0]), "accept_lens": raw([2]),
        }}
        row = {"proposed_draft_tokens": 3, "accepted_draft_tokens": 1,
               "retained_target_tokens": 1, "committed_tokens": 2,
               "committed_token_ids": [first, bonus], "previous_output_tail": [root]}
        return before, after, row

    def test_tree_leaf_does_not_supply_a_conditional_proposal_on_another_branch(self):
        before, after, row = self.tree_capture()
        result = export.eagle_tree_depths(before, after, row)
        self.assertEqual(result["proposed_depth"], 2)
        self.assertEqual(result["accepted_prefix_depth"], 1)
        self.assertEqual(result["accepted_node_indices"], [2])
        self.assertEqual(result["conditional_eligible_positions"], [1])
        self.assertTrue(result["accepted_path_ends_at_leaf"])
        before, after, row = self.tree_capture(leaf=False)
        result = export.eagle_tree_depths(before, after, row)
        self.assertEqual(result["conditional_eligible_positions"], [1, 2])
        self.assertFalse(result["accepted_path_ends_at_leaf"])
        row.update(accepted_draft_tokens=2, committed_tokens=3, committed_token_ids=[100, 88, 200])
        after["raw_result"]["accept_lens"]["value"] = [3]
        after["raw_result"]["next_token_ids"]["value"] = [100, 88, 200, 0]
        result = export.eagle_tree_depths(before, after, row)
        self.assertEqual(result["accepted_prefix_depth"], 2)
        self.assertEqual(result["accepted_node_indices"], [1, 3])

    def test_tree_adapter_rejects_ambiguous_incomplete_and_inconsistent_capture(self):
        changes = [
            lambda before, after, row: before["raw_tree"]["draft_token"].update(omitted="Capture limit."),
            lambda before, after, row: before["raw_tree"]["draft_token"]["value"].__setitem__(1, 100),
            lambda before, after, row: before["raw_tree"]["retrieve_next_sibling"]["value"][0].__setitem__(2, 1),
            lambda before, after, row: before["raw_tree"]["positions"]["value"].__setitem__(3, 11),
            lambda before, after, row: after["raw_result"]["accept_lens"].update(value=[3]),
            lambda before, after, row: after["raw_result"]["next_token_ids"]["value"].__setitem__(0, 99),
            lambda before, after, row: row.update(previous_output_tail=[999]),
        ]
        for change in changes:
            with self.subTest(change=change):
                capture = self.tree_capture()
                change(*capture)
                with self.assertRaises(export.EvidenceError):
                    export.eagle_tree_depths(*capture)
        before, after, row = self.tree_capture(leaf=False, bonus=88)
        with self.assertRaisesRegex(export.EvidenceError, "unaccepted child"):
            export.eagle_tree_depths(before, after, row)

    def tree_fixture(self):
        self.metadata("eagle3")
        self.block("eagle3", 1)
        meta = self.metadata("eagle3", diagnostic=True)
        block = self.block("eagle3", 1, diagnostic=True)
        rounds, captures = [], []
        for index, request in enumerate(block["requests"]):
            request["server_metadata"] = {"spec_verify_ct": 1}
            before, after, row = self.tree_capture(root=request["native_output_token_ids"][0],
                                                  first=request["native_output_token_ids"][1],
                                                  bonus=128009, leaf=index == 0)
            row.update(scope="separate_diagnostic", configuration="eagle3", request_id=request["request_id"],
                       round=1, native_verification_count=1, previous_output_count=1)
            rounds.append(row)
            shared = {key: meta[key] for key in ("configuration", "hostname", "gpu_uuid", "slurm_job_id")}
            shared.update(request_ids=[request["request_id"]], diagnostic_only=True, clock="time.perf_counter_ns",
                          clock_implementation="fixture monotonic clock", observation_id=f"observation-{index}",
                          capture_id=f"capture-{index}", verification_round_indices={request["request_id"]: 1})
            for offset, (record, event) in enumerate(((before, "eagle_tree_before_verify"), (after, "eagle_verification_result"))):
                record.update(shared, event=event, cpu_capture_start_ns=request["submitted_ns"] + 100 + offset * 100,
                              cpu_capture_end_ns=request["submitted_ns"] + 150 + offset * 100)
                captures.append(record)
        self.write("diagnostics/eagle3/01-gsm8k.json", block)
        self.write("traces/eagle3/sglang-rounds-1.jsonl", rounds, lines=True)
        self.write("traces/eagle3/phase-1.jsonl", captures, lines=True)
        return captures

    def test_tree_variant_preserves_branch_denominators_and_source_evidence(self):
        self.tree_fixture()
        result, manifest = export.Exporter(self.root).export()
        survival = self.cell(result, "metric-prefix-survival", variant=export.TREE_VARIANT, position=2)
        conditional = self.cell(result, "metric-conditional-acceptance", variant=export.TREE_VARIANT, position=2)
        self.assertEqual(survival["value"], 0)
        self.assertEqual(conditional["value"], 0)
        self.assertEqual(conditional["n"], 1)
        self.assertIsNone(self.cell(result, "metric-prefix-survival", position=1)["value"])
        self.assertIn("separate Tree-prefix diagnostic", self.cell(result, "metric-prefix-survival", position=1)["reason"])
        details = [cell for cell in manifest["cells"] if cell["key"]["variant"] == export.TREE_VARIANT
                   and cell["key"]["benchmark"] == "GSM8K" and cell["key"]["position"] == 2]
        self.assertEqual([cell["observations"][0]["eligible_rounds"] for cell in details], [3, 2])
        self.assertTrue(all(cell["observations"][0]["excluded_leaf_rounds"] == 1 for cell in details))
        self.assertIn("traces/eagle3/phase-1.jsonl", survival["evidence"])
        appendix = export.latex_appendix_tables(result, export.catalogue(export.CATALOGUE))
        self.assertIn("A leaf has no eligible next proposal", appendix)
        self.assertIn("chain and defines n by the draft feature history", appendix)

    def test_tree_variant_requires_every_joined_round_and_matching_hardware(self):
        captures = self.tree_fixture()
        corruptions = [captures[:-1], captures + [dict(captures[0], capture_id="different")],
                       [dict(row, gpu_uuid="other") for row in captures],
                       [dict(row, verification_round_indices={row["request_ids"][0]: 2}) for row in captures]]
        for records in corruptions:
            self.write("traces/eagle3/phase-1.jsonl", records, lines=True)
            result, _ = export.Exporter(self.root).export()
            cell = self.cell(result, "metric-prefix-survival", variant=export.TREE_VARIANT, position=1)
            self.assertIsNone(cell["value"])
            self.assertEqual(cell["n"], 0)

    @staticmethod
    def phase_record(meta, request, identity, start, end, sync, components):
        origin = request["submitted_ns"]
        clock = {"clock": "time.perf_counter_ns", "clock_implementation": "fixture monotonic clock"}
        def interval(name, sequence, left, right):
            return {"name": name, "sequence": sequence, "parent_sequence": 0,
                    "cpu_start_ns": origin + left * 1_000_000, "cpu_end_ns": origin + right * 1_000_000,
                    "gpu_elapsed_seconds": (right - left) / 1000, "device_index": 0, "stream_id": 1}
        return {**{key: meta[key] for key in ("configuration", "hostname", "gpu_uuid", "slurm_job_id")},
                **clock, "event": "cuda_phase_observation", "observation_id": identity,
                "request_ids": [request["request_id"]], "diagnostic_only": True,
                "outer": interval("worker_iteration", 0, start, end),
                "phases": [interval(name, index + 1, left, right) for index, (name, left, right) in enumerate(components)],
                "synchronized_cpu_end_ns": origin + sync * 1_000_000,
                "memory": {"device_synchronized_before_read": True}}

    def sglang_timing_fixture(self):
        captures = self.tree_fixture()
        meta = json.loads((self.root / "diagnostics/eagle3/run.json").read_text())
        block = json.loads((self.root / "diagnostics/eagle3/01-gsm8k.json").read_text())
        rounds = [json.loads(line) for line in (self.root / "traces/eagle3/sglang-rounds-1.jsonl").read_text().splitlines()]
        observations = []
        for index, (request, row) in enumerate(zip(block["requests"], rounds)):
            request["server_metadata"].update(spec_num_correct_drafts=1, spec_num_proposed_drafts=3)
            record = self.phase_record(meta, request, f"timing-{index}", 10, 45, 50,
                                       [("draft_proposal", 10, 20), ("target_verification", 20, 40)])
            record["verification_round_indices"] = {request["request_id"]: 1}
            observations.append(record)
            row.update({key: record[key] for key in ("hostname", "gpu_uuid", "slurm_job_id", "clock", "clock_implementation")})
            row.update(state_update_completed=True, state_update_boundary="The scheduler completes the state update.",
                       state_update_completed_ns=request["submitted_ns"] + 55_000_000,
                       native_postupdate_token_ids=row["committed_token_ids"])
        self.write("diagnostics/eagle3/01-gsm8k.json", block)
        self.write("traces/eagle3/sglang-rounds-1.jsonl", rounds, lines=True)
        self.write("traces/eagle3/phase-1.jsonl", captures + observations, lines=True)
        return captures, observations, rounds

    def test_sglang_component_times_require_synchronized_complete_rounds(self):
        self.sglang_timing_fixture()
        result, manifest = export.Exporter(self.root).export()
        for variant, expected in (("Round latency", 45), ("Draft latency", 10), ("Verification latency", 20)):
            cell = self.cell(result, "metric-component-time", variant=variant)
            self.assertAlmostEqual(cell["value"], expected)
            self.assertEqual(cell["n"], 1)
        details = next(cell for cell in manifest["cells"] if cell["key"]["metricId"] == "metric-component-time"
                       and cell["key"]["method"] == "EAGLE-3" and cell["key"]["benchmark"] == "GSM8K")
        self.assertEqual(details["observations"][0]["verification_rounds"], 3)

    def test_component_times_reject_missing_rounds_incomplete_updates_and_counter_mismatches(self):
        for issue in ("missing", "update", "clock", "synchronization", "counter"):
            with self.subTest(issue=issue):
                captures, observations, rounds = self.sglang_timing_fixture()
                if issue == "missing":
                    observations.pop()
                elif issue == "update":
                    rounds[0]["state_update_completed"] = False
                elif issue == "clock":
                    rounds[0]["clock_implementation"] = "another clock"
                elif issue == "synchronization":
                    observations[0]["synchronized_cpu_end_ns"] = rounds[0]["state_update_completed_ns"] + 1
                else:
                    path = "diagnostics/eagle3/01-gsm8k.json"
                    block = json.loads((self.root / path).read_text())
                    block["requests"][0]["server_metadata"]["spec_num_correct_drafts"] = 2
                    self.write(path, block)
                self.write("traces/eagle3/sglang-rounds-1.jsonl", rounds, lines=True)
                self.write("traces/eagle3/phase-1.jsonl", captures + observations, lines=True)
                result, _ = export.Exporter(self.root).export()
                self.assertIsNone(self.cell(result, "metric-component-time", variant="Round latency")["value"])

    def peagle_timing_fixture(self):
        self.metadata("peagle")
        self.block("peagle", 1)
        meta = self.metadata("peagle", diagnostic=True)
        block = self.block("peagle", 1, diagnostic=True)
        rounds, observations = [], []
        for index, request in enumerate(block["requests"]):
            first = self.phase_record(meta, request, f"prefill-{index}", 1, 18, 19, [("draft_proposal", 10, 15)])
            second = self.phase_record(meta, request, f"verify1-{index}", 20, 50, 55,
                                      [("verification_and_sampling", 20, 30), ("draft_proposal", 35, 40)])
            third = self.phase_record(meta, request, f"verify2-{index}", 70, 100, 105,
                                     [("verification_and_sampling", 70, 80), ("draft_proposal", 85, 90)])
            identities = request["request_id"]
            proposed = [[request["output_token_ids"][0], 999], [300, 301], [400, 401]]
            for step, record in enumerate((first, second, third)):
                record["scheduler_step_id"] = f"step-{index}-{step}"
                record["produced_drafts"] = {identities: {"token_ids": proposed[step], "row_index": 0}}
                if step:
                    producer = (first, second)[step - 1]
                    record["verification_proposals"] = {identities: {
                        "producer_observation_id": producer["observation_id"],
                        "producer_scheduler_step_id": producer["scheduler_step_id"], "producer_row_index": 0,
                        "token_ids": proposed[step - 1], "input_source": "previous_draft_tensor",
                        "tensor_identity_verified": True}}
            observations.extend((first, second, third))
            for number, record, offset, committed in ((1, second, 0, request["native_output_token_ids"][:2]),
                                                      (2, third, 2, request["native_output_token_ids"][2:])):
                rounds.append({**{key: record[key] for key in ("configuration", "hostname", "gpu_uuid", "slurm_job_id", "clock", "clock_implementation")},
                    "scope": "separate_diagnostic", "request_id": identities, "round": number,
                    "scheduler_step_id": record["scheduler_step_id"], "proposed_draft_tokens": 2,
                    "accepted_draft_tokens": len(committed) - 1, "retained_target_tokens": 1,
                    "committed_tokens": len(committed), "committed_token_ids": committed, "previous_output_count": offset,
                    "invalid_proposal_count": 0, "scheduled_proposal_contains_placeholders": True,
                    "scheduled_proposal_token_ids": [-1, -1], "state_update_completed": True,
                    "state_update_completed_ns": request["submitted_ns"] + (60 if number == 1 else 110) * 1_000_000,
                    "state_update_boundary": "The scheduler completes the state update."})
        self.write("traces/peagle/vllm-rounds-1.jsonl", rounds, lines=True)
        self.write("traces/peagle/phase-1.jsonl", observations, lines=True)
        snapshots = []
        for index in (0, 1):
            snapshots.append({"metrics_url": "http://127.0.0.1:31000/metrics", "hostname": meta["hostname"],
                              "gpu_uuid": meta["gpu_uuid"], "observed_completions": index * 3,
                              "observed_monotonic_ns": block["started_ns"] - 1 if index == 0 else block["ended_ns"] + 1,
                              "samples": [{"name": f"vllm:spec_decode_{name}_total", "labels": {}, "value": value * index}
                                          for name, value in (("num_drafts", 6), ("num_draft_tokens", 12), ("num_accepted_tokens", 3))]})
        self.write("diagnostics/peagle/native-01-gsm8k.json", {"benchmark": "gsm8k", "repetition": 1,
                                                             "before": snapshots[0], "after": snapshots[1]})
        return observations

    def test_peagle_links_drafting_to_later_verification_and_preserves_overlap(self):
        self.peagle_timing_fixture()
        result, manifest = export.Exporter(self.root).export()
        for variant, expected in (("Round latency", 62.5), ("Draft latency", 5), ("Verification latency", 10)):
            self.assertAlmostEqual(self.cell(result, "metric-component-time", method="P-EAGLE", variant=variant)["value"], expected)
        details = next(cell for cell in manifest["cells"] if cell["key"]["metricId"] == "metric-component-time"
                       and cell["key"]["variant"] == "Round latency" and cell["key"]["method"] == "P-EAGLE"
                       and cell["key"]["benchmark"] == "GSM8K")["observations"][0]
        self.assertEqual(details["overlapping_adjacent_round_pairs"], 3)
        self.assertEqual(details["verification_rounds"], 6)

    def test_peagle_round_times_reject_unverified_producer_links(self):
        observations = self.peagle_timing_fixture()
        link = observations[1]["verification_proposals"][observations[1]["request_ids"][0]]
        link["tensor_identity_verified"] = False
        self.write("traces/peagle/phase-1.jsonl", observations, lines=True)
        result, _ = export.Exporter(self.root).export()
        cell = self.cell(result, "metric-component-time", method="P-EAGLE", variant="Round latency")
        self.assertIsNone(cell["value"])
        self.assertIn("exact draft-producer linkage", cell["reason"])

    def test_memory_requires_matching_samples_and_allocation_ratios_stay_empty(self):
        meta = self.metadata("ar-sglang")
        block = self.block("ar-sglang", 1)
        self.write("gpu-memory.jsonl", [
            {"slurm_job_id": meta["slurm_job_id"], "hostname": meta["hostname"], "gpu_uuid": meta["gpu_uuid"],
             "monotonic_ns": block["started_ns"] - 1, "memory_used_mib": 16384},
            {"slurm_job_id": meta["slurm_job_id"], "hostname": meta["hostname"], "gpu_uuid": meta["gpu_uuid"],
             "monotonic_ns": block["started_ns"] + 1, "memory_used_mib": 8192},
            {"slurm_job_id": "another-job", "hostname": meta["hostname"], "gpu_uuid": meta["gpu_uuid"],
             "monotonic_ns": block["started_ns"] + 2, "memory_used_mib": 99999},
            {"slurm_job_id": meta["slurm_job_id"], "hostname": meta["hostname"], "gpu_uuid": meta["gpu_uuid"],
             "monotonic_ns": block["ended_ns"] + 1, "memory_used_mib": 16384},
        ], lines=True)
        self.write("ar-sglang/resources.json", {"whole_job": {"allocated_gpu_seconds_per_million_output_tokens": 1}})
        result, _ = export.Exporter(self.root).export()
        memory = self.cell(result, "metric-memory", method="AR · SGLang", variant="Sampled whole-device memory")
        self.assertEqual(memory["value"], 8)
        for metric in ("metric-gpu-time", "metric-per-gpu-throughput", "metric-actual-charge"):
            if any(cell["metricId"] == metric for cell in result["cells"]):
                self.assertIsNone(self.cell(result, metric, method="AR · SGLang")["value"])

    def test_explicit_memory_snapshot_excludes_live_samples_from_manifest(self):
        meta = self.metadata("ar-sglang")
        block = self.block("ar-sglang", 1)
        row = {"slurm_job_id": meta["slurm_job_id"], "hostname": meta["hostname"], "gpu_uuid": meta["gpu_uuid"],
               "monotonic_ns": block["started_ns"] + 1, "memory_used_mib": 8192}
        self.write("snapshots/memory.jsonl", [{**row, "monotonic_ns": tick} for tick in
                                             (block["started_ns"] - 1, block["started_ns"] + 1, block["ended_ns"] + 1)], lines=True)
        self.write("gpu-memory.jsonl", [{**row, "memory_used_mib": 16384}], lines=True)
        result, manifest = export.Exporter(self.root, memory_samples="snapshots/memory.jsonl").export()
        cell = self.cell(result, "metric-memory", method="AR · SGLang", variant="Sampled whole-device memory")
        self.assertEqual(cell["value"], 8)
        self.assertIn("snapshots/memory.jsonl", cell["evidence"])
        self.assertIn("snapshots/memory.jsonl", manifest["source_files"])
        self.assertNotIn("gpu-memory.jsonl", manifest["source_files"])
        self.assertEqual(manifest["memory_sample_file"], "snapshots/memory.jsonl")
        export.write_export(export.Exporter(self.root, memory_samples="snapshots/memory.jsonl"), self.root / "snapshot-export")
        team_manifest = json.loads((self.root / "snapshot-export/team-measurement-evidence.json").read_text())
        self.assertEqual(team_manifest["memory_sample_file"], "snapshots/memory.jsonl")
        with self.assertRaises(export.EvidenceError):
            export.Exporter(self.root, memory_samples="../outside.jsonl")

    def test_sampled_memory_requires_the_log_to_bracket_the_whole_block(self):
        meta = self.metadata("ar-sglang")
        block = self.block("ar-sglang", 1)
        base = {"slurm_job_id": meta["slurm_job_id"], "hostname": meta["hostname"], "gpu_uuid": meta["gpu_uuid"],
                "memory_used_mib": 8192}
        ticks = [block["started_ns"] + 1]
        for extra in (None, block["started_ns"] - 1, block["ended_ns"] + 1):
            if extra is not None: ticks.append(extra)
            self.write("gpu-memory.jsonl", [{**base, "monotonic_ns": tick} for tick in ticks], lines=True)
            result, _ = export.Exporter(self.root).export()
            cell = self.cell(result, "metric-memory", "AR · SGLang", "Sampled whole-device memory")
            if len(ticks) < 3:
                self.assertIsNone(cell["value"])
                self.assertIn("complete benchmark interval", cell["reason"])
            else:
                self.assertEqual(cell["value"], 8)

    def test_memory_snapshots_preserve_distinct_job_and_gpu_evidence(self):
        meta = self.metadata("team-ffn-100")
        block = self.block("team-ffn-100", 1)
        rows = [{"slurm_job_id": "41", "hostname": "compute", "gpu_uuid": "GPU-fixture",
                 "monotonic_ns": tick, "memory_used_mib": 8192}
                for tick in (block["started_ns"] - 1, block["started_ns"] + 1, block["ended_ns"] + 1)]
        self.write("gpu-memory-old.jsonl", [{**row, "slurm_job_id": "40", "gpu_uuid": "GPU-old", "memory_used_mib": 32768} for row in rows], lines=True)
        self.write("gpu-memory-new.jsonl", rows, lines=True)
        result, manifest = export.Exporter(self.root, team=True, memory_samples="gpu-memory-old.jsonl",
                                           additional_memory_samples=["gpu-memory-new.jsonl"]).export()
        cell = self.cell(result, "metric-memory", "Pilot FFN 100%", "Sampled whole-device memory")
        self.assertEqual(cell["value"], 8)
        self.assertEqual(manifest["memory_sample_files"], ["gpu-memory-old.jsonl", "gpu-memory-new.jsonl"])
        self.assertTrue(all(name in manifest["source_files"] for name in manifest["memory_sample_files"]))
        with self.assertRaises(export.EvidenceError):
            export.Exporter(self.root, memory_samples="gpu-memory-old.jsonl", additional_memory_samples=["gpu-memory-old.jsonl"])

    def test_timed_gpu_efficiency_uses_the_benchmark_interval_and_final_job_count(self):
        self.metadata("ar-sglang")
        self.block("ar-sglang", 1)
        allocation = {"job_id": "41", "gpu_count": 3, "elapsed_seconds": 3600,
                      "allocated_gpu_seconds": 10800, "state": "TIMEOUT",
                      "start": "2026-11-22T00:00:00", "end": "2026-11-22T01:00:00",
                      "source_command": ["sacct", "-X"], "raw_sacct": "saved parent record"}
        self.write("allocations/41.json", allocation)
        self.write("allocation.json", {**allocation, "job_id": "another-job"})
        result, manifest = export.Exporter(self.root).export()
        self.assertEqual(self.cell(result, "metric-gpu-time", "AR · SGLang")["value"], 1500000)
        self.assertAlmostEqual(self.cell(result, "metric-per-gpu-throughput", "AR · SGLang")["value"], 2 / 3)
        if any(cell["metricId"] == "metric-actual-charge" for cell in result["cells"]):
            self.assertIsNone(self.cell(result, "metric-actual-charge", "AR · SGLang")["value"])
        detail = next(cell for cell in manifest["cells"] if cell["key"]["metricId"] == "metric-gpu-time"
                      and cell["key"]["method"] == "AR · SGLang" and cell["key"]["benchmark"] == "GSM8K")
        self.assertEqual(detail["observations"][0]["timed_gpu_seconds"], 9)
        self.assertIn("allocations/41.json", manifest["source_files"])
        for change in ({"state": "RUNNING"}, {"state": ""}, {"allocated_gpu_seconds": 1}, {"job_id": "42"}):
            self.write("allocations/41.json", {**allocation, **change})
            invalid, _ = export.Exporter(self.root).export()
            self.assertIsNone(self.cell(invalid, "metric-gpu-time", "AR · SGLang")["value"])

    def test_framework_memory_requires_each_request_reset_scope(self):
        self.metadata("ar-sglang")
        self.block("ar-sglang", 1)
        meta = self.metadata("ar-sglang", diagnostic=True)
        block = self.block("ar-sglang", 1, diagnostic=True)
        observations = []
        for index, request in enumerate(block["requests"]):
            observations.append({
                **{key: meta[key] for key in ("configuration", "slurm_job_id", "hostname", "gpu_uuid")},
                "event": "cuda_phase_observation", "request_ids": [request["request_id"]],
                "outer": {"cpu_start_ns": request["submitted_ns"] + 2},
                "memory": {"max_memory_allocated_bytes": (10 + index) * 1024 ** 3,
                           "max_memory_reserved_bytes": (20 + index) * 1024 ** 3,
                           "device_synchronized_before_read": True,
                           "scope": {"request_ids": [request["request_id"]], "request_identity_available": True,
                                     "reset_id": str(index), "reset_monotonic_ns": request["submitted_ns"] + 1}},
            })
        self.write("traces/ar-sglang/phase-1-test.jsonl", observations, lines=True)
        result, _ = export.Exporter(self.root).export()
        self.assertEqual(self.cell(result, "metric-memory", method="AR · SGLang", variant="Framework-allocated memory")["value"], 12)
        self.assertEqual(self.cell(result, "metric-memory", method="AR · SGLang", variant="Framework-reserved memory")["value"], 22)
        observations[1]["memory"]["scope"]["request_ids"] = ["unrelated-request"]
        self.write("traces/ar-sglang/phase-1-test.jsonl", observations, lines=True)
        result, _ = export.Exporter(self.root).export()
        self.assertIsNone(self.cell(result, "metric-memory", method="AR · SGLang", variant="Framework-allocated memory")["value"])

    def test_export_writes_review_files_without_overwriting_a_directory(self):
        destination = self.root / "export"
        result = export.write_export(export.Exporter(self.root), destination)
        self.assertEqual(result["status"], "prepared")
        self.assertEqual({path.name for path in destination.iterdir()}, {
            "trial-results.json", "measurement-evidence.json", "team-trial-results.json",
            "team-measurement-evidence.json", "tables.tex", "trial-tables.tex",
            "appendix-tables.tex", "metric-definitions.tex", "evidence-availability.tex", "results-summary.tex",
        })
        stored = json.loads((destination / "trial-results.json").read_text())
        self.assertTrue(all(cell["value"] is None and cell["sd"] is None and cell["n"] == 0 and cell["reason"] for cell in stored["cells"]))
        with self.assertRaises(FileExistsError):
            export.write_export(export.Exporter(self.root), destination)

    def test_team_counts_use_primary_records_and_preserve_position_eight(self):
        for configuration in ("team-ar", "team-ffn-100"):
            self.metadata(configuration)
            self.block(configuration, 1)
        path = "team-ffn-100/01-gsm8k.json"
        block = json.loads((self.root / path).read_text())
        for request in block["requests"]:
            request["server_metadata"] = {
                "team_rounds_schema_version": 1,
                "team_initial_target_token_id": request["output_token_ids"][0],
                "team_initial_delivered_tokens": 1,
                "team_delivered_output_tokens": 2,
                "team_rounds": [{"request_id": request["request_id"], "round_index": 0,
                    "proposed_draft_tokens": 8, "accepted_draft_tokens": 1,
                    "retained_target_tokens": 1, "committed_tokens": 2, "delivered_tokens": 1,
                    "proposed_depth": 8, "accepted_prefix_depth": 1}],
            }
        self.write(path, block)
        result, manifest = export.Exporter(self.root, team=True).export()
        method = "Pilot FFN 100%"
        self.assertEqual(self.cell(result, "metric-acceptance-length", method=method)["value"], 2)
        self.assertEqual(self.cell(result, "metric-returned-round", method=method)["value"], 1)
        self.assertEqual(self.cell(result, "metric-prefix-survival", method=method, position=8)["value"], 0)
        self.assertIsNone(self.cell(result, "metric-conditional-acceptance", method=method, position=8)["value"])
        self.assertEqual(self.cell(result, "metric-speedup", method=method)["value"], 1)
        self.assertEqual(manifest["primary_expected_blocks"], 120)
        block["requests"][0]["server_metadata"]["team_delivered_output_tokens"] = 3
        self.write(path, block)
        result, _ = export.Exporter(self.root, team=True).export()
        self.assertIsNone(self.cell(result, "metric-acceptance-length", method=method)["value"])

    def test_sglang_raw_counters_preserve_pretrim_tau_and_nine_positions(self):
        meta = self.metadata("dflash")
        meta["server_settings"]["command"] = ["--speculative-algorithm", "DFLASH", "--speculative-dflash-block-size", "10"]
        self.write("dflash/run.json", meta)
        block = self.block("dflash", 1)
        for request, accepted in zip(block["requests"], (1, 8, 1)):
            histogram = [0] * 10
            histogram[accepted] = 1
            request["server_metadata"] = {
                "spec_verify_ct": 1, "spec_num_correct_drafts": accepted, "spec_num_proposed_drafts": 9,
                "spec_accepted_drafts": accepted, "spec_proposed_drafts": 9,
                "spec_correct_drafts_histogram": histogram, "completion_tokens": 3,
                "num_retractions": 0, "spec_accept_length": 999,
            }
        self.write("dflash/01-gsm8k.json", block)
        result, _ = export.Exporter(self.root).export()
        self.assertEqual(self.cell(result, "metric-acceptance-length", method="DFlash")["value"], 13 / 3)
        self.assertEqual(self.cell(result, "metric-prefix-survival", method="DFlash", position=8)["value"], 1 / 3)
        self.assertEqual(self.cell(result, "metric-conditional-acceptance", method="DFlash", position=9)["value"], 0)
        block["requests"][0]["server_metadata"]["spec_correct_drafts_histogram"][0] = 1
        self.write("dflash/01-gsm8k.json", block)
        result, _ = export.Exporter(self.root).export()
        self.assertIsNone(self.cell(result, "metric-acceptance-length", method="DFlash")["value"])


if __name__ == "__main__":
    unittest.main()
