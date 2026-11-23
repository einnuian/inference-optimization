#!/usr/bin/env python3
"""Export saved trial evidence as Site cells and method-by-benchmark LaTeX tables."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import shutil
from urllib.parse import quote

from trial_recorder import CONFIGURATIONS, INPUT_SHA256, TARGET_REVISION, block_metrics, check_ids
from trial_native_stats import delta as native_delta


PROJECT = Path(__file__).resolve().parents[1]
CATALOGUE = PROJECT.parent / "site" / "lib" / "project-metric-catalogue.ts"
SGLANG_REVISION = "0bcd822377da7b5718e674eaf9c870d349424dd1"
SGLANG_COUNTER_SOURCE = "https://github.com/sgl-project/sglang/blob/" + SGLANG_REVISION + "/python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L700-L773"
EAGLE_TREE_SOURCE = "https://github.com/sgl-project/sglang/blob/" + SGLANG_REVISION + "/python/sglang/srt/speculative/eagle_worker_common.py#L408-L658"
TREE_VARIANT = "Tree-prefix diagnostic"
POSITION_METRICS = {"metric-prefix-survival", "metric-conditional-acceptance"}
TREE_DEFINITIONS = {
    "metric-prefix-survival": "For EAGLE-3, count verification rounds whose proposal tree contains a node at the stated depth. Among those rounds, count rounds whose accepted path reaches that depth. Divide accepted rounds by eligible rounds. Depth one contains the root's children. The root is a previously generated target token. Counts precede final output clipping.",
    "metric-conditional-acceptance": "For EAGLE-3, count verification rounds whose accepted path reaches the preceding depth and has at least one proposed child at the stated depth. Among those rounds, count rounds that accept a child. Divide accepted rounds by eligible rounds and multiply by 100. A leaf has no eligible next proposal. Counts precede final output clipping.",
}
TREE_CONVENTION = "These tree-prefix diagnostics use the recorded branching tree. The EAGLE-3 paper measures n-alpha with a chain and defines n by the draft feature history. The tree values preserve their separate interpretation."
COMPONENT_TIMING_NOTE = "Round latency uses host monotonic timestamps from drafting through the final state update. Draft and verification latency use CUDA events around their recorded operations. EAGLE-3's draft interval covers proposal generation. Draft extension remains inside the full round. P-EAGLE links each draft to its later verification, and its consecutive full-round intervals can overlap. Diagnostic measurements include synchronization and recording overhead. The evidence identifies each operation and records the overlap counts."
PROPOSAL_DENOMINATOR_NOTE = "The current EAGLE-3 setting proposes 15 candidate tree nodes per round. DFlash proposes nine linear draft tokens, and P-EAGLE proposes seven linear draft tokens. EAGLE-3's denominator includes candidates on alternative branches; its numerator counts the accepted path. Each linear method's denominator contains its proposed sequence. These different proposal structures limit comparisons of the acceptance percentages. The team drafter proposes eight linear tokens per round."
LOWER_IS_BETTER = {"metric-actual-charge", "metric-completion", "metric-ttft", "metric-pause", "metric-gpu-time", "metric-memory", "metric-tpot", "metric-pooled-gap", "metric-component-time"}
BOLD_NOTE = "The bold mean is the best observed value in each benchmark column. The comparison covers the recorded implementations under their stated configurations. At least two eligible method entries are required. Ties at the displayed precision share bold formatting. Bold formatting conveys an observed comparison and makes no statistical-significance claim. Brackets show the available repetition count. Response differences remain part of the interpretation."
MAIN_METRICS = ("metric-otps", "metric-speedup", "metric-gpu-time", "metric-acceptance-length")
TIMED_RESOURCE_NOTE = "Compute cost uses each timed benchmark interval and the GPU count from its final Slurm allocation. The interval includes client gaps between requests. Startup, warmups, and time outside the benchmark have separate whole-job accounting. Per-prompt GPU-seconds multiply the allocated GPU count by that prompt's completion time in seconds."
DESCRIPTIVE_METRICS = {"metric-component-time"}
COMPONENT_COMPARISON_NOTE = "Component times support an explanation of measured runtime. Each method performs different work per verification round. A shorter component can still accompany a longer whole response. The table therefore reports component durations with equal emphasis."
BENCHMARKS = {
    "gsm8k": "GSM8K", "math500": "MATH-500", "aime25": "AIME 2025",
    "humaneval": "HumanEval", "mbpp": "MBPP", "livecodebench": "LiveCodeBench",
    "mt-bench": "MT-Bench", "alpaca": "Alpaca", "arena-hard-v2": "Arena-Hard",
    "cnn-dm": "CNN/DailyMail",
}
METHODS = {
    "eagle3": "EAGLE-3", "peagle": "P-EAGLE", "dflash": "DFlash", "dspark": "DSpark",
    "ar-sglang": "AR · SGLang", "ar-vllm-v1": "AR · vLLM ModelRunnerV1",
    "ar-vllm-v2": "AR · vLLM ModelRunnerV2",
}
DEFERRED_CONFIGURATIONS = frozenset({"dspark", "ar-vllm-v2"})
TEAM_METHODS = {
    "team-ffn-100": "Pilot FFN 100%",
    "team-ffn-75": "Pilot FFN 75%",
    "team-ffn-50": "Pilot FFN 50%",
    "team-ar": "AR · Transformers (pilot)",
}
TEAM_CONFIGURATIONS = {name: ("transformers-team", None if name == "team-ar" else "team-ar") for name in TEAM_METHODS}
TEAM_ROWS = tuple(TEAM_METHODS.values())
BENCHMARK_PANELS = (
    ("Mathematics", tuple(BENCHMARKS.values())[:3]),
    ("Code", tuple(BENCHMARKS.values())[3:6]),
    ("Conversation, instruction following, and summarization", tuple(BENCHMARKS.values())[6:]),
)
DRAFT_METRICS = {
    "metric-acceptance-length", "metric-acceptance-length-request", "metric-draft-fraction",
    "metric-accepted-round", "metric-returned-round", "metric-prefix-survival",
    "metric-conditional-acceptance", "metric-component-time",
}
VARIANTS = {
    "metric-completion": ("Median", "95th percentile"),
    "metric-ttft": ("Median", "95th percentile"),
    "metric-memory": ("Sampled whole-device memory", "Framework-allocated memory", "Framework-reserved memory"),
    "metric-component-time": ("Round latency", "Draft latency", "Verification latency"),
    "metric-prefix-survival": ("", TREE_VARIANT),
    "metric-conditional-acceptance": ("", TREE_VARIANT),
}
TIMING = {
    ("metric-completion", "Median"): ("end_to_end_seconds", "median", 1000),
    ("metric-completion", "95th percentile"): ("end_to_end_seconds", "p95", 1000),
    ("metric-ttft", "Median"): ("ttft_seconds", "median", 1000),
    ("metric-ttft", "95th percentile"): ("ttft_seconds", "p95", 1000),
    ("metric-tpot", ""): ("tpot_seconds", "mean", 1000),
    ("metric-request-speed", ""): ("output_rate", "mean", 1),
    ("metric-pause", ""): ("max_inter_chunk_seconds", "p95", 1000),
    ("metric-pooled-gap", ""): ("inter_chunk_seconds", "p95", 1000),
}
PAIR_FIELDS = ("prepared_sha256", "target_revision", "input_sha256", "repetitions",
               "temperature", "max_new_tokens", "model", "gpu_uuid", "gpu_record")


class EvidenceError(ValueError):
    pass


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def integer(value):
    return type(value) is int and value >= 0


def same(actual, expected, description):
    if actual is None or expected is None:
        if actual != expected:
            raise EvidenceError(description + " differs from the saved observations.")
    elif not finite(actual) or not finite(expected) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise EvidenceError(description + " differs from the saved observations.")


def catalogue(path):
    source = path.read_text(encoding="utf-8")
    literal = r"('(?:\\.|[^'\\])*')"
    pattern = r"\{\s*id:\s*" + literal + r",\s*name:\s*" + literal + r",\s*unit:\s*" + literal + r",\s*definition:\s*" + literal
    rows = [tuple(ast.literal_eval(value) for value in match) for match in re.findall(pattern, source)]
    if not rows or len(rows) != len(re.findall(r"\bid:\s*'metric-", source)):
        raise EvidenceError("The metric catalogue format changed. Review the exporter before publishing its labels.")
    result = {key: {"id": key, "name": name, "unit": unit, "definition": definition} for key, name, unit, definition in rows}
    if len(result) != len(rows):
        raise EvidenceError("The metric catalogue contains duplicate metric IDs.")
    return result


def validate_request(row):
    for field in ("input_token_ids", "output_token_ids", "native_output_token_ids"):
        check_ids(row[field])
    if not row["input_token_ids"] or not integer(row["output_tokens"]) or row["output_tokens"] != len(row["output_token_ids"]):
        raise EvidenceError("The request token counts differ from its token IDs.")
    native, delivered = row["native_output_token_ids"], row["output_token_ids"]
    if native[:len(delivered)] != delivered or len(native) - len(delivered) not in (0, 1):
        raise EvidenceError("The delivered token IDs differ from the native response and terminal-stop removal.")
    if row.get("finish_reason") not in ("stop", "length") or not isinstance(row.get("output_text"), str):
        raise EvidenceError("The request requires a completed response and stopping reason.")
    start, end = row["submitted_ns"], row["completed_ns"]
    if not integer(start) or not integer(end) or end <= start:
        raise EvidenceError("The request requires increasing submission and completion timestamps.")
    content = []
    previous = start
    for event in row["events"]:
        received = event["received_ns"]
        if not integer(received) or not previous <= received <= end:
            raise EvidenceError("The streaming timestamps fall outside the request interval.")
        previous = received
        if event.get("text"):
            content.append(received)
    latency = (end - start) / 1e9
    ttft = (content[0] - start) / 1e9 if content else None
    gaps = [(b - a) / 1e9 for a, b in zip(content, content[1:])]
    tpot = (latency - ttft) / (row["output_tokens"] - 1) if row["output_tokens"] > 1 and ttft is not None else None
    for field, value in (("end_to_end_seconds", latency), ("ttft_seconds", ttft),
                         ("tpot_seconds", tpot), ("output_rate", row["output_tokens"] / latency),
                         ("max_inter_chunk_seconds", max(gaps) if gaps else None)):
        same(row[field], value, field)
    if row.get("first_content_ns") != (content[0] if content else None):
        raise EvidenceError("The first-content timestamp differs from the nonempty stream events.")
    if len(row["inter_chunk_seconds"]) != len(gaps):
        raise EvidenceError("The saved stream gap count differs from the nonempty stream events.")
    for actual, expected in zip(row["inter_chunk_seconds"], gaps):
        same(actual, expected, "Inter-chunk latency")


def validate_block(block, benchmark, repetition):
    if block.get("benchmark") != benchmark or block.get("repetition") != repetition:
        raise EvidenceError("The benchmark block identity differs from its filename.")
    rows = block["requests"]
    if len(rows) != 3 or len({row["prompt_id"] for row in rows}) != 3:
        raise EvidenceError("Each benchmark block requires three distinct prompts.")
    for row in rows:
        if row.get("benchmark") != benchmark or not isinstance(row.get("request_id"), str) or not row["request_id"]:
            raise EvidenceError("The request identity differs from its benchmark block.")
        validate_request(row)
    if any(b["submitted_ns"] < a["completed_ns"] for a, b in zip(rows, rows[1:])):
        raise EvidenceError("The saved requests overlap. The trial requires one active request at a time.")
    if block["started_ns"] != rows[0]["submitted_ns"] or block["ended_ns"] != rows[-1]["completed_ns"]:
        raise EvidenceError("The benchmark interval differs from its request timestamps.")
    measured = block_metrics(rows, (block["ended_ns"] - block["started_ns"]) / 1e9)
    for field in ("output_tokens", "elapsed_seconds", "otps"):
        same(block["metrics"][field], measured[field], field)
    # Recalculate every timing statistic from the original request observations.
    block["metrics"] = measured
    return block


def eagle_tree_depths(before, after, row):
    """Reconstruct the unique greedy path from the pinned, single-request tree."""
    def raw_value(fields, name):
        field = fields.get(name, {})
        if not isinstance(field, dict) or "value" not in field or "omitted" in field:
            raise EvidenceError("The EAGLE-3 tree capture lacks the complete " + name + " field.")
        return field["value"]

    def vector(fields, name, length):
        values = raw_value(fields, name)
        if isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
            values = values[0]
        if not isinstance(values, list) or len(values) != length or any(type(value) is not int for value in values):
            raise EvidenceError("The EAGLE-3 tree requires one complete integer vector for " + name + ".")
        return values

    tree = before.get("raw_tree", {})
    size = raw_value(tree, "draft_token_num")
    if not integer(size) or size < 1 or size > 256:
        raise EvidenceError("The EAGLE-3 capture requires a complete tree within its recorded capture limit.")
    tokens = vector(tree, "draft_token", size)
    check_ids(tokens)
    positions = vector(tree, "positions", size)
    retrieve = vector(tree, "retrieve_index", size)
    children = vector(tree, "retrieve_next_token", size)
    siblings = vector(tree, "retrieve_next_sibling", size)
    if retrieve != list(range(size)):
        raise EvidenceError("The tree adapter requires the pinned single-request retrieve-index convention.")
    if siblings[0] != -1 or any(value < -1 or value >= size for value in children + siblings):
        raise EvidenceError("The EAGLE-3 child and sibling indices fall outside the captured tree.")
    depths, child_lists = {0: 0}, {}
    pending = [0]
    while pending:
        parent = pending.pop()
        child, listed, token_values = children[parent], [], set()
        while child != -1:
            if child in depths or tokens[child] in token_values:
                raise EvidenceError("The EAGLE-3 tree contains a cycle, repeated parent, or ambiguous sibling token.")
            depths[child] = depths[parent] + 1
            token_values.add(tokens[child])
            listed.append(child)
            pending.append(child)
            child = siblings[child]
        child_lists[parent] = listed
    if len(depths) != size or any(positions[node] - positions[0] != depth for node, depth in depths.items()):
        raise EvidenceError("The EAGLE-3 positions and connected tree depths do not reconcile.")
    accepted = row.get("accepted_draft_tokens")
    committed = row.get("committed_token_ids")
    if not integer(accepted) or row.get("proposed_draft_tokens") != size - 1 or row.get("retained_target_tokens") != 1:
        raise EvidenceError("The EAGLE-3 scheduler counts differ from its captured tree.")
    check_ids(committed)
    if len(committed) != accepted + 1 or row.get("committed_tokens") != len(committed):
        raise EvidenceError("The EAGLE-3 committed path requires its accepted tokens and one retained target token.")
    tail = row.get("previous_output_tail")
    if not isinstance(tail, list) or not tail or tail[-1] != tokens[0]:
        raise EvidenceError("The EAGLE-3 tree root differs from the previous target token.")
    result = after.get("raw_result", {})
    if after.get("result_type") != "GenerationBatchResult":
        raise EvidenceError("The EAGLE-3 result requires the pinned GenerationBatchResult schema.")
    if vector(result, "accept_lens", 1) != [len(committed)] or vector(result, "next_token_ids", size)[:len(committed)] != committed:
        raise EvidenceError("The compacted EAGLE-3 verification result differs from the committed path.")
    path = [0]
    for token in committed[:accepted]:
        matches = [node for node in child_lists[path[-1]] if tokens[node] == token]
        if len(matches) != 1:
            raise EvidenceError("An accepted EAGLE-3 token has no unique child on the captured path.")
        path.append(matches[0])
    if any(tokens[node] == committed[-1] for node in child_lists[path[-1]]):
        raise EvidenceError("The retained target token matches an unaccepted child on the captured EAGLE-3 path.")
    return {"proposed_depth": max(depths.values()), "accepted_prefix_depth": accepted,
            "conditional_eligible_positions": [depth + 1 for depth, node in enumerate(path) if child_lists[node]],
            "accepted_node_indices": path[1:], "proposal_node_depths": [depths[node] for node in range(size)],
            "accepted_path_ends_at_leaf": not child_lists[path[-1]]}


class Exporter:
    def __init__(self, trial, metric_catalogue=CATALOGUE, evidence_prefix="", *, team=False, memory_samples="gpu-memory.jsonl", additional_memory_samples=(), include_deferred=False):
        self.root = Path(trial).resolve(strict=True)
        self.memory_sample_files = []
        for selected in (memory_samples, *additional_memory_samples):
            selected_memory = self.root / Path(selected)
            if selected_memory.is_symlink() or not selected_memory.resolve().is_relative_to(self.root):
                raise EvidenceError("The memory samples must be regular files inside the trial directory.")
            relative = selected_memory.resolve().relative_to(self.root).as_posix()
            if relative in self.memory_sample_files:
                raise EvidenceError("Select each memory-sample file once.")
            self.memory_sample_files.append(relative)
        self.memory_samples = self.memory_sample_files[0]
        self.metrics = catalogue(Path(metric_catalogue))
        self.catalogue_path = Path(metric_catalogue)
        self.prefix = evidence_prefix.rstrip("/")
        self.team = team
        self.include_deferred = include_deferred
        self.deferred = frozenset() if team or include_deferred else DEFERRED_CONFIGURATIONS
        self.methods = TEAM_METHODS if team else METHODS
        self.configurations = TEAM_CONFIGURATIONS if team else CONFIGURATIONS
        self.files = {}
        self.runs = {}
        self.diagnostics = {}
        self.cell_details = []
        self.memory = None
        self.round_cache = {}
        self.raw_round_cache = {}
        self.phase_cache = {}
        self.tree_cache = {}
        self.tree_depth_cache = {}
        self.component_cache = {}
        self.allocation_cache = {}

    def relative(self, path):
        return Path(path).relative_to(self.root).as_posix()

    def read(self, path, *, lines=False):
        path = Path(path)
        if path.is_symlink() or not path.resolve().is_relative_to(self.root) or not path.is_file():
            raise EvidenceError("The evidence file is unavailable: " + self.relative(path))
        raw = path.read_bytes()
        self.files[self.relative(path)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        def reject(value):
            raise EvidenceError("The evidence contains a nonfinite JSON number: " + value)
        if lines:
            return [json.loads(line, parse_constant=reject) for line in raw.splitlines() if line.strip()]
        return json.loads(raw, parse_constant=reject)

    def load_run(self, configuration, diagnostic=False):
        directory = self.root / "diagnostics" / configuration if diagnostic else self.root / configuration
        result = {"directory": directory, "blocks": {}, "errors": {}, "metadata": None}
        try:
            meta = self.read(directory / "run.json")
            backend, reference = self.configurations[configuration]
            expected = {"configuration": configuration, "backend": backend, "reference": reference,
                        "repetitions": 3, "temperature": 0, "max_new_tokens": 2048,
                        "target_revision": TARGET_REVISION, "input_sha256": INPUT_SHA256}
            if any(meta.get(key) != value for key, value in expected.items()):
                raise EvidenceError("The run settings differ from the fixed trial procedure.")
            if self.team and meta.get("phase") != ("diagnostic" if diagnostic else "primary"):
                raise EvidenceError("The team run identifies a different measurement phase.")
            for field in PAIR_FIELDS + ("slurm_job_id", "hostname"):
                if meta.get(field) is None:
                    raise EvidenceError("The run record is missing " + field + ".")
            if not isinstance(meta["prepared_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", meta["prepared_sha256"]):
                raise EvidenceError("The run requires the prepared input bundle's SHA-256 checksum.")
            gpu_record = meta["gpu_record"]
            if not isinstance(gpu_record, str) or len(gpu_record.splitlines()) != 1 or "L40S" not in gpu_record or gpu_record.partition(",")[0].strip() != meta["gpu_uuid"]:
                raise EvidenceError("The run must identify its selected L40S GPU and matching UUID.")
            result["metadata"] = meta
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            result["error"] = str(error)
            return result
        seen = set()
        signatures = {}
        for benchmark in BENCHMARKS:
            for repetition in range(1, 4):
                key = (benchmark, repetition)
                path = directory / f"{repetition:02d}-{benchmark}.json"
                try:
                    block = validate_block(self.read(path), benchmark, repetition)
                    ids = [row["request_id"] for row in block["requests"]]
                    if len(set(ids)) != 3 or any(request_id in seen for request_id in ids):
                        raise EvidenceError("A request ID appears in more than one measurement record.")
                    signature = [(row["prompt_id"], row["input_token_ids"]) for row in block["requests"]]
                    if benchmark in signatures and signatures[benchmark] != signature:
                        raise EvidenceError("The selected prompts or their order change between repetitions.")
                    signatures[benchmark] = signature
                    seen.update(ids)
                    block["source"] = self.relative(path)
                    result["blocks"][key] = block
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
                    result["errors"][key] = str(error)
        return result

    def prepare(self):
        for configuration in self.methods:
            self.runs[configuration] = self.load_run(configuration)
            self.diagnostics[configuration] = self.load_run(configuration, diagnostic=True)

    @staticmethod
    def pair(method, reference, benchmark, repetition):
        if method["metadata"] is None or reference["metadata"] is None:
            reasons = [run.get("error", "The run metadata is unavailable.") for run in (method, reference) if run["metadata"] is None]
            raise EvidenceError("The comparison requires both recorded model configurations. " + " ".join(reasons))
        for field in PAIR_FIELDS:
            if method["metadata"][field] != reference["metadata"][field]:
                raise EvidenceError("The paired configurations differ in " + field + ".")
        key = (benchmark, repetition)
        if key not in method["blocks"] or key not in reference["blocks"]:
            raise EvidenceError("The comparison requires both completed benchmark blocks.")
        left, right = method["blocks"][key], reference["blocks"][key]
        if [(row["prompt_id"], row["input_token_ids"]) for row in left["requests"]] != [
                (row["prompt_id"], row["input_token_ids"]) for row in right["requests"]]:
            raise EvidenceError("The paired benchmark blocks use different prompts or ordering.")
        return left, right

    def diagnostic_directory(self, configuration):
        run = self.diagnostics[configuration]
        if run["metadata"] is None:
            raise EvidenceError("The separate diagnostic run has no valid metadata.")
        record = run["metadata"].get("server_settings", {}).get("diagnostic", {})
        relative = record.get("output_results_relative")
        if record.get("scope") != "separate_diagnostic" or not isinstance(relative, str):
            raise EvidenceError("The diagnostic launch record does not identify its trace directory.")
        path = (self.root.parent / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_dir():
            raise EvidenceError("The diagnostic trace directory is unavailable in this trial.")
        return path

    def rounds(self, configuration, benchmark, repetition):
        cache_key = (configuration, benchmark, repetition)
        if cache_key in self.round_cache:
            return self.round_cache[cache_key]
        if self.team:
            return self.team_rounds(configuration, benchmark, repetition)
        run = self.diagnostics[configuration]
        diagnostic, primary = self.pair(run, self.runs[configuration], benchmark, repetition)
        if configuration not in self.raw_round_cache:
            path = self.diagnostic_directory(configuration)
            files = sorted(path.glob("*-rounds-*.jsonl"))
            if not files:
                raise EvidenceError("The separate diagnostic run has no saved verification-round trace.")
            self.raw_round_cache[configuration] = [
                (row, self.relative(file)) for file in files for row in self.read(file, lines=True)
            ]
        request_map = {row["request_id"]: row for row in diagnostic["requests"]}
        grouped = defaultdict(dict)
        sources = {diagnostic["source"], primary["source"]}
        for row, source in self.raw_round_cache[configuration]:
            if row.get("request_id") not in request_map:
                continue
            if row.get("configuration") != configuration or row.get("scope") != "separate_diagnostic":
                raise EvidenceError("A round record identifies a different diagnostic configuration.")
            number = row.get("round")
            if not integer(number) or number == 0:
                raise EvidenceError("The round trace requires positive, consecutive round numbers.")
            prior = grouped[row["request_id"]].get(number)
            if prior is not None and prior != row:
                raise EvidenceError("Two round records assign different observations to the same request and round.")
            grouped[row["request_id"]][number] = row
            sources.add(source)
        all_rows = []
        per_request = []
        initial_tokens = 0
        for request_id, request in request_map.items():
            mapping = grouped[request_id]
            if not mapping or sorted(mapping) != list(range(1, len(mapping) + 1)):
                raise EvidenceError("A completed diagnostic request lacks a consecutive verification-round trace.")
            rows = [dict(mapping[number]) for number in sorted(mapping)]
            offset = rows[0]["previous_output_count"]
            if not integer(offset) or offset > len(request["native_output_token_ids"]):
                raise EvidenceError("The initial target-token count falls outside the saved response.")
            initial_tokens += min(offset, request["output_tokens"])
            delivered = 0
            for row in rows:
                for field in ("proposed_draft_tokens", "accepted_draft_tokens", "retained_target_tokens", "committed_tokens"):
                    if not integer(row[field]):
                        raise EvidenceError("Verification-round counts must be nonnegative integers.")
                check_ids(row["committed_token_ids"])
                if row["committed_tokens"] != len(row["committed_token_ids"]) or row["committed_tokens"] != row["accepted_draft_tokens"] + row["retained_target_tokens"]:
                    raise EvidenceError("Committed tokens differ from the accepted-draft and retained-target counts.")
                if row["accepted_draft_tokens"] > row["proposed_draft_tokens"] or row["previous_output_count"] != offset:
                    raise EvidenceError("The round counts or output offsets do not reconcile.")
                visible_native = request["native_output_token_ids"][offset:offset + row["committed_tokens"]]
                if row["committed_token_ids"][:len(visible_native)] != visible_native:
                    raise EvidenceError("The round tokens differ from the saved response token IDs.")
                row["delivered_tokens"] = max(0, min(request["output_tokens"] - offset, row["committed_tokens"]))
                delivered += row["delivered_tokens"]
                offset += row["committed_tokens"]
            initial = min(rows[0]["previous_output_count"], request["output_tokens"])
            if delivered + initial != request["output_tokens"]:
                raise EvidenceError("The round trace does not account for the complete delivered response.")
            native_count = (request.get("server_metadata") or {}).get("spec_verify_ct")
            if native_count is not None and native_count != len(rows):
                raise EvidenceError("The captured round count differs from the server's verification count.")
            per_request.append(sum(row["committed_tokens"] for row in rows) / len(rows))
            all_rows.extend(rows)
        if configuration in {"peagle", "dspark"}:
            native, native_source = self.native(configuration, benchmark, repetition, diagnostic=True)
            if native["totals"]["num_drafts"] != len(all_rows):
                raise EvidenceError("The diagnostic native counters and round trace count different numbers of rounds.")
            sources.add(native_source)
        result = {"rows": all_rows, "request_means": per_request, "initial_tokens": initial_tokens, "sources": sorted(sources)}
        self.round_cache[cache_key] = result
        return result

    def native(self, configuration, benchmark, repetition, diagnostic=False):
        run = self.diagnostics[configuration] if diagnostic else self.runs[configuration]
        path = run["directory"] / f"native-{repetition:02d}-{benchmark}.json"
        saved = self.read(path)
        if saved.get("benchmark") != benchmark or saved.get("repetition") != repetition:
            raise EvidenceError("The native counter record identifies a different benchmark block.")
        before, after = saved["before"], saved["after"]
        if before.get("hostname") != run["metadata"]["hostname"] or before.get("gpu_uuid") != run["metadata"]["gpu_uuid"]:
            raise EvidenceError("The native counters identify different hardware from the run.")
        if after["observed_completions"] - before["observed_completions"] != 3:
            raise EvidenceError("The native counter interval must contain exactly three completed requests.")
        return native_delta(before, after), self.relative(path)

    def eagle_tree_rounds(self, benchmark, repetition):
        key = (benchmark, repetition)
        if key in self.tree_depth_cache:
            return self.tree_depth_cache[key]
        trace = self.rounds("eagle3", benchmark, repetition)
        run = self.diagnostics["eagle3"]
        block = run["blocks"][(benchmark, repetition)]
        metadata = run["metadata"]
        requests = {row["request_id"]: row for row in block["requests"]}
        if "eagle3" not in self.tree_cache:
            files = sorted(self.diagnostic_directory("eagle3").glob("phase-*.jsonl"))
            records = defaultdict(list)
            for file in files:
                for record in self.read(file, lines=True):
                    if record.get("event") not in {"eagle_tree_before_verify", "eagle_verification_result"}:
                        continue
                    for request_id, number in (record.get("verification_round_indices") or {}).items():
                        if isinstance(request_id, str) and integer(number) and number > 0:
                            records[(request_id, number)].append((record, self.relative(file)))
            self.tree_cache["eagle3"] = records
        sources = set(trace["sources"])
        observed = []
        for row in trace["rows"]:
            request_id = row["request_id"]
            number = row.get("native_verification_count")
            if not integer(number) or number != row["round"]:
                raise EvidenceError("Each EAGLE-3 tree requires a matching native verification-round number.")
            pair = {}
            for record, source in self.tree_cache["eagle3"].get((request_id, number), []):
                if any(record.get(field) != metadata[field] for field in ("configuration", "hostname", "gpu_uuid", "slurm_job_id")):
                    raise EvidenceError("The EAGLE-3 tree capture identifies different hardware or a different configuration.")
                if record.get("request_ids") != [request_id] or record.get("verification_round_indices") != {request_id: number}:
                    raise EvidenceError("The EAGLE-3 tree adapter requires one identified request per verification round.")
                if record.get("diagnostic_only") is not True or record.get("clock") != "time.perf_counter_ns" or not record.get("clock_implementation"):
                    raise EvidenceError("The EAGLE-3 tree capture requires an identified diagnostic clock.")
                start, end = record.get("cpu_capture_start_ns"), record.get("cpu_capture_end_ns")
                request = requests[request_id]
                if not integer(start) or not integer(end) or not request["submitted_ns"] <= start <= end <= request["completed_ns"]:
                    raise EvidenceError("The EAGLE-3 tree capture falls outside its diagnostic request interval.")
                event = record["event"]
                if event in pair and pair[event] != record:
                    raise EvidenceError("Multiple EAGLE-3 tree captures identify the same native verification round.")
                pair[event] = record
                sources.add(source)
            if set(pair) != {"eagle_tree_before_verify", "eagle_verification_result"}:
                raise EvidenceError("A diagnostic EAGLE-3 round lacks its complete before-and-after tree capture.")
            before, after = pair["eagle_tree_before_verify"], pair["eagle_verification_result"]
            if not before.get("capture_id") or not before.get("observation_id") or any(before.get(field) != after.get(field) for field in ("capture_id", "observation_id", "clock_implementation")):
                raise EvidenceError("The EAGLE-3 tree and result identify different verification observations.")
            if before["cpu_capture_end_ns"] > after["cpu_capture_start_ns"]:
                raise EvidenceError("The EAGLE-3 result capture precedes its proposal-tree capture.")
            observed.append({"request_id": request_id, "native_verification_count": number,
                             "capture_id": before["capture_id"], **eagle_tree_depths(before, after, row)})
        self.tree_depth_cache[key] = (observed, sorted(sources))
        return self.tree_depth_cache[key]

    def eagle_tree_observation(self, metric, benchmark, repetition, position):
        observed, sources = self.eagle_tree_rounds(benchmark, repetition)
        eligible = [row for row in observed if row["proposed_depth"] >= position] if metric == "metric-prefix-survival" else [
            row for row in observed if position in row["conditional_eligible_positions"]]
        if not eligible:
            raise EvidenceError("No captured EAGLE-3 round has an eligible tree proposal at this depth.")
        passed = sum(row["accepted_prefix_depth"] >= position for row in eligible)
        return passed / len(eligible) * (100 if metric == "metric-conditional-acceptance" else 1), sources, {
            "scope": "Separate diagnostic EAGLE-3 tree-prefix observations", "position": position,
            "verification_rounds": len(observed), "eligible_rounds": len(eligible), "accepted_rounds": passed,
            "excluded_leaf_rounds": sum(row["accepted_prefix_depth"] == position - 1 and row["accepted_path_ends_at_leaf"] for row in observed),
            "definition": TREE_DEFINITIONS[metric], "paper_convention": TREE_CONVENTION,
            "source_revision": SGLANG_REVISION, "source_url": EAGLE_TREE_SOURCE,
            "capture_count": len(observed),
            "validation": "Every round has a complete connected tree, unique accepted path, matching compacted result, native round identity, and recorded request interval. The source files preserve all captured trees and paths.",
        }

    def team_rounds(self, configuration, benchmark, repetition):
        block = self.runs[configuration]["blocks"][(benchmark, repetition)]
        rows, request_means = [], []
        initial_tokens = initial_delivered = 0
        for request in block["requests"]:
            metadata = request.get("server_metadata") or {}
            if type(metadata.get("team_rounds_schema_version")) is not int or metadata["team_rounds_schema_version"] != 1:
                raise EvidenceError("The team request requires version 1 of the explicit round-count schema.")
            initial_id = metadata.get("team_initial_target_token_id")
            initial_count = metadata.get("team_initial_delivered_tokens")
            delivered_count = metadata.get("team_delivered_output_tokens")
            if not integer(initial_id) or type(initial_count) is not int or initial_count not in (0, 1) or not integer(delivered_count) or delivered_count != request["output_tokens"]:
                raise EvidenceError("The team request's initial token and delivered-token counts do not reconcile.")
            if initial_count == 1 and request["output_token_ids"][:1] != [initial_id]:
                raise EvidenceError("The team request's initial target token differs from the delivered response.")
            observed = metadata.get("team_rounds")
            if not isinstance(observed, list) or (initial_count == 0 and (observed or delivered_count)):
                raise EvidenceError("The team request requires a complete list of verification-round counts.")
            normalized = []
            for index, source in enumerate(observed):
                row = dict(source)
                if row.get("request_id") != request["request_id"] or type(row.get("round_index")) is not int or row["round_index"] != index:
                    raise EvidenceError("The team verification rounds require matching request IDs and consecutive zero-based indices.")
                for field in ("proposed_draft_tokens", "accepted_draft_tokens", "retained_target_tokens", "committed_tokens", "delivered_tokens", "proposed_depth", "accepted_prefix_depth"):
                    if not integer(row.get(field)):
                        raise EvidenceError("The team verification counts must be nonnegative integers.")
                if row["proposed_draft_tokens"] != 8 or row["proposed_depth"] != 8 or row["retained_target_tokens"] != 1:
                    raise EvidenceError("The team round must propose eight tokens and retain one target token.")
                if row["accepted_draft_tokens"] > 8 or row["accepted_prefix_depth"] != row["accepted_draft_tokens"] or row["committed_tokens"] != row["accepted_draft_tokens"] + 1:
                    raise EvidenceError("The team's accepted prefix and committed-token counts do not reconcile.")
                if row["delivered_tokens"] > row["committed_tokens"] or (index < len(observed) - 1 and row["delivered_tokens"] != row["committed_tokens"]):
                    raise EvidenceError("Output trimming must preserve the recorded team-round counts and occur in the final round.")
                row["depth_metrics_available"] = True
                normalized.append(row)
            if initial_count + sum(row["delivered_tokens"] for row in normalized) != delivered_count:
                raise EvidenceError("The team rounds and initial token do not account for every delivered output token.")
            initial_tokens += 1
            initial_delivered += initial_count
            if normalized:
                request_means.append(sum(row["committed_tokens"] for row in normalized) / len(normalized))
                rows.extend(normalized)
        if not rows:
            raise EvidenceError("The team requests contain no verification rounds eligible for this metric.")
        result = {"rows": rows, "request_means": request_means, "initial_tokens": initial_tokens,
                  "initial_delivered_tokens": initial_delivered, "sources": [block["source"]], "scope": "Primary team count records"}
        self.round_cache[(configuration, benchmark, repetition)] = result
        return result

    def sglang_counts(self, configuration, block, metric, position):
        command = self.runs[configuration]["metadata"].get("server_settings", {}).get("command", [])
        def option(name):
            if command.count(name) != 1 or command.index(name) + 1 >= len(command):
                raise EvidenceError("The SGLang counter calculation requires the recorded fixed proposal settings.")
            return command[command.index(name) + 1]
        algorithm = option("--speculative-algorithm")
        if algorithm != ("DFLASH" if configuration == "dflash" else "EAGLE3"):
            raise EvidenceError("The SGLang launch algorithm differs from the measured method.")
        width = int(option("--speculative-dflash-block-size" if configuration == "dflash" else "--speculative-num-draft-tokens")) - 1
        if width <= 0:
            raise EvidenceError("The speculative proposal width must be positive.")
        rounds = accepted = proposed = delivered = initial = trimmed = 0
        histogram = [0] * (width + 1)
        request_means = []
        for request in block["requests"]:
            meta = request.get("server_metadata") or {}
            native_count = len(request["native_output_token_ids"])
            if "spec_verify_ct" not in meta and native_count <= 1:
                initial += request["output_tokens"]
                continue
            count = meta.get("spec_verify_ct")
            correct = meta.get("spec_num_correct_drafts")
            candidates = meta.get("spec_num_proposed_drafts")
            bins = meta.get("spec_correct_drafts_histogram")
            if any(not integer(value) for value in (count, correct, candidates)) or count == 0:
                raise EvidenceError("The SGLang request requires explicit verification and draft-token counts.")
            if meta.get("num_retractions") != 0 or candidates != count * width or correct > candidates:
                raise EvidenceError("The SGLang counters require an unretracted request and the recorded fixed proposal width.")
            if not isinstance(bins, list) or len(bins) > width + 1 or any(not integer(value) for value in bins) or sum(bins) != count or sum(index * value for index, value in enumerate(bins)) != correct:
                raise EvidenceError("The SGLang acceptance histogram does not reconcile with its raw counters.")
            for name, value in (("spec_accepted_drafts", correct), ("spec_proposed_drafts", candidates)):
                if name in meta and meta[name] != value:
                    raise EvidenceError("The SGLang counter aliases disagree.")
            if meta.get("completion_tokens") != native_count or request["output_tokens"] < 1:
                raise EvidenceError("The SGLang completion count differs from the saved native response.")
            excess = 1 + correct + count - native_count
            if not 0 <= excess <= width:
                raise EvidenceError("The SGLang initial token, committed tokens, and final output trimming do not reconcile.")
            initial += 1
            rounds += count
            accepted += correct
            proposed += candidates
            delivered += request["output_tokens"] - 1
            trimmed += excess
            request_means.append((correct + count) / count)
            for index, value in enumerate(bins):
                histogram[index] += value
        if rounds == 0:
            raise EvidenceError("The SGLang requests contain no verification rounds eligible for this metric.")
        details = {"scope": "Primary SGLang request counters", "counter_source_revision": SGLANG_REVISION,
                   "counter_source_url": SGLANG_COUNTER_SOURCE, "verification_rounds": rounds,
                   "accepted_draft_tokens": accepted, "proposed_draft_tokens": proposed,
                   "retained_target_tokens": rounds, "committed_tokens": accepted + rounds,
                   "delivered_tokens": delivered, "initial_delivered_tokens": initial,
                   "native_tokens_removed_by_final_trimming": trimmed,
                   "acceptance_histogram": histogram, "eligible_requests_with_rounds": len(request_means),
                   "proposed_count_basis": "linear positions" if configuration == "dflash" else "tree candidate nodes"}
        values = {"metric-acceptance-length": (accepted + rounds) / rounds,
                  "metric-acceptance-length-request": statistics.mean(request_means),
                  "metric-accepted-round": accepted / rounds, "metric-draft-fraction": accepted / proposed * 100,
                  "metric-returned-round": delivered / rounds}
        if metric in values:
            return values[metric], details
        if configuration != "dflash":
            raise EvidenceError("The tree acceptance histogram does not identify comparable proposed prefix positions.")
        if position > width:
            raise EvidenceError("The configuration proposes no token at this draft position.")
        passed = sum(histogram[position:])
        eligible = rounds if metric == "metric-prefix-survival" else sum(histogram[position - 1:])
        if eligible == 0:
            raise EvidenceError("No verification round is eligible for conditional acceptance at this position.")
        details.update(position=position, accepted_rounds=passed, eligible_rounds=eligible)
        return passed / eligible * (100 if metric == "metric-conditional-acceptance" else 1), details

    def memory_peak(self, configuration, block):
        if self.memory is None:
            self.memory = [row for source in self.memory_sample_files for row in self.read(self.root / source, lines=True)]
        meta = self.runs[configuration]["metadata"]
        matching = sorted((row for row in self.memory if row.get("slurm_job_id") == meta["slurm_job_id"]
                           and row.get("hostname") == meta["hostname"] and row.get("gpu_uuid") == meta["gpu_uuid"]
                           and integer(row.get("monotonic_ns"))), key=lambda row: row["monotonic_ns"])
        if not matching or matching[0]["monotonic_ns"] > block["started_ns"] or matching[-1]["monotonic_ns"] < block["ended_ns"]:
            raise EvidenceError("The matching memory log does not span this complete benchmark interval. Sampling began after the block started or ended before it finished.")
        samples = [row for row in matching if block["started_ns"] <= row["monotonic_ns"] <= block["ended_ns"]]
        if not samples or any(not finite(row.get("memory_used_mib")) for row in samples):
            raise EvidenceError("The timed benchmark interval has no valid matching GPU memory samples.")
        return max(row["memory_used_mib"] for row in samples) / 1024, len(samples), {
            "sampler_first_monotonic_ns": matching[0]["monotonic_ns"], "sampler_last_monotonic_ns": matching[-1]["monotonic_ns"],
            "sampler_first_unix_ns": matching[0].get("unix_ns"), "sampler_last_unix_ns": matching[-1].get("unix_ns"),
            "first_interval_sample_ns": samples[0]["monotonic_ns"], "last_interval_sample_ns": samples[-1]["monotonic_ns"],
            "interval_maximum_sample_gap_seconds": max((b["monotonic_ns"] - a["monotonic_ns"] for a, b in zip(samples, samples[1:])), default=0) / 1e9,
            "coverage_note": "The matching job and GPU log brackets the entire timed benchmark. The value is the largest recorded sample within that interval. Peaks between samples remain unobserved.",
        }

    def phase_observations(self, configuration):
        if configuration not in self.phase_cache:
            path = self.diagnostic_directory(configuration)
            files = sorted(path.glob("phase-*.jsonl"))
            if not files:
                raise EvidenceError("The diagnostic run has no saved phase observations.")
            self.phase_cache[configuration] = [
                (row, self.relative(file)) for file in files for row in self.read(file, lines=True)
                if row.get("event") == "cuda_phase_observation"
            ]
        return self.phase_cache[configuration]

    def timed_allocation(self, configuration, block):
        job = str(self.runs[configuration]["metadata"]["slurm_job_id"])
        if not re.fullmatch(r"[0-9]+", job):
            raise EvidenceError("The timed allocation requires a numeric parent Slurm job ID.")
        if job not in self.allocation_cache:
            paths = (self.root / "allocations" / (job + ".json"), self.root / "allocation.json")
            path = next((candidate for candidate in paths if candidate.is_file()), None)
            if path is None:
                raise EvidenceError("The timed benchmark requires a final allocation record with its matching job ID and GPU count.")
            record = self.read(path)
            count, elapsed, total = (record.get(field) for field in ("gpu_count", "elapsed_seconds", "allocated_gpu_seconds"))
            state = str(record.get("state", "")).split()
            final_states = {"COMPLETED", "TIMEOUT", "CANCELLED", "FAILED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
            if (str(record.get("job_id")) != job or not integer(count) or count < 1 or not integer(elapsed) or elapsed < 1
                    or not integer(total) or total != count * elapsed or not record.get("source_command") or not record.get("raw_sacct")
                    or not state or state[0].rstrip("+") not in final_states
                    or record.get("end") in {None, "", "Unknown", "None"}):
                raise EvidenceError("The allocation record requires a final matching parent job and consistent allocated GPU time.")
            self.allocation_cache[job] = (record, self.relative(path))
        record, source = self.allocation_cache[job]
        duration = block["metrics"]["elapsed_seconds"]
        tokens = block["metrics"]["output_tokens"]
        if duration <= 0 or duration > record["elapsed_seconds"] or tokens <= 0:
            raise EvidenceError("The timed benchmark requires a positive interval and output count within its allocation duration.")
        return duration * record["gpu_count"], source, {
            "scope": "Timed benchmark allocation interval",
            "slurm_job_id": job, "allocated_gpu_count": record["gpu_count"],
            "benchmark_started_ns": block["started_ns"], "benchmark_ended_ns": block["ended_ns"],
            "timed_elapsed_seconds": duration, "timed_gpu_seconds": duration * record["gpu_count"],
            "output_tokens": tokens,
            "scope_note": "The interval includes sequential requests and client gaps within the benchmark block. Model startup, warmups, and time outside that block are excluded. The final Slurm record establishes the allocated GPU count and job identity.",
        }

    def framework_memory(self, configuration, benchmark, repetition, variant):
        run = self.diagnostics[configuration]
        block, primary = self.pair(run, self.runs[configuration], benchmark, repetition)
        meta = run["metadata"]
        requests = {row["request_id"] for row in block["requests"]}
        covered, peaks, sources, resets = set(), [], {block["source"], primary["source"]}, set()
        counter = "max_memory_allocated_bytes" if variant == "Framework-allocated memory" else "max_memory_reserved_bytes"
        for row, source in self.phase_observations(configuration):
            identities = row.get("request_ids", [])
            if not isinstance(identities, list) or not requests.intersection(identities):
                continue
            if len(identities) != 1 or any(row.get(field) != meta[field] for field in ("configuration", "slurm_job_id", "hostname", "gpu_uuid")):
                raise EvidenceError("The framework memory observation has a different request or hardware scope.")
            memory = row["memory"]
            scope = memory["scope"]
            if scope.get("request_ids") != identities or scope.get("request_identity_available") is not True or memory.get("device_synchronized_before_read") is not True:
                raise EvidenceError("The framework peak lacks a synchronized reset scope for this request.")
            reset = scope.get("reset_monotonic_ns")
            start = row.get("outer", {}).get("cpu_start_ns")
            if not integer(reset) or not integer(start) or reset > start or not integer(memory.get(counter)):
                raise EvidenceError("The framework peak has an invalid reset timestamp or byte count.")
            covered.update(identities)
            resets.add(scope.get("reset_id"))
            peaks.append(memory[counter] / 1024 ** 3)
            sources.add(source)
        if covered != requests:
            raise EvidenceError("The framework memory trace does not cover all three diagnostic requests.")
        return max(peaks), sorted(sources), {"memory_observations": len(peaks), "memory_reset_count": len(resets), "scope": "Separate diagnostic request intervals"}

    def component_time(self, configuration, benchmark, repetition, variant):
        cache_key = (configuration, benchmark, repetition)
        if cache_key in self.component_cache:
            values, sources, details = self.component_cache[cache_key]
            return values[variant], sources, details
        if configuration not in {"eagle3", "dflash", "peagle"}:
            raise EvidenceError("The configuration requires validated proposal, verification, and final state-update boundaries for its diagnostic round times.")
        trace = self.rounds(configuration, benchmark, repetition)
        run = self.diagnostics[configuration]
        block = run["blocks"][(benchmark, repetition)]
        requests = {row["request_id"]: row for row in block["requests"]}
        meta = run["metadata"]
        observations, sources = {}, set(trace["sources"])
        for observed, source in self.phase_observations(configuration):
            if not requests.keys() & set(observed.get("request_ids", [])):
                continue
            identity = observed.get("observation_id")
            if not isinstance(identity, str) or not identity:
                raise EvidenceError("A diagnostic phase observation has no unique identity.")
            if identity in observations and observations[identity] != observed:
                raise EvidenceError("Two diagnostic phase records assign different values to the same observation.")
            observations[identity] = observed
            sources.add(source)
        by_step, by_count = defaultdict(list), defaultdict(list)
        for observed in observations.values():
            for request_id in observed.get("request_ids", []):
                if observed.get("scheduler_step_id"):
                    by_step[(request_id, observed["scheduler_step_id"])].append(observed)
                count = (observed.get("verification_round_indices") or {}).get(request_id)
                if integer(count):
                    by_count[(request_id, count)].append(observed)

        def validate_observation(observed, request_id):
            if observed.get("request_ids") != [request_id] or observed.get("diagnostic_only") is not True:
                raise EvidenceError("Each diagnostic round time requires one identified request.")
            for field in ("configuration", "hostname", "gpu_uuid", "slurm_job_id"):
                if observed.get(field) != meta[field]:
                    raise EvidenceError("The phase observation identifies different hardware or a different configuration.")
            if observed.get("clock") != "time.perf_counter_ns" or not observed.get("clock_implementation"):
                raise EvidenceError("The phase observation requires an identified monotonic clock.")
            outer = observed.get("outer", {})
            start, end, sync = outer.get("cpu_start_ns"), outer.get("cpu_end_ns"), observed.get("synchronized_cpu_end_ns")
            request = requests[request_id]
            if any(not integer(value) for value in (start, end, sync)) or not request["submitted_ns"] <= start <= end <= sync <= request["completed_ns"]:
                raise EvidenceError("The synchronized phase interval falls outside its diagnostic request.")
            if observed.get("memory", {}).get("device_synchronized_before_read") is not True:
                raise EvidenceError("The worker observation lacks evidence of device synchronization before its final timestamp.")

        def phase(observed, name):
            matches = [item for item in observed.get("phases", []) if item.get("name") == name]
            if len(matches) != 1:
                raise EvidenceError("Each verification round requires exactly one recorded " + name + " interval.")
            item = matches[0]
            start, end = item.get("cpu_start_ns"), item.get("cpu_end_ns")
            outer = observed["outer"]
            parent = item.get("parent_sequence")
            parent_matches = [candidate for candidate in observed.get("phases", []) if candidate.get("sequence") == parent] if parent else [outer]
            if (any(not integer(value) for value in (start, end)) or not outer["cpu_start_ns"] <= start <= end <= outer["cpu_end_ns"]
                    or not finite(item.get("gpu_elapsed_seconds")) or item.get("device_index") != outer.get("device_index")
                    or not integer(item.get("device_index")) or not integer(parent) or not integer(item.get("stream_id"))
                    or len(parent_matches) != 1 or not parent_matches[0]["cpu_start_ns"] <= start <= end <= parent_matches[0]["cpu_end_ns"]):
                raise EvidenceError("A component interval lacks valid timing, device, stream, or outer-boundary evidence.")
            return item

        intervals = []
        for row in trace["rows"]:
            request_id = row["request_id"]
            if row.get("state_update_completed") is not True or not row.get("state_update_boundary"):
                raise EvidenceError("The diagnostic round lacks its completed final state-update boundary.")
            if any(row.get(field) != meta[field] for field in ("configuration", "hostname", "gpu_uuid", "slurm_job_id")):
                raise EvidenceError("The state-update record identifies different hardware or a different configuration.")
            end = row.get("state_update_completed_ns")
            if not integer(end) or end > requests[request_id]["completed_ns"]:
                raise EvidenceError("The final state-update timestamp falls outside the diagnostic request.")
            if configuration == "peagle":
                step = row.get("scheduler_step_id")
                candidates = by_step[(request_id, step)] if step else []
            else:
                count = row.get("native_verification_count")
                candidates = by_count[(request_id, count)] if integer(count) and count == row["round"] else []
            if len(candidates) != 1:
                raise EvidenceError("The diagnostic round requires exactly one matching worker observation.")
            consumer = candidates[0]
            validate_observation(consumer, request_id)
            if row.get("clock") != consumer["clock"] or row.get("clock_implementation") != consumer["clock_implementation"] or consumer["synchronized_cpu_end_ns"] > end:
                raise EvidenceError("The final state update requires the worker's clock and completed GPU synchronization.")
            if configuration == "peagle":
                link = (consumer.get("verification_proposals") or {}).get(request_id, {})
                producer = observations.get(link.get("producer_observation_id"))
                if producer is None or link.get("tensor_identity_verified") is not True or link.get("input_source") not in {"previous_draft_tensor", "scheduled_host_tokens"}:
                    raise EvidenceError("The P-EAGLE verification lacks its exact draft-producer linkage.")
                validate_observation(producer, request_id)
                produced = (producer.get("produced_drafts") or {}).get(request_id, {})
                tokens = link.get("token_ids")
                check_ids(tokens)
                if (producer.get("scheduler_step_id") != link.get("producer_scheduler_step_id")
                        or produced.get("row_index") != link.get("producer_row_index") or produced.get("token_ids") != tokens
                        or len(tokens) != row["proposed_draft_tokens"] or row.get("invalid_proposal_count") != 0
                        or tokens[:row["accepted_draft_tokens"]] != row["committed_token_ids"][:row["accepted_draft_tokens"]]):
                    raise EvidenceError("The P-EAGLE produced proposals do not reconcile with verification and accepted output.")
                if not row.get("scheduled_proposal_contains_placeholders") and row.get("scheduled_proposal_token_ids") != tokens:
                    raise EvidenceError("The P-EAGLE scheduled proposals differ from their verified producer.")
                draft = phase(producer, "draft_proposal")
                verify = phase(consumer, "verification_and_sampling")
                if producer["clock_implementation"] != consumer["clock_implementation"] or producer["synchronized_cpu_end_ns"] > consumer["outer"]["cpu_start_ns"]:
                    raise EvidenceError("The P-EAGLE producer and verification clocks or ordering do not reconcile.")
                start = draft["cpu_start_ns"]
            else:
                producer = consumer
                draft = phase(consumer, "draft_proposal" if configuration == "eagle3" else "draft")
                verify = phase(consumer, "target_verification" if configuration == "eagle3" else "verify")
                if draft["cpu_end_ns"] > verify["cpu_start_ns"]:
                    raise EvidenceError("The recorded SGLang draft must complete its host call before verification begins.")
                start = consumer["outer"]["cpu_start_ns"]
                if row.get("native_postupdate_token_ids") != row["committed_token_ids"]:
                    raise EvidenceError("The final SGLang state update does not confirm the committed token sequence.")
            if start >= end:
                raise EvidenceError("A complete diagnostic round requires an increasing time interval.")
            intervals.append({"request_id": request_id, "round": row["round"], "started_ns": start, "ended_ns": end,
                              "producer_observation_id": producer["observation_id"], "consumer_observation_id": consumer["observation_id"],
                              "round_seconds": (end - start) / 1e9, "draft_seconds": draft["gpu_elapsed_seconds"],
                              "verification_seconds": verify["gpu_elapsed_seconds"],
                              "draft_operation": draft["name"], "verification_operation": verify["name"],
                              "draft_stream_id": draft["stream_id"], "verification_stream_id": verify["stream_id"]})
        if configuration == "peagle":
            native, source = self.native(configuration, benchmark, repetition, diagnostic=True)
            totals = native["totals"]
            if totals["num_accepted_tokens"] != sum(row["accepted_draft_tokens"] for row in trace["rows"]) or totals["num_draft_tokens"] != sum(row["proposed_draft_tokens"] for row in trace["rows"]):
                raise EvidenceError("The P-EAGLE timed rounds and native counters disagree on proposal or acceptance totals.")
            sources.add(source)
        else:
            for request_id, request in requests.items():
                rows = [row for row in trace["rows"] if row["request_id"] == request_id]
                native = request.get("server_metadata") or {}
                if (native.get("spec_verify_ct") != len(rows)
                        or native.get("spec_num_correct_drafts") != sum(row["accepted_draft_tokens"] for row in rows)
                        or native.get("spec_num_proposed_drafts") != sum(row["proposed_draft_tokens"] for row in rows)):
                    raise EvidenceError("The timed SGLang rounds and native request counters do not reconcile.")
        overlap_pairs = 0
        for request_id in requests:
            ordered = sorted((item for item in intervals if item["request_id"] == request_id), key=lambda item: item["started_ns"])
            overlap_pairs += sum(left["ended_ns"] > right["started_ns"] for left, right in zip(ordered, ordered[1:]))
        values = {name: statistics.mean(item[field] for item in intervals) * 1000
                  for name, field in (("Round latency", "round_seconds"), ("Draft latency", "draft_seconds"), ("Verification latency", "verification_seconds"))}
        details = {
            "scope": "Separate diagnostic verification rounds", "verification_rounds": len(intervals),
            "clock": "Host monotonic timestamps for round latency; CUDA events for component intervals",
            "overlapping_adjacent_round_pairs": overlap_pairs,
            "per_request_round_counts": {request_id: sum(item["request_id"] == request_id for item in intervals) for request_id in requests},
            "draft_operations": sorted({item["draft_operation"] for item in intervals}),
            "verification_operations": sorted({item["verification_operation"] for item in intervals}),
            "round_latency_range_seconds": [min(item["round_seconds"] for item in intervals), max(item["round_seconds"] for item in intervals)],
            "instrumentation_note": "Diagnostic intervals include synchronization and instrumentation overhead. P-EAGLE links a draft to its later verification and final scheduler update. Consecutive intervals can overlap. Component event durations describe their named operations, and their sum does not define total request time.",
            "draft_scope": "The draft interval measures proposal generation. EAGLE-3 draft extension and other state work remain inside the full round interval.",
        }
        self.component_cache[cache_key] = (values, sorted(sources), details)
        return values[variant], sorted(sources), details

    def observation(self, metric, variant, configuration, benchmark, repetition, position):
        if configuration in self.deferred:
            raise EvidenceError(self.methods[configuration] + " is deferred to a later evaluation. Its current measurement cells remain empty.")
        run = self.runs[configuration]
        key = (benchmark, repetition)
        if key not in run["blocks"]:
            raise EvidenceError(run.get("error") or run["errors"].get(key, "The completed benchmark block is unavailable."))
        block = run["blocks"][key]
        sources = [block["source"]]
        details = {"requests": 3, "request_ids": [row["request_id"] for row in block["requests"]]}
        if metric == "metric-otps":
            value = block["metrics"]["otps"]
            details.update(output_tokens=block["metrics"]["output_tokens"], elapsed_seconds=block["metrics"]["elapsed_seconds"])
        elif (metric, variant) in TIMING:
            field, statistic, scale = TIMING[(metric, variant)]
            observed = block["metrics"][field]
            if observed[statistic] is None:
                raise EvidenceError("The benchmark block has no observations eligible for this metric.")
            value = observed[statistic] * scale
            details.update(eligible_observations=observed["n"], observation_unit="gaps" if field == "inter_chunk_seconds" else "requests")
        elif metric in {"metric-speedup", "metric-quality"}:
            reference = self.configurations[configuration][1]
            left, right = self.pair(run, self.runs[reference], benchmark, repetition)
            sources.append(right["source"])
            details["reference"] = self.methods[reference]
            if metric == "metric-speedup":
                if right["metrics"]["otps"] <= 0:
                    raise EvidenceError("The AR reference has no positive output throughput for this repetition.")
                value = left["metrics"]["otps"] / right["metrics"]["otps"]
                details.update(method_otps=left["metrics"]["otps"], reference_otps=right["metrics"]["otps"])
            else:
                matches = [a["output_token_ids"] == b["output_token_ids"] and a["output_tokens"] == b["output_tokens"]
                           and a["finish_reason"] == b["finish_reason"] for a, b in zip(left["requests"], right["requests"])]
                value = sum(matches) / 3 * 100
                details.update(matching_requests=sum(matches), checked_requests=3)
                checks = []
                for a, b, match in zip(left["requests"], right["requests"], matches):
                    differing = next((index for index, pair in enumerate(zip(a["output_token_ids"], b["output_token_ids"])) if pair[0] != pair[1]), None)
                    if differing is None and len(a["output_token_ids"]) != len(b["output_token_ids"]):
                        differing = min(len(a["output_token_ids"]), len(b["output_token_ids"]))
                    checks.append({"prompt_id": a["prompt_id"], "exact_output_agreement": match,
                                   "first_different_token_index": differing,
                                   "method_output_tokens": a["output_tokens"],
                                   "reference_output_tokens": b["output_tokens"],
                                   "method_finish_reason": a["finish_reason"],
                                   "reference_finish_reason": b["finish_reason"],
                                   "delivered_token_ids_match": a["output_token_ids"] == b["output_token_ids"],
                                   "output_length_match": a["output_tokens"] == b["output_tokens"],
                                   "stopping_reason_match": a["finish_reason"] == b["finish_reason"]})
                details["agreement_checks"] = checks
        elif metric == "metric-memory":
            if variant == "Sampled whole-device memory":
                value, sample_count, coverage = self.memory_peak(configuration, block)
                sources.extend(self.memory_sample_files)
                details.update(memory_sample_count=sample_count, scope="Timed benchmark interval", **coverage)
            else:
                value, sources, memory_details = self.framework_memory(configuration, benchmark, repetition, variant)
                details.update(memory_details)
        elif metric in {"metric-gpu-time", "metric-per-gpu-throughput"}:
            gpu_seconds, source, allocation_details = self.timed_allocation(configuration, block)
            sources.append(source)
            details.update(allocation_details)
            value = gpu_seconds / block["metrics"]["output_tokens"] * 1_000_000 if metric == "metric-gpu-time" else block["metrics"]["output_tokens"] / gpu_seconds
        elif metric == "metric-actual-charge":
            raise EvidenceError("A matching billed interval, net charge, currency and delivered token count are unavailable.")
        elif metric == "metric-component-time":
            value, sources, component_details = self.component_time(configuration, benchmark, repetition, variant)
            details.update(component_details)
        elif metric in POSITION_METRICS and configuration == "eagle3":
            if variant != TREE_VARIANT:
                raise EvidenceError("EAGLE-3 uses a branching proposal tree. Its measured depth statistics appear in the separate Tree-prefix diagnostic variant. The paper's chain-based n-alpha statistic requires its own chain evaluation.")
            value, sources, tree_details = self.eagle_tree_observation(metric, benchmark, repetition, position)
            details.update(tree_details)
        elif metric in DRAFT_METRICS:
            if configuration in {"dflash", "eagle3"} and any("spec_verify_ct" in (row.get("server_metadata") or {}) for row in block["requests"]):
                value, counter_details = self.sglang_counts(configuration, block, metric, position)
                details.update(counter_details)
            elif metric in {"metric-draft-fraction", "metric-accepted-round"} and configuration in {"peagle", "dspark"}:
                native, source = self.native(configuration, benchmark, repetition)
                counts = native["totals"]
                denominator = counts["num_draft_tokens"] if metric == "metric-draft-fraction" else counts["num_drafts"]
                if denominator == 0:
                    raise EvidenceError("The native counter interval has no eligible proposals or rounds.")
                value = counts["num_accepted_tokens"] / denominator * (100 if metric == "metric-draft-fraction" else 1)
                sources.append(source)
                details.update(counts, scope="Primary native counter interval")
            else:
                trace = self.rounds(configuration, benchmark, repetition)
                rows = trace["rows"]
                sources = trace["sources"]
                totals = {field: sum(row[field] for row in rows) for field in (
                    "proposed_draft_tokens", "accepted_draft_tokens", "committed_tokens", "delivered_tokens")}
                details.update(totals, verification_rounds=len(rows), initial_target_tokens=trace["initial_tokens"], scope=trace.get("scope", "Separate diagnostic run"))
                details["eligible_requests_with_rounds"] = len(trace["request_means"])
                if "initial_delivered_tokens" in trace:
                    details["initial_delivered_tokens"] = trace["initial_delivered_tokens"]
                if metric == "metric-acceptance-length":
                    value = totals["committed_tokens"] / len(rows)
                elif metric == "metric-acceptance-length-request":
                    value = statistics.mean(trace["request_means"])
                elif metric == "metric-accepted-round":
                    value = totals["accepted_draft_tokens"] / len(rows)
                elif metric == "metric-returned-round":
                    value = totals["delivered_tokens"] / len(rows)
                elif metric == "metric-draft-fraction":
                    if totals["proposed_draft_tokens"] == 0:
                        raise EvidenceError("The diagnostic rounds contain no proposed draft tokens.")
                    value = totals["accepted_draft_tokens"] / totals["proposed_draft_tokens"] * 100
                elif metric in {"metric-prefix-survival", "metric-conditional-acceptance"}:
                    if any(row.get("depth_metrics_available") is not True or not integer(row.get("proposed_depth"))
                           or not integer(row.get("accepted_prefix_depth")) or row["accepted_prefix_depth"] > row["proposed_depth"] for row in rows):
                        raise EvidenceError("The round trace does not establish comparable proposal depths and accepted prefixes for every round.")
                    eligible = [row for row in rows if row["proposed_depth"] >= position]
                    if metric == "metric-conditional-acceptance":
                        eligible = [row for row in eligible if row["accepted_prefix_depth"] >= position - 1]
                    if not eligible:
                        raise EvidenceError("No recorded verification round is eligible at this draft position.")
                    accepted = sum(row["accepted_prefix_depth"] >= position for row in eligible)
                    value = accepted / len(eligible) * (100 if metric == "metric-conditional-acceptance" else 1)
                    details.update(position=position, eligible_rounds=len(eligible), accepted_rounds=accepted)
                else:
                    raise EvidenceError("The exporter has no validated observation mapping for this metric.")
        else:
            raise EvidenceError("The exporter has no validated observation mapping for this metric.")
        if not finite(value):
            raise EvidenceError("The metric calculation requires a finite, nonnegative result.")
        return {"repetition": repetition, "value": value, "sources": sorted(set(sources)), **details}

    def cell(self, metric, variant, configuration, benchmark, position):
        observations, missing = [], []
        for repetition in range(1, 4):
            try:
                observations.append(self.observation(metric, variant, configuration, benchmark, repetition, position))
            except (OSError, ValueError, KeyError, TypeError, AttributeError, ZeroDivisionError) as error:
                missing.append({"repetition": repetition, "reason": str(error)})
        values = [row["value"] for row in observations]
        sources = sorted({path for row in observations for path in row["sources"]})
        evidence = "; ".join((self.prefix + "/" + quote(path, safe="/") if self.prefix else path) for path in sources)
        unit = "ms/request" if metric in {"metric-completion", "metric-ttft"} else "ms/round" if metric == "metric-component-time" else self.metrics[metric]["unit"]
        cell = {"metricId": metric, "variant": variant, "method": self.methods[configuration],
                "benchmark": BENCHMARKS[benchmark], "position": position,
                "value": statistics.mean(values) if values else None,
                "sd": statistics.stdev(values) if len(values) > 1 else None,
                "n": len(values), "unit": unit, "evidence": evidence}
        if missing:
            reasons = list(dict.fromkeys(row["reason"] for row in missing))
            cell["reason"] = " ".join(reasons)
        self.cell_details.append({"key": {key: cell[key] for key in ("metricId", "variant", "method", "benchmark", "position")},
                                  "observations": observations, "missing_repetitions": missing})
        return cell

    def export(self):
        self.prepare()
        cells = []
        for metric in self.metrics:
            eligible = [name for name in self.methods if self.configurations[name][1] is not None] if metric in DRAFT_METRICS | {"metric-speedup", "metric-quality"} else list(self.methods)
            positions = range(1, 10) if metric in {"metric-prefix-survival", "metric-conditional-acceptance"} else (None,)
            for variant in VARIANTS.get(metric, ("",)):
                for configuration in eligible:
                    if configuration in self.deferred:
                        continue
                    if variant == TREE_VARIANT and configuration != "eagle3":
                        continue
                    for benchmark in BENCHMARKS:
                        for position in positions:
                            cells.append(self.cell(metric, variant, configuration, benchmark, position))
        blocks = sum(len(run["blocks"]) for configuration, run in self.runs.items() if configuration not in self.deferred)
        expected_blocks = (len(self.methods) - len(self.deferred)) * 30
        result = {"trialId": self.root.name, "status": "complete" if blocks == expected_blocks else "partial" if blocks else "prepared", "cells": cells,
                  "deferredMethods": [self.methods[name] for name in self.methods if name in self.deferred]}
        manifest = {
            "schema_version": 1, "trial_id": self.root.name,
            "catalogue_sha256": hashlib.sha256(self.catalogue_path.read_bytes()).hexdigest(),
            "primary_completed_blocks": blocks, "primary_expected_blocks": expected_blocks,
            "primary_expected_timed_requests": expected_blocks * 3,
            "deferred_configurations": sorted(self.deferred),
            "deferred_expected_blocks": len(self.deferred) * 30,
            "deferred_expected_timed_requests": len(self.deferred) * 90,
            "n_definition": "Each cell's n counts eligible repetitions. Each repetition uses three prompts. The value is the mean of the repetition estimates, and sd is their sample standard deviation.",
            "acceptance_definition": "Each repetition pools committed tokens and verification rounds before calculating mean acceptance length. The numerator includes retained target tokens before final output trimming.",
            "scope_note": ("Primary request records supply timing and verification-round counts. Detailed phase timing and memory use separate diagnostic runs. " if self.team else
                           "Primary timing uses the primary request records. Acceptance uses validated native counters or separate diagnostic runs with matching prompts and settings. ") +
                          "GPU efficiency uses timed benchmark intervals and a matching final Slurm allocation. Whole-job totals retain a separate scope.",
            "counter_semantics": {"sglang_revision": SGLANG_REVISION, "source": SGLANG_COUNTER_SOURCE,
                                  "committed_token_formula": "Accepted draft tokens plus verification rounds, before output trimming and excluding the initial target token."},
            "tree_variant_definitions": {"variant": TREE_VARIANT, "definitions": TREE_DEFINITIONS,
                                         "paper_convention": TREE_CONVENTION},
            "pending_site_rows": [],
            "memory_sample_file": self.memory_samples,
            "memory_sample_files": self.memory_sample_files,
            "allocation_records": [{"source": source, **record} for record, source in self.allocation_cache.values()],
            "configuration_provenance": [{"configuration": configuration, "method": self.methods[configuration],
                                          "source": self.relative(run["directory"] / "run.json"),
                                          "artifact_origin": run["metadata"].get("artifact_origin"),
                                          "slurm_job_id": run["metadata"]["slurm_job_id"],
                                          "hostname": run["metadata"]["hostname"], "gpu_uuid": run["metadata"]["gpu_uuid"],
                                          "completed_blocks": len(run["blocks"])}
                                         for configuration, run in self.runs.items() if run["metadata"] is not None],
            "source_files": self.files, "cells": self.cell_details,
        }
        return result, manifest


def latex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                    "·": r"\textperiodcentered{}", "τ": r"\(\tau\)", "×": r"\(\times\)",
                    "’": "'", "‘": "`", "“": "``", "”": "''", "–": "--", "—": "---"}
    return "".join(replacements.get(character, character) for character in str(value))


def latex_tables(result, metrics):
    lookup = {(cell["metricId"], cell["variant"], cell["method"], cell["benchmark"], cell["position"]): cell for cell in result["cells"]}
    winners = best_observed_cells(result["cells"])
    active_methods = {cell["method"] for cell in result["cells"]}
    output = [r"% This file requires the array and booktabs packages and uses full-width benchmark panels.",
              "% The three benchmark panels share one caption for each metric."]
    for metric in MAIN_METRICS:
        controls = metric in {"metric-otps", "metric-gpu-time"}
        rows = [METHODS[key] for key in METHODS if METHODS[key] in active_methods and (controls or CONFIGURATIONS[key][1] is not None)]
        rows += [name for configuration, name in TEAM_METHODS.items() if controls or TEAM_CONFIGURATIONS[configuration][1] is not None]
        title = f"{metrics[metric]['name']} ({metrics[metric]['unit']})."
        output += [r"\begin{table*}[!tp]", r"\centering", r"\small",
                   r"\setlength{\tabcolsep}{3pt}", r"\caption{" + latex_escape(title) + "}",
                   r"\label{tab:" + metric.removeprefix("metric-") + "}"]
        for panel_title, benchmarks in BENCHMARK_PANELS:
            output += [r"\textit{" + panel_title + r"}\par\smallskip",
                       r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.7cm}" + "r" * len(benchmarks) + "@{}}", r"\toprule",
                       "Method & " + " & ".join(latex_escape(name) for name in benchmarks) + r" \\", r"\midrule"]
            for method in rows:
                values = []
                for benchmark in benchmarks:
                    cell = lookup.get((metric, "", method, benchmark, None))
                    values.append(latex_value(cell, cell_identity(cell) in winners if cell else False))
                label = method.replace(" (pilot)", "")
                output.append(latex_escape(label) + " & " + " & ".join(values) + r" \\")
            output += [r"\bottomrule", r"\end{tabular}", r"\par\medskip"]
        output += [r"\begin{minipage}{\linewidth}\footnotesize",
                   "Brackets give the number of eligible repetitions. With two or more repetitions, entries give the mean and sample standard deviation. "
                   "A single repetition supplies one observed value. Each repetition uses the same three prompts. Empty cells have no validated measurement. "
                   "The target is Meta Llama-3.1-8B-Instruct on an NVIDIA L40S GPU. Generation uses greedy decoding, one active request, and a 2,048-token output limit. "
                   "Pilot FFN rows identify the notebook's P-EAGLE-style drafter at feed-forward widths of 100\%, 75\%, and 50\%. "
                   "Run metadata identifies each checkpoint's origin. "
                   "Their AR reference uses Transformers. The P-EAGLE row uses the H-Spec reproduction. "
                   "Acceptance uses validated native counters or matched diagnostic runs. The evidence records identify each measurement scope. "
                   "Committed tokens are counted before output trimming.",
                   latex_escape(TIMED_RESOURCE_NOTE) if metric == "metric-gpu-time" else "",
                   latex_escape(BOLD_NOTE + (" Lower values are preferred." if metric in LOWER_IS_BETTER else " Higher values are preferred.")),
                   r"\end{minipage}", r"\end{table*}", ""]
    return "\n".join(output)


def latex_number(value, precision):
    """Keep ordinary measurements readable and format extreme magnitudes as powers of ten."""
    if value == 0:
        return "0"
    exponent = math.floor(math.log10(abs(value)))
    if -4 <= exponent < 6:
        digits = max(0, precision - exponent - 1)
        rounded = float(f"{value:.{precision}g}")
        result = f"{rounded:.{digits}f}"
        return result.rstrip("0").rstrip(".") if "." in result else result
    mantissa, power = f"{value:.{precision - 1}e}".split("e")
    return mantissa.rstrip("0").rstrip(".") + r"\times 10^{" + str(int(power)) + "}"


def cell_identity(cell):
    return tuple(cell[key] for key in ("metricId", "variant", "method", "benchmark", "position"))


def best_observed_cells(cells):
    """Rank observed means by benchmark while preserving distinct denominator definitions."""
    groups = defaultdict(list)
    for cell in cells:
        if cell["value"] is None or cell["n"] < 1 or cell["metricId"] in DESCRIPTIVE_METRICS:
            continue
        structure = "tree" if cell["method"] == "EAGLE-3" else "linear"
        structure = structure if cell["metricId"] in POSITION_METRICS | {"metric-draft-fraction"} else "shared"
        group = (cell["metricId"], cell["variant"], cell["benchmark"], cell["position"], structure)
        groups[group].append(cell)
    winners = set()
    for group, eligible in groups.items():
        if len(eligible) < 2:
            continue
        best = (min if group[0] in LOWER_IS_BETTER else max)(cell["value"] for cell in eligible)
        best_display = mean_display({**eligible[0], "value": best})
        winners.update(cell_identity(cell) for cell in eligible if mean_display(cell) == best_display)
    return winners


def mean_display(cell):
    value = cell["value"]
    if cell["metricId"] in {"metric-otps", "metric-speedup"} and 0.01 <= abs(value) < 100000:
        return f"{value:.2f}"
    return latex_number(value, 4)


def latex_value(cell, bold=False):
    if cell is None or cell["value"] is None:
        return ""
    value = mean_display(cell)
    if bold:
        value = r"\mathbf{" + value + "}"
    if cell["sd"] is not None:
        value += r" \pm " + latex_number(cell["sd"], 2)
    return r"\(" + value + r"\;[" + str(cell["n"]) + r"]\)"


def latex_appendix_tables(result, metrics):
    """Include only metric variants, rows, and panels with measured observations."""
    output = ["% Generated appendix tables contain measured observations only."]
    ordering = list(METHODS.values()) + list(TEAM_ROWS)
    winners = best_observed_cells(result["cells"])
    for metric, definition in metrics.items():
        if metric in MAIN_METRICS:
            continue
        for variant in VARIANTS.get(metric, ("",)):
            available = [cell for cell in result["cells"] if cell["metricId"] == metric and cell["variant"] == variant and cell["value"] is not None]
            positions = sorted({cell["position"] for cell in available}, key=lambda value: value or 0)
            for position in positions:
                selected = [cell for cell in available if cell["position"] == position]
                rows = [name for name in ordering if any(cell["method"] == name for cell in selected)]
                lookup = {(cell["method"], cell["benchmark"]): cell for cell in selected}
                subtitle = [variant] if variant else []
                if position is not None:
                    subtitle.append(f"Position {position}")
                title = definition["name"] + ("; " + "; ".join(subtitle) if subtitle else "") + " (" + selected[0]["unit"] + ")."
                identity = re.sub(r"[^a-z0-9]+", "-", (metric + "-" + variant + "-" + str(position)).lower()).strip("-")
                output += [r"\begin{table*}[!htp]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
                           r"\caption{" + latex_escape(title) + "}", r"\label{tab:appendix-" + identity + "}"]
                for panel_title, group in BENCHMARK_PANELS:
                    columns = [name for name in group if any(cell["benchmark"] == name for cell in selected)]
                    if not columns:
                        continue
                    output += [r"\textit{" + panel_title + r"}\par\smallskip",
                               r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.7cm}" + "r" * len(columns) + "@{}}", r"\toprule",
                               "Method & " + " & ".join(latex_escape(name) for name in columns) + r" \\", r"\midrule"]
                    for method in rows:
                        output.append(latex_escape(method.replace(" (pilot)", "")) + " & " +
                                      " & ".join(latex_value(lookup.get((method, benchmark)), cell_identity(lookup[(method, benchmark)]) in winners)
                                                 if (method, benchmark) in lookup else "" for benchmark in columns) + r" \\")
                    output += [r"\bottomrule", r"\end{tabular}\par\medskip"]
                explanation = TREE_DEFINITIONS[metric] + " " + TREE_CONVENTION if variant == TREE_VARIANT else definition["definition"]
                if metric == "metric-component-time":
                    explanation += " " + COMPONENT_TIMING_NOTE
                if metric == "metric-draft-fraction":
                    explanation += " " + PROPOSAL_DENOMINATOR_NOTE
                if metric in {"metric-gpu-time", "metric-per-gpu-throughput"}:
                    explanation += " " + TIMED_RESOURCE_NOTE
                output += [r"\begin{minipage}{\linewidth}\footnotesize", latex_escape(explanation),
                           latex_escape(COMPONENT_COMPARISON_NOTE if metric in DESCRIPTIVE_METRICS else BOLD_NOTE + (" Lower values are preferred." if metric in LOWER_IS_BETTER else " Higher values are preferred.") +
                                        (" Tree and linear proposal denominators form separate comparison groups." if metric in POSITION_METRICS | {"metric-draft-fraction"} else "")),
                           "Brackets give eligible repetition counts. Standard deviation appears when at least two repetitions are available. "
                           "Each repetition uses three prompts. The evidence records preserve observation counts and measurement scopes. "
                           "Empty cells lack an eligible measured value.", r"\end{minipage}", r"\end{table*}", ""]
    if len(output) == 1:
        output.append("The current evidence contains no measured appendix metrics.")
    return "\n".join(output)


def latex_definitions(metrics):
    output = ["Each table uses the following measurement definition. Repetition estimates are summarized by their mean and sample standard deviation."]
    for item in metrics.values():
        output += [r"\subsection{" + latex_escape(item["name"]) + "}",
                   r"\textit{Unit: " + latex_escape(item["unit"]) + ".} " + latex_escape(item["definition"])]
    return "\n\n".join(output)


def latex_availability(result, metrics):
    output = ["Each empty cell has a reason in the accompanying primary or team measurement-evidence JSON record. "
              "Those records also identify the source files, hashes, repetition values, and eligible observations."]
    for metric, definition in metrics.items():
        for variant in VARIANTS.get(metric, ("",)):
            cells = [cell for cell in result["cells"] if cell["metricId"] == metric and cell["variant"] == variant]
            measured = sum(cell["value"] is not None for cell in cells)
            missing = [cell for cell in cells if cell["value"] is None]
            if not missing:
                continue
            title = definition["name"] + ("; " + variant if variant else "")
            output += [r"\paragraph*{" + latex_escape(title) + "}", f"{measured} of {len(cells)} cells contain validated measurements."]
            if metric in {"metric-gpu-time", "metric-per-gpu-throughput"}:
                explanation = "Each remaining value requires a completed benchmark block and a final Slurm record that validates the matching parent job and allocated GPU count."
            elif metric == "metric-actual-charge":
                explanation = "A matching billed interval, net GPU charge, currency, and delivered-token count are unavailable."
            elif metric == "metric-component-time":
                explanation = "The available raw worker intervals require validation against complete verification rounds, including the final state update."
            elif metric == "metric-memory" and variant != "Sampled whole-device memory":
                explanation = "Each value requires a synchronized framework memory peak with a reset scope covering all three diagnostic requests."
            elif metric in {"metric-prefix-survival", "metric-conditional-acceptance"}:
                explanation = "Each position requires a validated proposal depth and a positive eligible-round count. Methods with shorter proposals leave later positions empty. Tree proposals require an explicit path interpretation."
            elif metric in DRAFT_METRICS:
                explanation = "Each value requires validated native counts or a matching complete diagnostic trace."
            else:
                explanation = "Remaining cells require completed benchmark records that pass the metric's checks. Streaming metrics also require eligible output chunks."
            output.append(explanation)
    return "\n\n".join(output)


def latex_findings(primary, primary_manifest, team, team_manifest):
    output = [r"\subsection{Available observations}"]
    for name, manifest in (("initial", primary_manifest), ("team", team_manifest)):
        output.append(f"The {name} record set contains {manifest['primary_completed_blocks']} of {manifest['primary_expected_blocks']} planned benchmark blocks. "
                      "Each complete block contains three request executions.")
    completed_requests = primary_manifest["primary_completed_blocks"] * 3
    output.append(f"The current initial scope contains {completed_requests} completed timed request executions out of "
                  f"{primary_manifest['primary_expected_blocks'] * 3} planned executions.")
    output.append("The 30 unique prompts recur across three timing repetitions. A complete configuration therefore contains 90 timed request executions.")
    artifacts = [row for row in team_manifest.get("configuration_provenance", []) if row["configuration"] != "team-ar"]
    if not artifacts:
        output.append("The team checkpoint measurements remain pending. The current records contain no validated team drafter run.")
    for artifact in artifacts:
        origin = {"teammate-original": "uses an imported original team checkpoint",
                  "local-reproduction": "uses a newly trained local reproduction checkpoint"}.get(artifact["artifact_origin"],
                  "has an unresolved checkpoint origin in the available run metadata")
        output.append(latex_escape(artifact["method"]) + " " + origin + ". The run evidence preserves available training provenance and leaves unknown fields unresolved.")
    all_cells = primary["cells"] + team["cells"]
    diagnostics = [cell for cell in all_cells if cell["metricId"] == "metric-component-time"
                   and cell["variant"] == "Round latency" and cell["value"] is not None]
    if diagnostics:
        output.append(r"\subsection{Saved diagnostic coverage}")
        for method in dict.fromkeys(cell["method"] for cell in diagnostics):
            values = [cell for cell in diagnostics if cell["method"] == method]
            repetitions = sum(cell["n"] for cell in values)
            output.append(latex_escape(method) + f" has validated complete-round timing for {repetitions} benchmark repetitions "
                          f"across {len(values)} benchmarks. Those intervals cover {repetitions * 3} diagnostic request executions.")
        output.append("These separate diagnostic executions include synchronization and trace-construction overhead. "
                      "Primary requests supply the throughput and response-time tables. The appendix reports the eligible diagnostic measurements and their repetition counts.")
    output.append(r"\subsection{Throughput and paired speedup}")
    for method in list(METHODS.values()) + list(TEAM_ROWS):
        values = [cell for cell in all_cells if cell["metricId"] == "metric-speedup" and cell["method"] == method and cell["value"] is not None]
        if values:
            lower, upper = min(values, key=lambda cell: cell["value"]), max(values, key=lambda cell: cell["value"])
            output.append(latex_escape(method) + f" has benchmark mean speedups from {lower['value']:.3f}" + r"\(\times\) on " +
                          latex_escape(lower["benchmark"]) + f" to {upper['value']:.3f}" + r"\(\times\) on " + latex_escape(upper["benchmark"]) +
                          f" across {len(values)} measured benchmarks. Table entries retain the repetition counts and standard deviations.")
    output.append("The throughput ratio describes delivered output per elapsed second under the recorded prompts and settings. Response length and output-agreement results define the scope of each comparison.")
    measured_team = [name for configuration, name in TEAM_METHODS.items() if configuration != "team-ar"
                     and any(cell["method"] == name and cell["metricId"] == "metric-otps" and cell["value"] is not None for cell in team["cells"])]
    if measured_team:
        output.append(r"\subsection{Team pilot observations}")
        for method in measured_team:
            throughput = [cell for cell in team["cells"] if cell["method"] == method and cell["metricId"] == "metric-otps" and cell["value"] is not None]
            low, high = min(throughput, key=lambda cell: cell["value"]), max(throughput, key=lambda cell: cell["value"])
            output.append(latex_escape(method) + f" has observed benchmark mean OTPS from {low['value']:.2f} tokens/s on " +
                          latex_escape(low["benchmark"]) + f" to {high['value']:.2f} tokens/s on " + latex_escape(high["benchmark"]) +
                          f" across {len(throughput)} benchmarks. These measurements use the team's recorded Transformers implementation.")
            paired = [cell for cell in team["cells"] if cell["method"] == method and cell["metricId"] == "metric-speedup" and cell["value"] is not None]
            if len(paired) < len(throughput):
                output.append(f"The active AR reference supports paired speedup on {len(paired)} of these {len(throughput)} benchmarks. "
                              "Compatibility checks leave the remaining speedup and exact-agreement cells empty. The measured OTPS remains available for each completed method block.")
            acceptance = [cell for cell in team["cells"] if cell["method"] == method and cell["metricId"] == "metric-acceptance-length" and cell["value"] is not None]
            if acceptance:
                low, high = min(acceptance, key=lambda cell: cell["value"]), max(acceptance, key=lambda cell: cell["value"])
                output.append(latex_escape(method) + f" has benchmark mean acceptance length from {low['value']:.3f} to {high['value']:.3f} committed tokens per verification round. "
                              "The calculation uses the saved primary round counts before final output trimming.")
        if len(measured_team) < 3:
            output.append(f"The current saved records contain measurements for {len(measured_team)} of the three team widths. "
                          "A comparison of feed-forward widths requires completed measurements for the remaining variants and their matched reference.")
    output.append(r"\subsection{Exact response agreement}")
    any_mismatch = False
    for method in list(METHODS.values()) + list(TEAM_ROWS):
        observations = [observation for manifest in (primary_manifest, team_manifest) for cell in manifest["cells"]
                        if cell["key"]["metricId"] == "metric-quality" and cell["key"]["method"] == method
                        for observation in cell["observations"]]
        if not observations:
            continue
        checked = sum(row["checked_requests"] for row in observations)
        matched = sum(row["matching_requests"] for row in observations)
        per_prompt = defaultdict(list)
        length_checks = defaultdict(list)
        differences = []
        for observation in observations:
            for check in observation.get("agreement_checks", []):
                per_prompt[check["prompt_id"]].append(check["exact_output_agreement"])
                if "method_output_tokens" in check and "reference_output_tokens" in check:
                    length_checks[check["prompt_id"]].append((check["method_output_tokens"], check["reference_output_tokens"]))
                if check["first_different_token_index"] is not None:
                    differences.append(check["first_different_token_index"])
        paragraph = latex_escape(method) + f" matches its AR reference on {matched} of {checked} checked request executions ({matched / checked * 100:.1f}" + r"\%)."
        if per_prompt:
            consistent = sum(all(values) for values in per_prompt.values())
            paragraph += f" All observed repetitions match for {consistent} of {len(per_prompt)} distinct prompts."
            if all(len(set(values)) == 1 for values in per_prompt.values()):
                paragraph += " Each prompt has the same agreement outcome across its observed repetitions."
        if differences:
            paragraph += f" The earliest recorded token divergence occurs at returned token {min(differences) + 1}."
        if length_checks:
            equal_length = sum(all(left == right for left, right in pairs) for pairs in length_checks.values())
            paragraph += f" Output lengths match in every observed repetition for {equal_length} of {len(length_checks)} distinct prompts."
            if all(len(set(pairs)) == 1 for pairs in length_checks.values()):
                unique_pairs = [pairs[0] for pairs in length_checks.values()]
                method_total = sum(left for left, _ in unique_pairs)
                reference_total = sum(right for _, right in unique_pairs)
                paragraph += (f" Across one response per distinct prompt, the method delivers {method_total:,} tokens and its AR reference delivers {reference_total:,}. "
                              "Each observed repetition has these same per-prompt lengths.")
        output.append(paragraph)
        any_mismatch |= matched != checked
    if any_mismatch:
        output.append("The response mismatches are a material finding. Their cause remains unresolved. "
                      "The measured throughput gains describe the recorded executions. An exact-response-preservation claim requires resolving these mismatches. "
                      "A causal explanation requires checking target scores, input token IDs, verification behavior, and cache state at the first divergent position.")
    output.append("Benchmark task accuracy remains unmeasured. Exact token agreement evaluates reproduction of the AR response. "
                  "Math correctness, code pass@1, summarization quality, and conversational judging require separate scoring procedures.")
    output.append(latex_allocations(primary_manifest, team_manifest))
    return "\n\n".join(output)


def latex_allocations(*manifests):
    records, configurations = {}, defaultdict(set)
    for manifest in manifests:
        for item in manifest.get("configuration_provenance", []):
            configurations[str(item["slurm_job_id"])].add(item["configuration"])
        for record in manifest.get("allocation_records", []):
            job = str(record["job_id"])
            if job in records and records[job] != record:
                raise EvidenceError("The report has conflicting final records for the same parent allocation.")
            records[job] = record
    if not records:
        return ""
    output = [r"\begin{table*}[!htp]", r"\centering\small",
              r"\caption{Allocated compute resources for the evaluation jobs.}", r"\label{tab:allocations}",
              r"\begin{tabular}{@{}lp{5.0cm}lrr@{}}", r"\toprule",
              r"Job ID & Purpose & GPU model / count & GPU-hours & CPU-hours \\", r"\midrule"]
    for job, record in sorted(records.items()):
        names = configurations[job]
        initial = any(name in METHODS for name in names)
        pilot = any(name in TEAM_METHODS for name in names)
        purpose = "Released-method evaluation and the original pilot attempt." if initial and pilot else "Released-method evaluation." if initial else "Pilot width evaluation and its AR reference."
        cpus = re.search(r"(?:^|,)cpu=([0-9]+)(?:,|$)", str(record.get("alloc_tres", "")))
        cpu_hours = f"{record['elapsed_seconds'] * int(cpus.group(1)) / 3600:.4f}" if cpus else ""
        output.append(latex_escape(job) + " & " + latex_escape(purpose) + " & NVIDIA L40S / " + str(record["gpu_count"]) +
                      f" & {record['allocated_gpu_seconds'] / 3600:.4f} & {cpu_hours}" + r" \\")
    output += [r"\bottomrule\end{tabular}", r"\par\smallskip\begin{minipage}{\linewidth}\footnotesize",
               "Each parent allocation appears once. These totals include all elapsed allocation time, including setup, model loading, qualification, warmups, recorded requests, and incomplete attempts. "
               "The table covers the evaluation allocations used by these result rows. The project compute ledger separately preserves earlier preparation jobs. "
               "Final Slurm records and their hashes accompany the exported evidence. CPU-hours count allocated CPU cores multiplied by elapsed hours.",
               r"\end{minipage}\end{table*}"]
    return "\n".join(output)


def write_export(exporter, destination, report_output=None):
    result, manifest = exporter.export()
    team_exporter = Exporter(exporter.root, exporter.catalogue_path, exporter.prefix, team=True, memory_samples=exporter.memory_samples,
                             additional_memory_samples=exporter.memory_sample_files[1:],
                             include_deferred=exporter.include_deferred)
    team_result, team_manifest = team_exporter.export()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in (("trial-results.json", result), ("measurement-evidence.json", manifest),
                        ("team-trial-results.json", team_result), ("team-measurement-evidence.json", team_manifest)):
        with (destination / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    combined = {**result, "cells": result["cells"] + team_result["cells"]}
    (destination / "tables.tex").write_text(latex_tables(combined, exporter.metrics), encoding="utf-8")
    (destination / "appendix-tables.tex").write_text(latex_appendix_tables(combined, exporter.metrics), encoding="utf-8")
    (destination / "metric-definitions.tex").write_text(latex_definitions(exporter.metrics), encoding="utf-8")
    (destination / "evidence-availability.tex").write_text(latex_availability(combined, exporter.metrics), encoding="utf-8")
    (destination / "results-summary.tex").write_text(latex_findings(result, manifest, team_result, team_manifest), encoding="utf-8")
    (destination / "trial-tables.tex").write_text(
        "\\documentclass[conference,onecolumn]{IEEEtran}\n"
        "\\usepackage[T1]{fontenc}\n\\usepackage[utf8]{inputenc}\n\\usepackage{array,booktabs}\n"
        "\\begin{document}\n\\input{tables.tex}\n\\clearpage\n\\appendices\n"
        "\\section{Additional measurements}\n\\input{appendix-tables.tex}\n\\end{document}\n", encoding="utf-8")
    if report_output is not None:
        report_output = Path(report_output)
        report_output.mkdir(parents=True, exist_ok=True)
        for name in ("tables.tex", "appendix-tables.tex", "metric-definitions.tex", "evidence-availability.tex", "results-summary.tex"):
            target = report_output / name
            if target.is_symlink():
                raise EvidenceError("The generated report file must be a regular local file: " + str(target))
            shutil.copyfile(destination / name, target)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Create a new export directory.")
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--evidence-prefix", default="", help="Prefix evidence paths with their publication location.")
    parser.add_argument("--report-output", type=Path, help="Refresh generated LaTeX fragments in this report subdirectory.")
    parser.add_argument("--memory-samples", type=Path, default=Path("gpu-memory.jsonl"),
                        help="Select an immutable memory-sample snapshot inside the trial directory. Relative paths start at the trial directory.")
    parser.add_argument("--additional-memory-samples", type=Path, action="append", default=[],
                        help="Include another immutable snapshot, preserving its own source hash and job/GPU identity. Repeat for additional files.")
    parser.add_argument("--include-deferred", action="store_true",
                        help="Include DSpark and its vLLM V2 reference after their evaluation resumes.")
    args = parser.parse_args()
    try:
        result = write_export(Exporter(args.trial, args.catalogue, args.evidence_prefix, memory_samples=args.memory_samples,
                                       additional_memory_samples=args.additional_memory_samples,
                                       include_deferred=args.include_deferred), args.output, args.report_output)
    except (OSError, ValueError) as error:
        parser.exit(1, "The trial export could not be completed. " + str(error) + "\n")
    measured = sum(cell["value"] is not None for cell in result["cells"])
    print(f"The primary export contains {measured} measured cells. Team measurements use a separate result file.")
    print("Empty cells retain their reasons in the evidence records.")
    print("The Site files and LaTeX tables are ready for review in " + str(args.output) + ".")


if __name__ == "__main__":
    main()
