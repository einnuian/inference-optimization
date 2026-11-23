#!/usr/bin/env python3
"""Validate the completed trial and upload its saved measurements to Weights & Biases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from datetime import datetime, timezone

ENTITY = "ayoadetayo-columbia-university"
PROJECT = "hpml-first30-trial-2026-11-22"
TARGET_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
INPUT_SHA256 = "b5ae4556936743902d455576aa0d71deb9005ac5f28b8b9a1fd4daa49b20fd1c"
CONFIGURATIONS = {
    "ar-sglang": ("sglang", None),
    "eagle3": ("sglang", "ar-sglang"),
    "dflash": ("sglang", "ar-sglang"),
    "ar-vllm-v1": ("vllm", None),
    "peagle": ("vllm", "ar-vllm-v1"),
    "ar-vllm-v2": ("vllm", None),
    "dspark": ("vllm", "ar-vllm-v2"),
    "team-ar": ("transformers-team", None),
    "team-ffn-100": ("transformers-team", "team-ar"),
    "team-ffn-75": ("transformers-team", "team-ar"),
    "team-ffn-50": ("transformers-team", "team-ar"),
}
TEAM_CONFIGURATIONS = frozenset(name for name in CONFIGURATIONS if name.startswith("team-"))
METHOD_LABELS = {
    "ar-sglang": "AR · SGLang", "eagle3": "EAGLE-3", "dflash": "DFlash",
    "ar-vllm-v1": "AR · vLLM ModelRunnerV1", "peagle": "P-EAGLE",
    "ar-vllm-v2": "AR · vLLM ModelRunnerV2", "dspark": "DSpark",
    "team-ar": "AR · Transformers (pilot)",
    "team-ffn-100": "Pilot FFN 100%",
    "team-ffn-75": "Pilot FFN 75%",
    "team-ffn-50": "Pilot FFN 50%",
}
METHOD_LABEL_ALIASES = {
    **{"Team FFN %s%% (%s)" % (width, qualifier): "team-ffn-" + width
       for width in ("100", "75", "50") for qualifier in ("pilot", "pilot reproduction")},
    "AR · Transformers (team)": "team-ar",
}
BENCHMARKS = (
    "gsm8k", "math500", "aime25", "humaneval", "mbpp", "livecodebench",
    "mt-bench", "alpaca", "arena-hard-v2", "cnn-dm",
)
BENCHMARK_LABELS = {
    "GSM8K", "MATH-500", "AIME 2025", "HumanEval", "MBPP", "LiveCodeBench",
    "MT-Bench", "Alpaca", "Arena-Hard", "CNN/DailyMail",
}
SAFE_CONFIG_FIELDS = (
    "configuration", "reference", "backend", "model", "slurm_job_id", "gpu_uuid",
    "gpu_record", "target_revision", "input_sha256", "prepared_sha256",
    "recorder_sha256", "server_record_sha256", "clock", "sglang_incremental",
    "repetitions", "temperature", "max_new_tokens", "adapter_sha256", "phase",
)
SOURCE_FIELDS = (
    "classification", "notebook_commit", "notebook_sha256", "checkpoint_sha256",
    "training_data_sha256", "training_examples", "training_updates", "training_target_revision",
    "training_provenance", "draft_vocabulary_size", "draft_vocabulary_sha256",
    "feed_forward_width", "draft_proposals_per_round", "torch_version", "transformers_version",
    "cuda_version", "python_version", "target_attention_implementation",
    "artifact_origin", "training_notebook_commit", "training_dataset_revision",
    "training_torch_version", "training_transformers_version", "training_seed",
)
OPTIONAL_EVIDENCE = (
    "resources.json", "comparison.json", "diagnostics.json", "rounds.jsonl",
    "gpu-samples.jsonl",
)
CATALOGUE_SUPPLEMENTS = (
    "report/benchmark_report.pdf", "report/benchmark_report.tex", "publication-sources.json",
    "provenance/continuation/attempt-outcome.json",
)


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def checked_file(directory, name):
    path = directory / name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != directory.resolve():
        raise ValueError("The trial evidence must contain a regular local file: " + str(path))
    return path


def number(value, label, nullable=False):
    if nullable and value is None:
        return value
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(label + " must be a finite, nonnegative measurement.")
    return value


def same_number(actual, expected, label):
    number(actual, label)
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(label + " differs from the saved request observations.")


def evidence_digest(paths, root=None):
    digest = hashlib.sha256()
    label = (lambda path: path.relative_to(root).as_posix()) if root else (lambda path: path.name)
    for path in sorted(paths, key=label):
        digest.update(label(path).encode("utf-8") + b"\0")
        digest.update(str(path.stat().st_size).encode("ascii") + b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def source_provenance(run):
    settings = run.get("server_settings") or {}
    result = {key: settings[key] for key in SOURCE_FIELDS if key in settings}
    if any(value is not None and type(value) not in (str, int, float, bool) for value in result.values()):
        raise ValueError("Source provenance fields must contain scalar values.")
    for field in ("training_manifest", "checkpoint_manifest", "import_manifest"):
        if settings.get(field) is not None:
            encoded = json.dumps(settings[field], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            result[field + "_sha256"] = hashlib.sha256(encoded).hexdigest()
    return result


def validate_team_materialization(row):
    metadata = row.get("server_metadata") or {}
    if (not isinstance(metadata.get("team_rounds"), list) or
            metadata.get("team_rounds_materialization") !=
            "Expanded after measurement from the raw compact stream counts."):
        raise ValueError("The team result requires its completed post-measurement count expansion.")


def native_evidence(directory, configuration, run, blocks):
    names = {"native-after-warmup.json"} | {
        "native-%02d-%s.json" % (repeat, benchmark) for repeat in range(1, 4) for benchmark in BENCHMARKS}
    observed = {path.name for path in directory.glob("native-*.json")}
    if configuration not in {"peagle", "dspark"}:
        if observed:
            raise ValueError("Native counter files are unexpected for " + configuration)
        return []
    if observed != names:
        raise ValueError(configuration + " requires its warmup counter snapshot and all 30 native benchmark records.")
    paths = [checked_file(directory, name) for name in sorted(names)]
    from trial_native_stats import delta
    previous = load_json(directory / "native-after-warmup.json")
    if previous.get("gpu_uuid") != run["gpu_uuid"]:
        raise ValueError("The native counters identify a different physical GPU.")
    for block in sorted(blocks, key=lambda item: item["started_ns"]):
        saved = load_json(directory / ("native-%02d-%s.json" % (block["repetition"], block["benchmark"])))
        if saved.get("benchmark") != block["benchmark"] or saved.get("repetition") != block["repetition"]:
            raise ValueError("The native counter record differs from its benchmark block.")
        if saved.get("before") != previous:
            raise ValueError("The native counter records must preserve consecutive snapshot boundaries.")
        after = saved["after"]
        if (type(previous.get("observed_completions")) is not int or
                after.get("observed_completions") != previous["observed_completions"] + 3):
            raise ValueError("Each native counter interval must contain three completed requests.")
        if saved.get("metrics") != delta(previous, after, gpu_uuid=run["gpu_uuid"]):
            raise ValueError("The saved native metrics differ from their counter observations.")
        previous = after
    return paths


def validate_configuration(root, configuration):
    directory = root / configuration
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("The completed configuration directory is missing: " + configuration)
    required = ["run.json", "summary.json", "warmup.json", "live.jsonl"]
    block_names = {"%02d-%s.json" % (repetition, benchmark)
                   for repetition in range(1, 4) for benchmark in BENCHMARKS}
    actual_blocks = {path.name for path in directory.glob("[0-9][0-9]-*.json")}
    if actual_blocks != block_names:
        raise ValueError(configuration + " must contain all 30 completed benchmark blocks.")
    paths = [checked_file(directory, name) for name in required + sorted(block_names)]
    for name in OPTIONAL_EVIDENCE:
        if (directory / name).exists():
            paths.append(checked_file(directory, name))
    run = load_json(directory / "run.json")
    backend, reference = CONFIGURATIONS[configuration]
    expected = {"configuration": configuration, "backend": backend, "reference": reference,
                "repetitions": 3, "temperature": 0, "max_new_tokens": 2048,
                "target_revision": TARGET_REVISION, "input_sha256": INPUT_SHA256}
    for key, value in expected.items():
        if run.get(key) != value:
            raise ValueError(configuration + " has an unexpected recorded setting: " + key)
    hash_fields = ["input_sha256", "prepared_sha256", "recorder_sha256", "server_record_sha256"]
    if configuration in TEAM_CONFIGURATIONS:
        hash_fields.append("adapter_sha256")
        settings = run.get("server_settings") or {}
        origin = settings.get("artifact_origin")
        if origin is not None and origin not in {"teammate-original", "local-reproduction", "paired-reference"}:
            raise ValueError("The team artifact origin requires its recorded classification.")
        if (run.get("phase") != "primary" or settings.get("phase") != "primary" or
                settings.get("adapter_sha256") != run.get("adapter_sha256") or
                settings.get("target_revision") != TARGET_REVISION):
            raise ValueError("The team run requires matching primary-run adapter and target provenance.")
        for field, length in (("notebook_sha256", 64), ("notebook_commit", 40)):
            if not re.fullmatch("[0-9a-f]{%d}" % length, settings.get(field, "")):
                raise ValueError("The team run requires its pinned notebook provenance.")
        if configuration != "team-ar":
            for field in ("checkpoint_sha256", "training_data_sha256", "draft_vocabulary_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", settings.get(field, "")):
                    raise ValueError("The team drafter requires source provenance for " + field)
    for field in hash_fields:
        value = run.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(configuration + " requires a SHA-256 checksum for " + field)
    warmup = load_json(directory / "warmup.json")
    if not isinstance(warmup, list) or len(warmup) != 3:
        raise ValueError(configuration + " must contain three completed warmup records.")
    if len({row.get("prompt_id") for row in warmup}) != 3 or any(
            row.get("finish_reason") not in {"stop", "length"} for row in warmup):
        raise ValueError(configuration + " requires three distinct, completed warmup requests.")
    if configuration in TEAM_CONFIGURATIONS:
        if list(directory.glob("*.expanding")):
            raise ValueError("The team result is still saving its post-measurement count expansion.")
        for row in warmup:
            validate_team_materialization(row)
    summary = load_json(directory / "summary.json")
    if set(summary) != set(BENCHMARKS):
        raise ValueError(configuration + " has an incomplete benchmark summary.")
    request_rows, benchmark_rows, prompt_sets, input_ids, request_ids, blocks = [], [], {}, {}, set(), []
    for benchmark in BENCHMARKS:
        rates = []
        for repetition in range(1, 4):
            block = load_json(directory / ("%02d-%s.json" % (repetition, benchmark)))
            blocks.append(block)
            if block.get("benchmark") != benchmark or block.get("repetition") != repetition:
                raise ValueError("The benchmark block identity differs from its filename.")
            requests = block.get("requests")
            if not isinstance(requests, list) or len(requests) != 3:
                raise ValueError("Each benchmark block must contain three completed requests.")
            prompt_ids = tuple(row["prompt_id"] for row in requests)
            if len(set(prompt_ids)) != 3:
                raise ValueError("Each benchmark block must contain three distinct prompt IDs.")
            if benchmark in prompt_sets and prompt_sets[benchmark] != prompt_ids:
                raise ValueError("The prompt order changes between repetitions.")
            prompt_sets[benchmark] = prompt_ids
            for row in requests:
                if configuration in TEAM_CONFIGURATIONS:
                    validate_team_materialization(row)
                if row["benchmark"] != benchmark or row["request_id"] in request_ids:
                    raise ValueError("The request identity is duplicated or belongs to another benchmark.")
                request_ids.add(row["request_id"])
                actual_input = row.get("input_token_ids")
                if not isinstance(actual_input, list) or not actual_input or any(
                        type(token) is not int or token < 0 for token in actual_input):
                    raise ValueError("Each request must preserve its original input token IDs.")
                if row["prompt_id"] in input_ids and input_ids[row["prompt_id"]] != actual_input:
                    raise ValueError("The input token IDs change between repetitions.")
                input_ids[row["prompt_id"]] = actual_input
                count = row["output_tokens"]
                if type(count) is not int or count < 0 or count != len(row["output_token_ids"]):
                    raise ValueError("The saved output token count differs from the delivered token IDs.")
                if row.get("finish_reason") not in {"stop", "length"} or not isinstance(row.get("output_text"), str):
                    raise ValueError("Every uploaded request requires its completed response and stopping reason.")
                latency = number(row["end_to_end_seconds"], "Completion time")
                if latency <= 0:
                    raise ValueError("Completion time must be positive.")
                ttft = number(row["ttft_seconds"], "TTFT", nullable=True)
                tpot = number(row["tpot_seconds"], "TPOT", nullable=True)
                if ttft is not None and ttft > latency:
                    raise ValueError("TTFT exceeds the completion time.")
                request_rows.append([benchmark, row["prompt_id"], repetition, count, ttft,
                                     latency, tpot, row["output_text"], row["finish_reason"]])
            if block["started_ns"] != requests[0]["submitted_ns"] or block["ended_ns"] != requests[-1]["completed_ns"]:
                raise ValueError("The benchmark interval differs from its request timestamps.")
            elapsed = (block["ended_ns"] - block["started_ns"]) / 1e9
            if elapsed <= 0:
                raise ValueError("The benchmark interval must be positive.")
            token_count = sum(row["output_tokens"] for row in requests)
            same_number(block["metrics"]["elapsed_seconds"], elapsed, "Benchmark elapsed time")
            same_number(block["metrics"]["output_tokens"], token_count, "Benchmark output tokens")
            same_number(block["metrics"]["otps"], token_count / elapsed, "Benchmark OTPS")
            rates.append(block["metrics"]["otps"])
        saved = summary[benchmark]["across_repetitions"]["otps"]
        if saved.get("n") != 3:
            raise ValueError("The OTPS summary must contain three repetitions.")
        same_number(saved["mean"], statistics.mean(rates), "Mean OTPS")
        same_number(saved["sample_sd"], statistics.stdev(rates), "OTPS sample standard deviation")
        benchmark_rows.append([benchmark, saved["mean"], saved["sample_sd"], 9, 3, 3])
    paths.extend(native_evidence(directory, configuration, run, blocks))
    return {"kind": "configuration", "configuration": configuration, "run": run, "summary": summary,
            "source_provenance": source_provenance(run),
            "requests": request_rows, "benchmarks": benchmark_rows, "prompt_sets": prompt_sets,
            "input_ids": input_ids, "paths": paths, "evidence_sha256": evidence_digest(paths)}


def trial_identity(root):
    """Keep allocation-specific pilot roots distinct when each ends in results."""
    return root.parent.name + "/results" if root.name == "results" else root.name


def assign_run_identity(root, record, evidence_version):
    identity = "\0".join((trial_identity(root), record["configuration"], INPUT_SHA256))
    if record["configuration"] in TEAM_CONFIGURATIONS:
        metadata = record.get("run", {})
        identity += "\0" + str(metadata.get("gpu_uuid", "")) + "\0" + str(metadata.get("slurm_job_id", ""))
    if evidence_version != "v1":
        identity += "\0" + evidence_version
    record["run_id"] = hashlib.sha256(identity.encode()).hexdigest()[:20]
    record["evidence_version"] = evidence_version


def validate_trial(root, evidence_version="v1", configurations=None, *, allow_unpaired_team=False):
    root = root.resolve(strict=True)
    if allow_unpaired_team and configurations is None:
        raise ValueError("Select explicit completed team configurations when publishing without a complete reference.")
    configurations = list(CONFIGURATIONS) if configurations is None else list(configurations)
    if not configurations or len(set(configurations)) != len(configurations) or any(
            name not in CONFIGURATIONS for name in configurations):
        raise ValueError("Select distinct known trial configurations.")
    for name in configurations:
        reference = CONFIGURATIONS[name][1]
        if (reference is not None and reference not in configurations and
                not (allow_unpaired_team and name in TEAM_CONFIGURATIONS)):
            raise ValueError("Include the matching reference " + reference + " when selecting " + name + ".")
    records = [validate_configuration(root, name) for name in configurations]
    by_name = {record["configuration"]: record for record in records}
    for record in records:
        reference = CONFIGURATIONS[record["configuration"]][1]
        if reference in by_name:
            for field in ("gpu_uuid", "gpu_record"):
                if record["run"].get(field) != by_name[reference]["run"].get(field):
                    reason = "The selected method and its reference differ in " + field + "."
                    if allow_unpaired_team and record["configuration"] in TEAM_CONFIGURATIONS:
                        record["unpaired_reference_reason"] = reason
                    else:
                        raise ValueError(reason)
    first = records[0]
    for record in records[1:]:
        for field in ("target_revision", "input_sha256", "prepared_sha256"):
            if record["run"].get(field) != first["run"].get(field):
                raise ValueError("The configurations differ in " + field)
        if record["prompt_sets"] != first["prompt_sets"]:
            raise ValueError("The configurations use different prompt IDs or ordering.")
        if record["input_ids"] != first["input_ids"]:
            raise ValueError("The configurations use different input token IDs.")
    for names, fields in (
            (set(CONFIGURATIONS) - TEAM_CONFIGURATIONS, ("recorder_sha256",)),
            (TEAM_CONFIGURATIONS, ("recorder_sha256", "adapter_sha256"))):
        group = [record for record in records if record["configuration"] in names]
        if not group:
            continue
        for field in fields:
            if len({record["run"][field] for record in group}) != 1:
                raise ValueError("The configurations within a recorder group differ in " + field)
    for record in records:
        assign_run_identity(root, record, evidence_version)
    return root, records


def approved_catalogue_source(relative):
    parts = relative.parts
    if not parts or relative.is_absolute() or any(part in {"..", "."} for part in parts):
        return False
    if relative.as_posix() in CATALOGUE_SUPPLEMENTS:
        return True
    if len(parts) == 1:
        return (relative.name in {"gpu-memory.jsonl", "gpu-samples.jsonl", "allocation.json",
                                  "allocation-report.json", "compute-usage.json", "resource-summary.json"} or
                bool(re.fullmatch(r"gpu-memory-snapshot-[0-9]{8}T[0-9]{4}(?:[0-9]{2})?Z?\.jsonl", relative.name)))
    if len(parts) == 2 and parts[0] == "allocations":
        return bool(re.fullmatch(r"[0-9]+\.json", parts[1]))
    filename = relative.name
    if (len(parts) == 5 and parts[:2] == ("diagnostics", "attempts") and parts[3] == "logit-probe" and
            filename in {"humaneval-source-index-2.json", "status.json"}):
        return any(re.fullmatch(re.escape(configuration) + r"-[0-9]{8}T[0-9]{12}Z", parts[2])
                   for configuration in ("ar-sglang", "eagle3", "dflash"))
    block = re.fullmatch(r"(?:native-)?0[123]-(.+)\.json", filename)
    ordinary = filename in {"run.json", "summary.json", "warmup.json", "live.jsonl",
                            "native-after-warmup.json", *OPTIONAL_EVIDENCE} or bool(
        block and block.group(1) in BENCHMARKS)
    if ordinary:
        return ((len(parts) == 2 and parts[0] in CONFIGURATIONS) or
                (len(parts) == 3 and parts[0] == "diagnostics" and parts[1] in CONFIGURATIONS))
    trace_name = bool(
        re.fullmatch(r"(?:sglang|vllm|team)-rounds-[0-9]+\.jsonl", filename) or
        re.fullmatch(r"phase-[0-9]+-[A-Za-z0-9_-]+\.jsonl", filename)
    )
    direct_trace = ((len(parts) == 2 and parts[0] in CONFIGURATIONS) or
                    (len(parts) == 3 and parts[0] == "diagnostics" and parts[1] in CONFIGURATIONS))
    attempt_trace = (len(parts) == 5 and parts[:2] == ("diagnostics", "attempts") and parts[3] == "trace" and
                     any(re.fullmatch(re.escape(configuration) + r"-[0-9]{8}T[0-9]{12}Z", parts[2])
                         for configuration in CONFIGURATIONS))
    return trace_name and (direct_trace or attempt_trace)


def catalogue_source(root, relative, evidence):
    relative = Path(relative)
    if not approved_catalogue_source(relative):
        raise ValueError("The catalogue references an unapproved evidence filename: " + str(relative))
    path = root / relative
    if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
        raise ValueError("Catalogue evidence must use regular paths inside the trial.")
    resolved = path.resolve(strict=True)
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError("Catalogue evidence must stay inside the trial directory.")
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(data)
            digest.update(data)
    if evidence.get("sha256") != digest.hexdigest() or evidence.get("bytes") != size:
        raise ValueError("The catalogue source differs from its evidence manifest: " + str(relative))
    return resolved


def validate_catalogue_pair(root, configuration, sources):
    reference = CONFIGURATIONS[configuration][1]
    if reference is None:
        raise ValueError("A paired catalogue metric requires a speculative method and its reference.")
    names = [configuration + "/run.json", reference + "/run.json"]
    if any(name not in sources for name in names):
        raise ValueError("A paired catalogue metric requires both hashed run records.")
    method, baseline = [load_json(root / name) for name in names]
    for field in ("prepared_sha256", "target_revision", "input_sha256", "repetitions",
                  "temperature", "max_new_tokens", "model", "gpu_uuid", "gpu_record"):
        if method.get(field) is None or method.get(field) != baseline.get(field):
            raise ValueError("A paired catalogue metric uses incompatible " + field + " evidence.")


def validate_catalogue(root, directory=None, evidence_version="v1", records=None):
    explicit = directory is not None
    directory = Path(directory) if explicit else root / "export"
    if not directory.exists():
        if explicit:
            raise ValueError("The selected catalogue export directory is missing.")
        return None
    if directory.is_symlink():
        raise ValueError("The catalogue export directory must be a regular trial directory.")
    directory = directory.resolve(strict=True)
    if root not in directory.parents:
        raise ValueError("The catalogue export must stay inside the trial directory.")
    all_cells, paths, known_keys = [], set(), set()
    lookup = {**METHOD_LABEL_ALIASES, **{label: configuration for configuration, label in METHOD_LABELS.items()}}
    for result_name, manifest_name in (
            ("trial-results.json", "measurement-evidence.json"),
            ("team-trial-results.json", "team-measurement-evidence.json")):
        result_path = checked_file(directory, result_name)
        manifest_path = checked_file(directory, manifest_name)
        paths.update((result_path, manifest_path))
        result, manifest = load_json(result_path), load_json(manifest_path)
        if result.get("trialId") != root.name or manifest.get("trial_id") != root.name or manifest.get("schema_version") != 1:
            raise ValueError("The catalogue export identifies another trial or manifest format.")
        sources = manifest.get("source_files")
        if not isinstance(sources, dict):
            raise ValueError("The catalogue evidence manifest requires its source-file checksums.")
        if records is not None:
            team_file = result_name.startswith("team-")
            configurations = [record["configuration"] for record in records
                              if (record["configuration"] in TEAM_CONFIGURATIONS) == team_file]
            required_sources = {configuration + "/run.json" for configuration in configurations}
            required_sources.update("%s/%02d-%s.json" % (configuration, repetition, benchmark)
                                    for configuration in configurations
                                    for repetition in range(1, 4) for benchmark in BENCHMARKS)
            if not required_sources <= sources.keys():
                raise ValueError("The catalogue export predates complete primary results. Regenerate the export before uploading.")
        for relative, evidence in sources.items():
            paths.add(catalogue_source(root, relative, evidence))
        cells = result.get("cells")
        details = manifest.get("cells")
        if not isinstance(cells, list) or not isinstance(details, list) or len(cells) != len(details):
            raise ValueError("Every catalogue cell requires its repetition evidence.")
        fields = ("metricId", "variant", "method", "benchmark", "position")
        by_key = {tuple(detail["key"][field] for field in fields): detail for detail in details}
        if len(by_key) != len(details):
            raise ValueError("The catalogue manifest contains duplicate cell keys.")
        for cell in cells:
            key = tuple(cell[field] for field in fields)
            if key not in by_key:
                raise ValueError("The catalogue contains a duplicated cell or a cell without evidence.")
            if (cell["method"] not in lookup or not re.fullmatch(r"metric-[a-z0-9-]+", cell["metricId"]) or
                    not isinstance(cell["variant"], str) or cell["benchmark"] not in BENCHMARK_LABELS or
                    not isinstance(cell["unit"], str) or not cell["unit"]):
                raise ValueError("The catalogue requires known methods and explicit metric labels and units.")
            configuration = lookup[cell["method"]]
            if cell["metricId"] in {"metric-speedup", "metric-quality"} and cell["value"] is not None:
                validate_catalogue_pair(root, configuration, sources)
            canonical_key = tuple(configuration if field == "method" else cell[field] for field in fields)
            if canonical_key in known_keys:
                raise ValueError("The catalogue contains a duplicated cell under equivalent method labels.")
            known_keys.add(canonical_key)
            if (result_name.startswith("team-")) != (configuration in TEAM_CONFIGURATIONS):
                raise ValueError("The team and baseline catalogue files must identify their corresponding methods.")
            if cell["position"] is not None and (type(cell["position"]) is not int or cell["position"] < 1):
                raise ValueError("A catalogue position must be a positive integer or null.")
            detail = by_key[key]
            observations = detail["observations"]
            if type(cell["n"]) is not int or not 0 <= cell["n"] <= 3 or len(observations) != cell["n"]:
                raise ValueError("The catalogue repetition count differs from its observations.")
            repetitions = [observation["repetition"] for observation in observations]
            if len(set(repetitions)) != len(repetitions) or any(rep not in {1, 2, 3} for rep in repetitions):
                raise ValueError("Catalogue observations must identify distinct trial repetitions.")
            missing = detail["missing_repetitions"]
            missing_repetitions = [item["repetition"] for item in missing]
            if (len(set(missing_repetitions)) != len(missing_repetitions) or
                    set(missing_repetitions) != {1, 2, 3} - set(repetitions) or
                    any(not isinstance(item.get("reason"), str) or not item["reason"] for item in missing)):
                raise ValueError("Every unavailable catalogue repetition requires its recorded reason.")
            values = [number(observation["value"], "Catalogue observation") for observation in observations]
            for observation in observations:
                if any(source not in sources for source in observation["sources"]):
                    raise ValueError("A catalogue observation references an unhashed source file.")
            if values:
                same_number(cell["value"], statistics.mean(values), "Catalogue cell value")
                if len(values) > 1:
                    same_number(cell["sd"], statistics.stdev(values), "Catalogue cell standard deviation")
                elif cell["sd"] is not None:
                    raise ValueError("A single observed repetition has no sample standard deviation.")
            elif cell["value"] is not None or cell["sd"] is not None or not cell.get("reason"):
                raise ValueError("An unmeasured catalogue cell requires null values and an explicit reason.")
            all_cells.append({**cell, "configuration": configuration})
    for name in ("tables.tex", "trial-tables.tex", "appendix-tables.tex", "metric-definitions.tex",
                 "evidence-availability.tex", "results-summary.tex"):
        if (directory / name).exists():
            paths.add(checked_file(directory, name))
    for relative in CATALOGUE_SUPPLEMENTS:
        path = root / relative
        if path.exists() or path.is_symlink():
            if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
                raise ValueError("Catalogue report files must use regular paths inside the trial.")
            if root not in path.resolve(strict=True).parents:
                raise ValueError("Catalogue report files must stay inside the trial directory.")
            paths.add(checked_file(path.parent, path.name))
    paths = sorted(paths)
    record = {"kind": "catalogue", "configuration": "metric-catalogue",
              "cells": all_cells, "paths": paths, "digest_root": root,
              "evidence_sha256": evidence_digest(paths, root)}
    assign_run_identity(root, record, evidence_version)
    return record


def receipt_name(evidence_version):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", evidence_version):
        raise ValueError("Use a short evidence-version label containing letters, digits, dots, underscores, or hyphens.")
    return "wandb-upload.json" if evidence_version == "v1" else "wandb-upload-" + evidence_version + ".json"


def read_receipt(root, records, evidence_version="v1"):
    path = root / receipt_name(evidence_version)
    if not path.exists():
        return {"schema": 1, "entity": ENTITY, "project": PROJECT, "trial": trial_identity(root),
                "evidence_version": evidence_version, "runs": {}}
    checked_file(root, path.name)
    receipt = load_json(path)
    if any(receipt.get(key) != value for key, value in
           (("schema", 1), ("entity", ENTITY), ("project", PROJECT), ("trial", trial_identity(root)))):
        raise ValueError("The existing upload receipt belongs to a different destination or trial.")
    if not isinstance(receipt.get("runs"), dict):
        raise ValueError("The existing upload receipt has an invalid run list.")
    if receipt.get("evidence_version", "v1") != evidence_version:
        raise ValueError("The upload receipt identifies another evidence version.")
    receipt["evidence_version"] = evidence_version
    for record in records:
        previous = receipt["runs"].get(record["configuration"])
        if previous and (previous.get("run_id") != record["run_id"] or
                         previous.get("evidence_sha256") != record["evidence_sha256"]):
            raise ValueError("The saved evidence changed after an upload. Choose a new --evidence-version to preserve the previous runs and receipt.")
    return receipt


def save_receipt(root, receipt):
    receipt["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                         prefix=".wandb-upload-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(receipt, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, root / receipt_name(receipt.get("evidence_version", "v1")))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def upload_settings(wandb):
    if wandb.__version__ != "0.30.0":
        raise ValueError("Use the prepared tracking environment with wandb 0.30.0.")
    return wandb.Settings(disable_code=True, disable_git=True, save_code=False,
                          console="off", capture_loggers={}, x_disable_meta=True,
                          x_disable_stats=True, x_disable_machine_info=True, quiet=True)


def save_configuration_tables(wandb, run, record):
    run.summary["benchmark_results"] = wandb.Table(
        columns=["benchmark", "otps_mean", "otps_sample_sd", "measured_requests",
                 "distinct_prompts", "repetitions"], data=record["benchmarks"])
    run.summary["request_results"] = wandb.Table(
        columns=["benchmark", "prompt_id", "repetition", "output_tokens", "ttft_seconds",
                 "end_to_end_seconds", "tpot_seconds", "output_text", "finish_reason"],
        data=record["requests"])
    for benchmark, values in record["summary"].items():
        for metric, stats in values["across_repetitions"].items():
            for statistic in ("n", "mean", "sample_sd"):
                value = stats[statistic]
                if value is not None:
                    run.summary["%s/%s/%s" % (benchmark, metric, statistic)] = number(value, metric)


def save_catalogue_table(wandb, run, record):
    columns = ["configuration", "method", "metricId", "variant", "benchmark", "position",
               "value", "sd", "n", "unit", "reason", "evidence"]
    rows = [[cell.get(field) for field in columns] for cell in record["cells"]]
    run.summary["metric_catalogue"] = wandb.Table(columns=columns, data=rows)
    run.summary["catalogue_cell_count"] = len(rows)
    run.summary["measured_cell_count"] = sum(cell["value"] is not None for cell in record["cells"])
    run.summary["unavailable_cell_count"] = sum(cell["value"] is None for cell in record["cells"])
    run.summary["repetition_definition"] = "Each cell's n counts eligible repetitions of three prompts."


def upload(root, records, receipt):
    os.environ.update(WANDB_DISABLE_CODE="true", WANDB_DISABLE_GIT="true",
                      WANDB_CONSOLE="off", WANDB_ERROR_REPORTING="false")
    import wandb

    settings = upload_settings(wandb)
    for record in records:
        if evidence_digest(record["paths"], record.get("digest_root")) != record["evidence_sha256"]:
            raise ValueError("The trial files changed after validation. Finish saving the evidence before uploading.")
    for record in records:
        name = record["configuration"]
        catalogue_run = record["kind"] == "catalogue"
        if evidence_digest(record["paths"], record.get("digest_root")) != record["evidence_sha256"]:
            raise ValueError("The trial files changed after validation. Finish saving the evidence before uploading.")
        config = ({"configuration": name, "input_sha256": INPUT_SHA256,
                   "catalogue_cell_count": len(record["cells"])} if catalogue_run else
                  {key: record["run"][key] for key in SAFE_CONFIG_FIELDS if key in record["run"]})
        config.update(trial=trial_identity(root), evidence_sha256=record["evidence_sha256"],
                      evidence_version=record["evidence_version"])
        if not catalogue_run:
            config.update(measured_requests=90, warmup_requests=3,
                          recorder_group="team" if name in TEAM_CONFIGURATIONS else "baseline")
            if record["source_provenance"]:
                config["source_provenance"] = record["source_provenance"]
        print("Uploading saved measurements for %s to %s/%s." % (name, ENTITY, PROJECT), flush=True)
        run = wandb.init(entity=ENTITY, project=PROJECT, id=record["run_id"],
                         name=METHOD_LABELS.get(name, name), group=trial_identity(root),
                         job_type="catalogue" if catalogue_run else "benchmark", resume="allow",
                         mode="online", settings=settings)
        try:
            previous = dict(run.config)
            if previous and any(previous.get(key) != value for key, value in config.items()):
                raise ValueError("The existing Weights & Biases run identifies different evidence. Choose a new --evidence-version.")
            run.config.update(config, allow_val_change=False)
            if catalogue_run:
                save_catalogue_table(wandb, run, record)
            else:
                save_configuration_tables(wandb, run, record)
            artifact_metadata = {"configuration": name, "evidence_sha256": record["evidence_sha256"],
                                 "input_sha256": INPUT_SHA256, "evidence_version": record["evidence_version"]}
            if not catalogue_run:
                artifact_metadata.update(measured_requests=90, warmup_requests=3)
            artifact = wandb.Artifact("trial-%s-%s" % (record["run_id"], name), type="trial-evidence",
                                      metadata=artifact_metadata)
            for path in record["paths"]:
                artifact_name = path.relative_to(root).as_posix() if catalogue_run else path.name
                artifact.add_file(str(path), name=artifact_name)
            saved = run.log_artifact(artifact)
            saved.wait()
            entry = {"run_id": record["run_id"], "url": run.url,
                     "artifact": saved.qualified_name, "evidence_sha256": record["evidence_sha256"],
                     "evidence_version": record["evidence_version"], "kind": record["kind"],
                     "status": "uploaded"}
            if catalogue_run:
                entry["catalogue_cells"] = len(record["cells"])
            else:
                entry["measured_requests"] = 90
            run.finish(exit_code=0)
        except BaseException:
            run.finish(exit_code=1)
            raise
        receipt["runs"][name] = entry
        save_receipt(root, receipt)
        print("Saved %s." % entry["url"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="Directory containing saved configuration folders.")
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS,
                        help="Upload these complete configurations and their references. The default includes all eleven.")
    parser.add_argument("--allow-unpaired-team", action="store_true",
                        help="Publish explicitly selected complete team configurations when a compatible AR reference is unavailable. Paired catalogue metrics still require compatible evidence.")
    parser.add_argument("--catalogue-export", type=Path, help="Use this explicit catalogue directory. The default is RESULTS/export when it exists.")
    publication_scope = parser.add_mutually_exclusive_group()
    publication_scope.add_argument("--configuration-runs-only", action="store_true",
                                   help="Save configuration runs while keeping the existing metric catalogue unchanged.")
    publication_scope.add_argument("--catalogue-only", action="store_true",
                                   help="Validate the selected configurations and save only the metric catalogue snapshot.")
    parser.add_argument("--evidence-version", default="v1", help="Use a new label when evidence changes after an upload.")
    parser.add_argument("--upload", action="store_true", help="Upload the validated saved measurements to the existing project.")
    args = parser.parse_args()
    receipt_name(args.evidence_version)
    root, records = validate_trial(args.results, args.evidence_version, args.configurations,
                                  allow_unpaired_team=args.allow_unpaired_team)
    configuration_count = len(records)
    catalogue = (None if args.configuration_runs_only else
                 validate_catalogue(root, args.catalogue_export, args.evidence_version, records=records))
    if args.catalogue_only:
        if catalogue is None:
            parser.error("A catalogue-only publication requires an existing catalogue export.")
        records = [catalogue]
    elif catalogue is not None:
        records.append(catalogue)
    receipt = read_receipt(root, records, args.evidence_version)
    print("Validated %d configurations, %d measured requests, and %d warmup records." %
          (configuration_count, configuration_count * 90, configuration_count * 3), flush=True)
    if catalogue is not None:
        print("Validated %d catalogue cells and their source observations." % len(catalogue["cells"]), flush=True)
    for record in records:
        print("%s: %d evidence files; SHA-256 %s." %
              (record["configuration"], len(record["paths"]), record["evidence_sha256"]), flush=True)
    if args.upload:
        upload(root, records, receipt)
        print("The validated runs and evidence artifacts are saved in %s/%s." % (ENTITY, PROJECT), flush=True)
    else:
        print("Validation is complete. Add --upload to save these results in the existing Weights & Biases project.", flush=True)


if __name__ == "__main__":
    main()
