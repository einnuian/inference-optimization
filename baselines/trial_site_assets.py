#!/usr/bin/env python3
"""Package completed trial recordings as validated, static Site downloads."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

from trial_wandb import (
    BENCHMARKS, CONFIGURATIONS, METHOD_LABELS, INPUT_SHA256, TARGET_REVISION,
    checked_file, load_json, validate_catalogue, validate_trial,
)

# Wrangler's static-asset uploader enforces this limit for each individual file.
MAX_ASSET_BYTES = 25 * 1024 * 1024
# Compression also bounds the total deployed directory, including many request blocks.
COMPRESS_JSON_BYTES = 64 * 1024
BENCHMARK_LABELS = dict(zip(BENCHMARKS, (
    "GSM8K", "MATH-500", "AIME 2025", "HumanEval", "MBPP", "LiveCodeBench",
    "MT-Bench", "Alpaca", "Arena-Hard", "CNN/DailyMail",
)))
RESOURCE_FILES = (
    "gpu-memory.jsonl", "gpu-samples.jsonl", "allocation.json", "allocation-report.json",
    "compute-usage.json", "resource-summary.json",
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def asset_info(path, relative):
    size = path.stat().st_size
    if size > MAX_ASSET_BYTES:
        raise ValueError("The static asset exceeds the 25 MiB hosting limit: " + relative)
    return {"path": relative, "bytes": size, "sha256": digest(path)}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_prompts(path, records):
    if path.is_symlink() or not path.is_file():
        raise ValueError("The prepared prompts must be a regular local file.")
    bundle = load_json(path)
    checksum = digest(path)
    if bundle.get("input_sha256") != INPUT_SHA256 or bundle.get("target_revision") != TARGET_REVISION:
        raise ValueError("The prepared prompts identify a different input set or target revision.")
    prompts = bundle.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 30:
        raise ValueError("The prepared bundle must contain the 30 trial prompts.")
    lookup = {prompt["id"]: prompt for prompt in prompts}
    if len(lookup) != 30:
        raise ValueError("The prepared prompt IDs must be distinct.")
    groups = {}
    for benchmark in BENCHMARKS:
        rows = [prompt for prompt in prompts if prompt["benchmark"] == benchmark]
        indices = [prompt["sourceIndex"] for prompt in rows]
        if (len(rows) != 3 or any(type(index) is not int or index < 0 for index in indices) or
                indices != sorted(set(indices))):
            raise ValueError("Each benchmark must preserve three distinct prepared prompts in source order.")
        for row in rows:
            messages = row.get("messages")
            if (not isinstance(messages, list) or len(messages) != 1 or
                    messages[0].get("role") != "user" or not isinstance(messages[0].get("content"), str)):
                raise ValueError("The trial viewer requires one exact user message per prompt.")
        groups[benchmark] = rows
    for record in records:
        if record["run"]["prepared_sha256"] != checksum:
            raise ValueError("The prompt file differs from the file recorded by " + record["configuration"])
        for benchmark, ids in record["prompt_sets"].items():
            if list(ids) != [prompt["id"] for prompt in groups[benchmark]]:
                raise ValueError("The recorded prompt order differs from the prepared inputs.")
        if any(record["input_ids"].get(identifier) != prompt["input_token_ids"]
               for identifier, prompt in lookup.items()):
            raise ValueError("The recorded input tokens differ from the prepared inputs.")
    return bundle, groups


def add_source(paths, source, root):
    if source.is_symlink() or not source.is_file():
        raise ValueError("Recording assets must use regular local files: " + str(source))
    relative = source.relative_to(root)
    if "diagnostic-runtime" in relative.parts:
        raise ValueError("Installed diagnostic packages do not belong in the recording assets.")
    if root not in source.resolve().parents or any(part.is_symlink() for part in source.parents if part != root.parent):
        raise ValueError("Recording assets must remain inside the trial directory.")
    paths[relative.as_posix()] = source


def validate_streams(record):
    for source in record["paths"]:
        if not re.fullmatch(r"0[123]-(.+)\.json", source.name):
            continue
        for request in load_json(source)["requests"]:
            events = request.get("events")
            if not isinstance(events, list) or not events:
                raise ValueError("Each measured recording must preserve its stream events.")
            previous = request["submitted_ns"]
            for event in events:
                timestamp = event.get("received_ns")
                if type(timestamp) is not int or not previous <= timestamp <= request["completed_ns"]:
                    raise ValueError("Stream arrival times must remain ordered within the request interval.")
                if "text" in event and not isinstance(event["text"], str):
                    raise ValueError("A recorded stream chunk must preserve its text as a string.")
                previous = timestamp


def package_trial(results, prompts, site, catalogue_export=None):
    root = results.resolve(strict=True)
    site = site.resolve(strict=True)
    if not re.fullmatch(r"trial-first30-[0-9]{8}", root.name):
        raise ValueError("Use a trial-first30-YYYYMMDD result directory.")
    if not (site / "app").is_dir() or not (site / "lib").is_dir() or not (site / "package.json").is_file():
        raise ValueError("The Site directory must contain its app, lib and package.json files.")
    completed = [name for name in CONFIGURATIONS if (root / name / "summary.json").is_file()]
    if not completed:
        raise ValueError("The trial has no completed configurations to publish.")
    root, records = validate_trial(root, configurations=completed, allow_unpaired_team=True)
    bundle, groups = validate_prompts(prompts, records)
    paths = {}
    for record in records:
        validate_streams(record)
        for source in record["paths"]:
            add_source(paths, source, root)
    for name in RESOURCE_FILES:
        if (root / name).exists():
            add_source(paths, checked_file(root, name), root)
    catalogue = validate_catalogue(root, directory=catalogue_export, records=records)
    if catalogue:
        for source in catalogue["paths"]:
            add_source(paths, source, root)
    parent = site / "public" / "measurements"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / root.name
    if destination.exists():
        marker = destination / "index.json"
        if destination.is_symlink() or not marker.is_file():
            raise ValueError("The destination contains files outside this packager's ownership.")
        previous = load_json(marker)
        if previous.get("schema") != 1 or previous.get("trialId") != root.name:
            raise ValueError("The existing assets belong to another format or trial.")
    staging = Path(tempfile.mkdtemp(prefix=".trial-assets-", dir=parent))
    try:
        assets = []
        for relative, source in sorted(paths.items()):
            compress = (source.suffix in {".json", ".jsonl"} and
                        source.stat().st_size > min(COMPRESS_JSON_BYTES, MAX_ASSET_BYTES))
            asset_relative = relative + ".gz" if compress else relative
            target = staging / asset_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if compress:
                with source.open("rb") as input_file, target.open("wb") as output_file:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=output_file, mtime=0) as compressed:
                        shutil.copyfileobj(input_file, compressed)
                entry = asset_info(target, asset_relative)
                entry.update(compression="gzip", sourcePath=relative,
                             uncompressedBytes=source.stat().st_size, uncompressedSha256=digest(source))
            else:
                asset_info(source, relative)
                shutil.copyfile(source, target)
                entry = asset_info(target, relative)
            assets.append(entry)
        save_json(staging / "prompts.json", bundle)
        assets.append(asset_info(staging / "prompts.json", "prompts.json"))
        by_path = {asset["path"]: asset for asset in assets}
        index = {
            "schema": 1, "trialId": root.name, "baseUrl": "/measurements/" + root.name,
            "inputSha256": INPUT_SHA256, "targetRevision": TARGET_REVISION,
            "promptFile": by_path["prompts.json"],
            "requestCount": len(records) * 90, "repetitions": [1, 2, 3],
            "benchmarks": [{"id": benchmark, "label": BENCHMARK_LABELS[benchmark],
                            "prompts": [{"id": row["id"], "ordinal": ordinal, "sourceIndex": row["sourceIndex"],
                                         "sourceItemId": row["sourceItemId"], "sha256": row["promptSha256"]}
                                        for ordinal, row in enumerate(groups[benchmark], 1)]} for benchmark in BENCHMARKS],
            "configurations": [{"id": record["configuration"], "label": METHOD_LABELS[record["configuration"]],
                                "reference": record["run"]["reference"], "requestCount": 90,
                                "evidenceSha256": record["evidence_sha256"]} for record in records],
            "files": assets,
        }
        # This index is also the ownership marker for repeatable asset refreshes.
        save_json(staging / "index.json", index)
        archive = staging / "all-recordings.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
            archive_sources = {**paths, "prompts.json": staging / "prompts.json", "index.json": staging / "index.json"}
            for relative, source in sorted(archive_sources.items()):
                info = zipfile.ZipInfo(root.name + "/" + relative, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with source.open("rb") as input_file, zipped.open(info, "w") as output_file:
                    shutil.copyfileobj(input_file, output_file)
        index["archive"] = asset_info(archive, "all-recordings.zip")
        save_json(staging / "index.json", index)
        asset_info(staging / "index.json", "index.json")
        backup = None
        if destination.exists():
            backup = Path(tempfile.mkdtemp(prefix=".previous-trial-assets-", dir=parent))
            backup.rmdir()
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except BaseException:
            if backup is not None:
                os.replace(backup, destination)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=site / "lib",
                                         prefix=".trial-recordings-", suffix=".json", delete=False) as handle:
            temporary_index = Path(handle.name)
            json.dump(index, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        try:
            os.replace(temporary_index, site / "lib" / "trial-recordings.json")
        finally:
            temporary_index.unlink(missing_ok=True)
        return index
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--catalogue-export", type=Path, help="Use a specific immutable metric export inside the trial.")
    args = parser.parse_args()
    index = package_trial(args.results, args.prompts, args.site, args.catalogue_export)
    print("Packaged %d completed configurations and %d measured responses." %
          (len(index["configurations"]), index["requestCount"]))
    print("The recording archive contains %d bytes. Every asset meets the 25 MiB hosting limit." % index["archive"]["bytes"])


if __name__ == "__main__":
    main()
