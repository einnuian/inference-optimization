#!/usr/bin/env python3
"""Create an isolated engine package with the trial's diagnostic hooks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import difflib
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import tempfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def is_native_library(path):
    name = path.name
    if not (name.endswith(".so") or ".so." in name):
        return False
    with path.open("rb") as handle:
        return handle.read(4) == b"\x7fELF"


def package_copy_function(package, linked_libraries):
    """Copy source and data files while sharing native library contents through hard links."""
    def copy_file(source, destination):
        source = Path(source)
        destination = Path(destination)
        if not is_native_library(source):
            return shutil.copy2(source, destination)
        actual_source = source.resolve(strict=True)
        try:
            os.link(actual_source, destination)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            print("The native library uses another filesystem. Copying %s." % source, flush=True)
            return shutil.copy2(source, destination)
        original = actual_source.stat()
        linked = destination.stat()
        if (original.st_dev, original.st_ino) != (linked.st_dev, linked.st_ino):
            raise RuntimeError("The native library link does not identify its source inode.")
        linked_libraries.append({"relative_path": source.relative_to(package).as_posix(),
                                 "source_path": str(actual_source), "device": original.st_dev,
                                 "inode": original.st_ino, "size_bytes": original.st_size,
                                 "modified_ns": original.st_mtime_ns})
        return str(destination)

    return copy_file


def require_independent_file(source, copied):
    original = source.stat()
    actual = copied.stat()
    if (original.st_dev, original.st_ino) == (actual.st_dev, actual.st_ino):
        raise RuntimeError("The patched Python file must have an independent inode: " + str(copied))


def atomic_replace(path, content):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".hpml-diagnostic-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), path.stat().st_mode & 0o777)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_destination(engine, source, destination):
    root_value = os.environ.get("HPML_ROOT")
    if not root_value:
        raise ValueError("Set HPML_ROOT to the existing project directory.")
    root = Path(root_value).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("HPML_ROOT must identify an existing directory.")
    results = (root / "results").resolve()
    destination = destination.resolve()
    try:
        parts = destination.relative_to(results).parts
    except ValueError as error:
        raise ValueError("The diagnostic runtime must stay under HPML_ROOT/results.") from error
    if len(parts) != 3 or parts[1] != "diagnostic-runtime" or parts[2] != engine:
        raise ValueError("Use results/trial-first30-YYYYMMDD/diagnostic-runtime/" + engine + " as the destination.")
    match = re.fullmatch(r"trial-first30-([0-9]{8})", parts[0])
    if match is None:
        raise ValueError("The diagnostic trial directory must use the name trial-first30-YYYYMMDD.")
    datetime.strptime(match.group(1), "%Y%m%d")
    if source == destination or source in destination.parents:
        raise ValueError("The copied package must stay outside the source package.")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("The diagnostic destination already contains files. Select a fresh destination.")
    return destination


def require_class(text, name):
    pattern = r"^class\s+" + re.escape(name) + r"\b"
    if len(re.findall(pattern, text, flags=re.MULTILINE)) != 1:
        raise ValueError("The source must define exactly one " + name + " class.")


def append_hooks(text, imports, calls):
    if "_hpml_diagnostic_os" in text:
        raise ValueError("The source already contains the HPML diagnostic patch.")
    lines = ["", "", "# HPML diagnostic hooks apply to this isolated package copy.",
             "import os as _hpml_diagnostic_os", "if _hpml_diagnostic_os.environ.get(\"HPML_DIAGNOSTIC_DIR\"):"]
    lines.extend("    " + line for line in imports + calls)
    return text.rstrip("\n") + "\n".join(lines) + "\n"


def phase_marker(text, marker, calls, after=False):
    lines = text.splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.strip() == marker]
    if len(matches) != 1:
        raise ValueError("The DFlash phase marker must occur exactly once: " + marker)
    index = matches[0]
    indentation = lines[index][:len(lines[index]) - len(lines[index].lstrip())]
    if indentation != " " * 8:
        raise ValueError("The DFlash phase marker must have eight spaces of indentation.")
    insertion = [indentation + "if _hpml_diagnostic_os.environ.get(\"HPML_DIAGNOSTIC_DIR\"):\n"]
    insertion.extend(indentation + "    " + call + "\n" for call in calls)
    if after:
        index += 1
    lines[index:index] = insertion
    return "".join(lines)


def patch_source(engine, relative, text):
    if engine == "vllm" and relative == "v1/core/sched/output.py":
        require_class(text, "SchedulerOutput")
        marker = "    num_spec_tokens_to_schedule: int = 0"
        if text.count(marker) != 1 or "hpml_diagnostic_step_id" in text:
            raise ValueError("The pinned SchedulerOutput diagnostic field marker must occur once.")
        return text.replace(marker, marker + "\n\n    # Explicitly serialized identity for the isolated diagnostic package.\n    hpml_diagnostic_step_id: str | None = None", 1)
    if engine == "sglang" and relative == "srt/managers/tp_worker.py":
        require_class(text, "TpModelWorker")
        return append_hooks(text, ["from trial_phase_hooks import install_sglang_worker"],
                            ["install_sglang_worker(TpModelWorker)"])
    if engine == "sglang" and relative == "srt/managers/scheduler_components/batch_result_processor.py":
        require_class(text, "SchedulerBatchResultProcessor")
        return append_hooks(text, ["from trial_round_hooks import install_sglang_processor"],
                            ["install_sglang_processor(SchedulerBatchResultProcessor)"])
    if engine == "sglang" and relative == "srt/speculative/eagle_worker_v2.py":
        require_class(text, "EAGLEWorkerV2")
        require_class(text, "EagleDraftWorker")
        return append_hooks(text, ["from trial_phase_hooks import install_sglang_worker, install_sglang_eagle_draft, install_sglang_eagle_verify"],
                            ["install_sglang_worker(EAGLEWorkerV2)",
                             "install_sglang_eagle_draft(EagleDraftWorker)",
                             "install_sglang_eagle_verify(EAGLEWorkerV2)"])
    if engine == "sglang" and relative == "srt/speculative/dflash_worker_v2.py":
        require_class(text, "DFlashWorkerV2")
        text = append_hooks(text, ["from trial_phase_hooks import begin_phase, end_phase, install_sglang_worker"],
                            ["install_sglang_worker(DFlashWorkerV2)"])
        text = phase_marker(text, "# --- 1) Draft a fixed block with the draft model.",
                            ['begin_phase("draft")'], after=True)
        text = phase_marker(text, "# --- 2) Target verify.",
                            ['end_phase("draft")', 'begin_phase("verify")'])
        return phase_marker(text, "# --- 3) Materialize committed verify-input tokens into draft KV cache.",
                            ['end_phase("verify")'])
    if engine == "vllm" and relative == "v1/core/sched/scheduler.py":
        require_class(text, "Scheduler")
        return append_hooks(text, ["from trial_round_hooks import install_vllm_scheduler"],
                            ["install_vllm_scheduler(Scheduler)"])
    if engine == "vllm" and relative in {"v1/worker/gpu_model_runner.py", "v1/worker/gpu/model_runner.py"}:
        require_class(text, "GPUModelRunner")
        return append_hooks(text, ["from trial_phase_hooks import install_vllm_runner"],
                            ["install_vllm_runner(GPUModelRunner)"])
    raise ValueError("The requested engine source file has no reviewed diagnostic patch.")


def prepare(engine, package, destination):
    from download_checkpoints import require_compute_allocation

    require_compute_allocation()
    package = package.resolve(strict=True)
    if not package.is_dir() or package.name != engine or not (package / "__init__.py").is_file():
        raise ValueError("Select the existing " + engine + " import package containing __init__.py.")
    destination = validate_destination(engine, package, destination)
    copied_package = destination / engine
    relatives = {
        "sglang": ["srt/managers/scheduler_components/batch_result_processor.py", "srt/managers/tp_worker.py",
                   "srt/speculative/eagle_worker_v2.py", "srt/speculative/dflash_worker_v2.py"],
        "vllm": ["v1/core/sched/scheduler.py", "v1/core/sched/output.py", "v1/worker/gpu_model_runner.py",
                 "v1/worker/gpu/model_runner.py"],
    }[engine]
    changes = []
    for relative in relatives:
        original_path = package / relative
        if not original_path.is_file() or original_path.is_symlink():
            raise ValueError("The reviewed source file is missing or is a symbolic link: " + relative)
        original = original_path.read_bytes()
        text = original.decode("utf-8")
        patched_text = patch_source(engine, relative, text)
        compile(patched_text, str(copied_package / relative), "exec")
        patched = patched_text.encode("utf-8")
        diff = "".join(difflib.unified_diff(text.splitlines(keepends=True), patched_text.splitlines(keepends=True),
                                            fromfile=str(original_path), tofile=str(copied_package / relative)))
        changes.append({"relative_path": relative, "original_sha256": sha256(original),
                        "patched_sha256": sha256(patched), "unified_diff": diff, "patched_bytes": patched})
    os.umask(0o077)
    created_destination = not destination.exists()
    destination.mkdir(parents=True, exist_ok=True)
    created_manifest = False
    manifest = destination / "diagnostic-instrumentation.json"
    linked_libraries = []
    try:
        print("Copying Python files and package data. Native shared libraries use hard links on the same filesystem.", flush=True)
        shutil.copytree(package, copied_package,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
                        copy_function=package_copy_function(package, linked_libraries))
        for change in changes:
            path = copied_package / change["relative_path"]
            original_path = package / change["relative_path"]
            require_independent_file(original_path, path)
            if sha256(path.read_bytes()) != change["original_sha256"]:
                raise ValueError("The source changed while the package was copied: " + change["relative_path"])
            atomic_replace(path, change["patched_bytes"])
            require_independent_file(original_path, path)
            if sha256(original_path.read_bytes()) != change["original_sha256"]:
                raise ValueError("The original Python source changed during diagnostic preparation: " + change["relative_path"])
        evidence = {"schema": 1, "engine": engine, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
                    "uid": os.getuid(), "source_package": str(package),
                    "destination": str(destination), "copied_package": str(copied_package),
                    "copy_strategy": "Independent Python and data copies with hard-linked ELF shared libraries.",
                    "linked_native_libraries": linked_libraries,
                    "results_relative_package": copied_package.relative_to(
                        (Path(os.environ["HPML_ROOT"]).resolve() / "results").resolve()).as_posix(),
                    "patches": [{key: value for key, value in change.items() if key != "patched_bytes"}
                                for change in changes]}
        with manifest.open("x", encoding="utf-8") as handle:
            created_manifest = True
            json.dump(evidence, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except BaseException:
        if created_manifest:
            manifest.unlink()
        if copied_package.exists():
            shutil.rmtree(copied_package)
        if created_destination and destination.exists() and not any(destination.iterdir()):
            destination.rmdir()
        raise
    print("Prepared the isolated %s diagnostic package at %s." % (engine, copied_package), flush=True)
    print("Saved the source checksums and complete patch in %s." % (destination / "diagnostic-instrumentation.json"), flush=True)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("sglang", "vllm"), required=True)
    parser.add_argument("--package", type=Path, required=True, help="Existing import package containing __init__.py.")
    parser.add_argument("--destination", type=Path, required=True, help="New engine directory under the trial's diagnostic-runtime directory.")
    args = parser.parse_args()
    prepare(args.engine, args.package, args.destination)


if __name__ == "__main__":
    main()
