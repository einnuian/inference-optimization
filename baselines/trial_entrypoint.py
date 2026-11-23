#!/usr/bin/env python3
"""Launch an isolated serving package for a separate diagnostic run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import socket
import subprocess
import sys
import time

CONFIGURATIONS = {"ar-sglang": "sglang", "eagle3": "sglang", "dflash": "sglang",
                  "ar-vllm-v1": "vllm", "peagle": "vllm", "ar-vllm-v2": "vllm", "dspark": "vllm"}


def validate_package(parent, backend):
    parent = parent.resolve(strict=True)
    package = parent / backend
    if not package.is_dir() or not (package / "__init__.py").is_file():
        raise ValueError("The selected directory must contain the copied " + backend + " import package.")
    manifest = parent / "diagnostic-instrumentation.json"
    if manifest.is_symlink():
        raise ValueError("The instrumentation record must be stored beside its copied package.")
    with manifest.open(encoding="utf-8") as handle:
        evidence = json.load(handle)
    if evidence.get("schema") != 1 or evidence.get("engine") != backend:
        raise ValueError("The instrumentation record does not identify the selected package.")
    root_value = os.environ.get("HPML_ROOT")
    if not root_value:
        raise ValueError("Set HPML_ROOT to the existing project directory.")
    results = (Path(root_value).resolve(strict=True) / "results").resolve()
    recorded_relative = evidence.get("results_relative_package")
    if not isinstance(recorded_relative, str):
        raise ValueError("The instrumentation record requires its package path relative to the results directory.")
    relative = Path(recorded_relative)
    if (relative.is_absolute() or len(relative.parts) != 4 or
            re.fullmatch(r"trial-first30-[0-9]{8}", relative.parts[0]) is None or
            relative.parts[1:] != ("diagnostic-runtime", backend, backend)):
        raise ValueError("The instrumentation record contains an invalid diagnostic package location.")
    try:
        actual_relative = package.resolve().relative_to(results)
    except ValueError as error:
        raise ValueError("The copied package must stay inside HPML_ROOT/results.") from error
    if actual_relative != relative:
        raise ValueError("The selected package differs from the recorded location within the results directory.")
    patches = evidence.get("patches")
    required_count = 4
    if not isinstance(patches, list) or len(patches) != required_count:
        raise ValueError("The instrumentation record must identify all %d patched source files." % required_count)
    expected = {
        "sglang": {"srt/managers/scheduler_components/batch_result_processor.py", "srt/managers/tp_worker.py",
                   "srt/speculative/eagle_worker_v2.py", "srt/speculative/dflash_worker_v2.py"},
        "vllm": {"v1/core/sched/scheduler.py", "v1/core/sched/output.py", "v1/worker/gpu_model_runner.py",
                 "v1/worker/gpu/model_runner.py"},
    }[backend]
    if {patch.get("relative_path") for patch in patches} != expected:
        raise ValueError("The instrumentation record must identify the reviewed engine source files.")
    for patch in patches:
        relative = Path(patch["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("The instrumentation record contains an invalid source path.")
        path = package / relative
        if path.is_symlink() or not path.is_file() or package.resolve() not in path.resolve().parents:
            raise ValueError("The patched source file must stay inside its copied package.")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != patch["patched_sha256"]:
            raise ValueError("The copied source differs from its instrumentation record: " + str(relative))
    return parent


def validate_allocation_record(path, parent, output, backend, configuration, results):
    path = path.resolve(strict=True)
    if not path.is_file() or results not in path.parents:
        raise ValueError("The host allocation record must stay inside the project results directory.")
    with path.open(encoding="utf-8") as handle:
        launch = json.load(handle)
    with (parent / "diagnostic-instrumentation.json").open(encoding="utf-8") as handle:
        preparation = json.load(handle)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isdigit():
        raise ValueError("The diagnostic entrypoint requires its active Slurm job ID.")
    for record in (launch, preparation):
        if (str(record.get("slurm_job_id")) != job_id or record.get("hostname") != socket.gethostname() or
                record.get("uid") != os.getuid()):
            raise ValueError("The diagnostic preparation and host launch must identify this job, host, and user.")
    if launch.get("configuration") != configuration:
        raise ValueError("The host launch record identifies a different configuration.")
    for key, now in (("launch_unix_ns", time.time_ns()), ("launch_perf_counter_ns", time.perf_counter_ns())):
        timestamp = launch.get(key)
        if type(timestamp) is not int or not 0 <= now - timestamp <= 3600 * 1_000_000_000:
            raise ValueError("The host launch record must come from this host within the previous hour.")
    diagnostic = launch.get("diagnostic") or {}
    expected = {"scope": "separate_diagnostic", "package_results_relative": parent.relative_to(results).as_posix(),
                "output_results_relative": output.relative_to(results).as_posix(),
                "allocation_record_results_relative": path.relative_to(results).as_posix()}
    if any(diagnostic.get(key) != value for key, value in expected.items()):
        raise ValueError("The host launch record identifies different diagnostic files.")
    manifest_hash = hashlib.sha256((parent / "diagnostic-instrumentation.json").read_bytes()).hexdigest()
    if diagnostic.get("instrumentation_sha256") != manifest_hash:
        raise ValueError("The instrumentation record changed after the host launch check.")
    entrypoint = Path(__file__).resolve()
    if diagnostic.get("entrypoint_sha256") != hashlib.sha256(entrypoint.read_bytes()).hexdigest():
        raise ValueError("The diagnostic entrypoint changed after the host launch check.")
    command = launch.get("command")
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command) or command.count(str(entrypoint)) != 1:
        raise ValueError("The host launch record must contain this diagnostic entrypoint.")
    flags = {"--package": str(parent), "--backend": backend, "--configuration": configuration,
             "--diagnostic-output": str(output), "--allocation-record": str(path)}
    for flag, expected_value in flags.items():
        if command.count(flag) != 1:
            raise ValueError("The host launch record requires exactly one " + flag + " argument.")
        index = command.index(flag)
        if index + 1 >= len(command) or command[index + 1] != expected_value:
            raise ValueError("The host launch record differs in " + flag)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 1 or not visible[0].strip() or visible[0].strip() in {"-1", "NoDevFiles"}:
        raise ValueError("The diagnostic server requires one visible allocated GPU.")
    gpu = subprocess.check_output(["nvidia-smi", "--id=" + visible[0].strip(),
                                   "--query-gpu=uuid,name", "--format=csv,noheader,nounits"],
                                  text=True, timeout=30).strip()
    if len(gpu.splitlines()) != 1 or "L40S" not in gpu:
        raise ValueError("The diagnostic server requires the allocated L40S GPU.")
    gpu_uuid = gpu.partition(",")[0].strip()
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]+", gpu_uuid):
        raise ValueError("The visible GPU must provide its physical UUID.")
    existing_uuid = os.environ.get("HPML_GPU_UUID")
    if existing_uuid is not None and existing_uuid != gpu_uuid:
        raise ValueError("HPML_GPU_UUID differs from the visible allocated GPU.")
    os.environ["HPML_GPU_UUID"] = gpu_uuid
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, help="Directory containing the copied engine package.")
    parser.add_argument("--backend", choices=("sglang", "vllm"), required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    parser.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--allocation-record", type=Path, required=True)
    parser.add_argument("server_args", nargs=argparse.REMAINDER, help="Original server arguments following --.")
    args = parser.parse_args()
    if CONFIGURATIONS[args.configuration] != args.backend:
        parser.error("The configuration must use the selected serving engine.")
    if not args.server_args or args.server_args[0] != "--":
        parser.error("Place the original server arguments after --.")
    from download_checkpoints import require_compute_allocation

    if args.backend == "vllm" or os.environ.get("HPML_ROOT") != "/hpml":
        require_compute_allocation()
    parent = validate_package(args.package, args.backend)
    root_value = os.environ.get("HPML_ROOT")
    if not root_value:
        raise ValueError("Set HPML_ROOT to the existing project directory.")
    root = Path(root_value).resolve(strict=True)
    results = (root / "results").resolve()
    output = args.diagnostic_output.resolve()
    if results not in output.parents:
        raise ValueError("The diagnostic output must stay inside HPML_ROOT/results.")
    validate_allocation_record(args.allocation_record, parent, output, args.backend, args.configuration, results)
    helper_directory = Path(__file__).resolve().parent
    for name in ("trial_round_hooks.py", "trial_phase_hooks.py"):
        if not (helper_directory / name).is_file():
            raise ValueError("The diagnostic helper is missing: " + name)
    if args.backend in sys.modules:
        raise RuntimeError("The serving package loaded before the diagnostic path was selected.")
    os.umask(0o077)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HPML_DIAGNOSTIC_DIR"] = str(output)
    os.environ["HPML_CONFIGURATION"] = args.configuration
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    paths = [str(parent), str(helper_directory)]
    sys.path[:0] = paths
    existing_pythonpath = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + ([existing_pythonpath] if existing_pythonpath else []))
    module = "sglang.launch_server" if args.backend == "sglang" else "vllm.entrypoints.openai.api_server"
    sys.argv = [module, *args.server_args[1:]]
    print("Launching a separate diagnostic run for %s." % args.configuration, flush=True)
    print("The serving engine uses the isolated package at %s." % (parent / args.backend), flush=True)
    print("Diagnostic observations will be saved in %s." % output, flush=True)
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
