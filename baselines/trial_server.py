#!/usr/bin/env python3
"""Print or launch one prepared server in the foreground."""

import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import time
from datetime import datetime, timezone

from trial_recorder import CONFIGURATIONS, TARGET_REVISION, save_json


def server_command(configuration, root, bundle, port):
    home = root / "cache/huggingface"
    backend, _ = CONFIGURATIONS[configuration]
    context = str(bundle["context_length"])
    environment = {"HPML_ROOT": str(root), "HF_HOME": str(home), "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                   "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1", "PYTHONNOUSERSITE": "1",
                   "PYTHONDONTWRITEBYTECODE": "1"}
    if bundle["target_revision"] != TARGET_REVISION:
        raise ValueError("Use the prepared target revision.")
    if backend == "sglang":
        from qualify_sglang import require_snapshot
        target, provenance = require_snapshot(home, "target")
        container_path = lambda path: "/hpml/" + str(path.relative_to(root))
        command = ["bash", str(Path(__file__).with_name("setup_sglang.sh")), "--exec",
                   "python3", "-m", "sglang.launch_server", "--model-path", container_path(target),
                   "--served-model-name", "hpml-target", "--host", "127.0.0.1", "--port", str(port),
                   "--dtype", "bfloat16", "--context-length", context, "--mem-fraction-static", "0.7",
                   "--max-running-requests", "1", "--attention-backend", "flashinfer",
                   "--cuda-graph-backend-decode", "disabled", "--cuda-graph-backend-prefill", "disabled",
                   "--disable-radix-cache", "--disable-overlap-schedule", "--random-seed", "0",
                   "--stream-interval", "1"]
        if configuration != "ar-sglang":
            name = "eagle3-sglang" if configuration == "eagle3" else "dflash"
            draft, _ = require_snapshot(home, name)
            if configuration == "eagle3":
                # The released drafter declares 2,048 positions. SGLang's
                # documented EAGLE setting extends its rotary-position cache
                # to the matched target context without changing the weights.
                environment["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
            command += ["--speculative-draft-model-path", container_path(draft)]
            command += (["--speculative-algorithm", "EAGLE3", "--speculative-num-steps", "3",
                         "--speculative-eagle-topk", "4", "--speculative-num-draft-tokens", "16"]
                        if configuration == "eagle3" else
                        ["--speculative-algorithm", "DFLASH", "--speculative-dflash-block-size", "10"])
    else:
        from qualify_hspec import METHODS, require_snapshot, TARGET_ID
        target, provenance = require_snapshot(home, "target", TARGET_ID, TARGET_REVISION)
        environment.update(VLLM_USE_V2_MODEL_RUNNER="0" if configuration in {"ar-vllm-v1", "peagle"} else "1",
                           TRITON_CACHE_DIR=str(root / "cache/hspec/triton"),
                           TORCHINDUCTOR_CACHE_DIR=str(root / "cache/hspec/torchinductor"))
        command = [str(root / "envs/hspec-cu129-py313/bin/python"), "-m", "vllm.entrypoints.openai.api_server",
                   "--model", str(target), "--tokenizer", str(target), "--served-model-name", "hpml-target",
                   "--host", "127.0.0.1", "--port", str(port), "--dtype", "bfloat16", "--seed", "0",
                   "--tensor-parallel-size", "1", "--max-model-len", context, "--max-num-seqs", "1",
                   "--max-num-batched-tokens", context, "--gpu-memory-utilization", "0.65",
                   "--no-enable-prefix-caching", "--enable-chunked-prefill", "--enforce-eager",
                   "--compilation-config", '{"mode":0}', "--attention-config", '{"backend":"FLASH_ATTN","flash_attn_version":2}',
                   "--load-format", "safetensors", "--generation-config", "auto"]
        if configuration in METHODS:
            method = METHODS[configuration]
            draft, _ = require_snapshot(home, configuration, method["repo"], method["revision"])
            command += ["--speculative-config", json.dumps({"model": str(draft), **method["settings"]})]
    return command, environment, provenance


def diagnostic_command(command, configuration, root, package, output, allocation_record):
    """Select the verified package copy for a separate diagnostic launch."""
    import hashlib
    from trial_entrypoint import validate_package

    backend = CONFIGURATIONS[configuration][0]
    root = root.resolve(strict=True)
    results = (root / "results").resolve()
    package = package.resolve(strict=True)
    output = output.resolve()
    allocation_record = allocation_record.resolve()
    try:
        package_relative = package.relative_to(results)
        output_relative = output.relative_to(results)
        allocation_relative = allocation_record.relative_to(results)
    except ValueError as error:
        raise ValueError("The diagnostic package and output must stay inside the selected project's results directory.") from error
    if (not package_relative.parts or not output_relative.parts or not allocation_relative.parts or
            package_relative.parts[0] != output_relative.parts[0] or
            package_relative.parts[0] != allocation_relative.parts[0]):
        raise ValueError("The diagnostic package and output must belong to the same trial directory.")
    package = validate_package(package, backend)
    entrypoint = Path(__file__).resolve().with_name("trial_entrypoint.py")
    if not entrypoint.is_file():
        raise ValueError("The diagnostic serving entrypoint is missing.")
    manifest = package / "diagnostic-instrumentation.json"
    details = {"scope": "separate_diagnostic", "package_directory": str(package),
               "output_directory": str(output), "instrumentation_record": str(manifest),
               "package_results_relative": package_relative.as_posix(),
               "output_results_relative": output_relative.as_posix(),
               "allocation_record_results_relative": allocation_relative.as_posix(),
               "instrumentation_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "entrypoint_sha256": hashlib.sha256(entrypoint.read_bytes()).hexdigest()}

    def launch_path(path):
        if backend == "sglang":
            try:
                return str(Path("/hpml") / path.relative_to(root))
            except ValueError as error:
                raise ValueError("SGLang diagnostic files must stay inside HPML_ROOT for the container mount.") from error
        return str(path)

    module = "sglang.launch_server" if backend == "sglang" else "vllm.entrypoints.openai.api_server"
    positions = [index for index in range(len(command) - 1) if command[index:index + 2] == ["-m", module]]
    if len(positions) != 1:
        raise ValueError("The launch command must contain exactly one reviewed serving module.")
    index = positions[0]
    replacement = [launch_path(entrypoint), "--package", launch_path(package), "--backend", backend,
                   "--diagnostic-output", launch_path(output), "--configuration", configuration,
                   "--allocation-record", launch_path(allocation_record), "--"]
    return command[:index] + replacement + command[index + 2:], details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", choices=CONFIGURATIONS)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=os.environ.get("HPML_ROOT"))
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--launch", action="store_true", help="Replace this foreground process with the server.")
    parser.add_argument("--record", type=Path, help="Save the exact launch settings before starting the server.")
    parser.add_argument("--diagnostic-package", type=Path, help="Select the verified directory containing the copied engine package.")
    parser.add_argument("--diagnostic-output", type=Path, help="Save observations from a separate diagnostic run in this directory.")
    args = parser.parse_args()
    if (args.diagnostic_package is None) != (args.diagnostic_output is None):
        parser.error("Provide both --diagnostic-package and --diagnostic-output for a diagnostic run.")
    if args.diagnostic_package is not None and args.record is None:
        parser.error("Provide --record so the diagnostic entrypoint can verify its host launch record.")
    if args.root is None:
        parser.error("Set HPML_ROOT or provide --root.")
    bundle = json.loads(args.prepared.read_text())
    command, environment, provenance = server_command(args.configuration, args.root, bundle, args.port)
    diagnostic_provenance = None
    if args.diagnostic_package is not None:
        command, diagnostic_provenance = diagnostic_command(
            command, args.configuration, args.root, args.diagnostic_package, args.diagnostic_output, args.record)
    print("Configuration: " + args.configuration, flush=True)
    print("Context length: %d tokens. One active request. Temperature is set by the client." % bundle["context_length"], flush=True)
    print(shlex.join(command), flush=True)
    if args.launch:
        if args.record is None:
            parser.error("Provide a fresh --record path for the launch settings.")
        from download_checkpoints import require_compute_allocation
        require_compute_allocation()
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", args.port))
        os.umask(0o077)
        save_json(args.record, {"configuration": args.configuration, "command": command,
                               "environment": environment, "target_provenance": provenance,
                   "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(), "uid": os.getuid(),
                   "launch_utc": datetime.now(timezone.utc).isoformat(),
                   "launch_unix_ns": time.time_ns(), "launch_perf_counter_ns": time.perf_counter_ns(),
                   "diagnostic": diagnostic_provenance})
        os.environ.update(environment)
        print("Starting the foreground server. Its logs remain in this terminal. Press Ctrl+C to stop.", flush=True)
        os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
