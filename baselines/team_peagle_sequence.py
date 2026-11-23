#!/usr/bin/env python3
"""Run the team reproduction sequentially and preserve phase and process evidence."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
import uuid

from download_checkpoints import require_compute_allocation
from team_peagle_trial import CONFIGURATIONS, NOTEBOOK_SHA256, export_notebook, sha256
from trial_recorder import BENCHMARKS, INPUT_SHA256, TARGET_REVISION, save_json
from trial_sequence import install_interrupt_handlers, port_free, ready, stop_server, utc


class PhaseLog:
    def __init__(self, path, gpu_uuid):
        self.handle = Path(path).open("x")
        self.context = {"gpu_uuid": gpu_uuid, "allocated_gpu_count": int(gpu_uuid is not None),
                        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname()}

    def write(self, event):
        self.handle.write(json.dumps({**self.context, **event}, allow_nan=False) + "\n")
        self.handle.flush()

    @contextmanager
    def phase(self, name, configuration=None):
        identifier = uuid.uuid4().hex
        started, unix = time.perf_counter_ns(), time.time_ns()
        info = {"phase_id": identifier, "phase": name, "configuration": configuration,
                "started_ns": started, "started_unix_ns": unix}
        self.write({**info, "event": "started"})
        try:
            yield info
        except BaseException as error:
            info.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        else:
            info["status"] = "completed"
        finally:
            ended = time.perf_counter_ns()
            self.write({**info, "event": "ended", "ended_ns": ended, "ended_unix_ns": time.time_ns(),
                        "elapsed_seconds": (ended - started) / 1e9})

    def close(self):
        self.handle.close()


def run_logged(command, path, environment, phase):
    """Keep child logs and stop only the process group created for this command."""
    with Path(path).open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        phase.update(process_pid=process.pid, command=command, log=str(path))
        try:
            returncode = process.wait()
        except BaseException:
            stop_server(process)
            raise
    phase["returncode"] = returncode
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def validate_completed(path, configuration, phase, gpu_uuid, prepared_sha256):
    path = Path(path)
    run = json.loads((path / "run.json").read_text())
    for name, expected in (("configuration", configuration), ("phase", phase), ("gpu_uuid", gpu_uuid),
                           ("prepared_sha256", prepared_sha256), ("input_sha256", INPUT_SHA256),
                           ("target_revision", TARGET_REVISION), ("repetitions", 3)):
        if run[name] != expected:
            raise ValueError("The saved team run differs in " + name)
    summaries = json.loads((path / "summary.json").read_text())
    if set(summaries) != BENCHMARKS:
        raise ValueError("The team summary must cover all ten benchmarks.")
    for repetition in range(1, 4):
        for benchmark in BENCHMARKS:
            block = json.loads((path / f"{repetition:02d}-{benchmark}.json").read_text())
            if len(block["requests"]) != 3 or block["repetition"] != repetition or block["benchmark"] != benchmark:
                raise ValueError("A team benchmark block is incomplete.")
            for request in block["requests"]:
                if "team_rounds" not in request["server_metadata"]:
                    raise ValueError("The team result needs its post-measurement count expansion.")


def execution_phases(output, diagnostics):
    """Read an explicit phase policy before the queued controller starts."""
    path = Path(output) / "sequence-policy.json"
    phases = ["primary", "diagnostic"] if diagnostics else ["primary"]
    if not path.exists():
        return phases, None
    policy = json.loads(path.read_text())
    if (not isinstance(policy, dict) or not {"phases", "reason"} <= set(policy)
            or set(policy) - {"phases", "reason", "checkpoint_storage", "configurations", "reuse_completed_primary"}
            or policy["phases"] not in (["primary"], ["diagnostic"], ["primary", "diagnostic"])
            or not isinstance(policy["reason"], str) or not policy["reason"].strip()
            or policy.get("checkpoint_storage", "copies") not in ("copies", "verified_references")):
        raise ValueError("The sequence policy must name valid phases and explain the selection.")
    if "configurations" in policy:
        selected = policy["configurations"]
        if (not isinstance(selected, list) or not selected
                or any(not isinstance(name, str) or name not in CONFIGURATIONS for name in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("The sequence policy must select each known configuration once.")
    reuse = policy.get("reuse_completed_primary", {})
    if (not isinstance(reuse, dict) or any(name not in CONFIGURATIONS or not isinstance(path, str)
            or not Path(path).is_absolute() for name, path in reuse.items())):
        raise ValueError("The reuse policy must name known configurations and absolute result paths.")
    return policy["phases"], {**policy, "path": str(path), "sha256": sha256(path)}


def require_preparation_idle(output):
    if (Path(output) / ".preparation.lock").exists():
        raise ValueError("The CPU preparation lock is active. Complete its job before starting model evaluation.")


def reuse_completed_primary(source, destination, configuration, gpu_uuid, prepared_sha256):
    """Reuse complete same-GPU evidence while preserving its original run metadata."""
    source, destination = Path(source), Path(destination)
    if not (source / "summary.json").exists() or not (source / "run.json").exists():
        return {"status": "unavailable", "source": str(source)}
    run = json.loads((source / "run.json").read_text())
    if run.get("gpu_uuid") != gpu_uuid:
        return {"status": "different_gpu", "source": str(source), "source_gpu_uuid": run.get("gpu_uuid")}
    try:
        validate_completed(source, configuration, "primary", gpu_uuid, prepared_sha256)
    except FileNotFoundError:
        return {"status": "incomplete", "source": str(source)}
    temporary = destination.parent / (".reuse-" + uuid.uuid4().hex)
    try:
        shutil.copytree(source, temporary)
        validate_completed(temporary, configuration, "primary", gpu_uuid, prepared_sha256)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"status": "reused_completed", "source": str(source),
            "source_slurm_job_id": run.get("slurm_job_id"), "source_run_sha256": sha256(source / "run.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--import-from", type=Path)
    parser.add_argument("--original-vocabulary", type=Path)
    parser.add_argument("--reported-settings", type=Path)
    parser.add_argument("--copy-checkpoints", action="store_true")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--resume-training", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--configurations", nargs="+", choices=tuple(CONFIGURATIONS), default=list(CONFIGURATIONS))
    parser.add_argument("--skip-source-qualification", action="store_true",
                        help="Record that source-equivalence qualification was omitted for this execution.")
    parser.add_argument("--port", type=int, default=31002)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if len(args.configurations) != len(set(args.configurations)):
        parser.error("Select each configuration once.")
    if args.import_from and args.train:
        parser.error("Choose the original checkpoint import or local training for this sequence.")
    root = Path(os.environ["HPML_ROOT"]).resolve(strict=True)
    output = args.output.resolve()
    output.relative_to(root)
    checkpoint_dir = (args.checkpoint_dir or output / "checkpoints").resolve()
    checkpoint_dir.relative_to(root)
    python = str(args.python or root / "envs/team-peagle-cu124-py310/bin/python")
    prepared = args.prepared.resolve(strict=True)
    prepared_sha256 = sha256(prepared)
    base = Path(__file__).resolve().parent
    adapter = base / "team_peagle_trial.py"
    phases, policy = execution_phases(output, args.diagnostics)
    if policy and "checkpoint_storage" in policy:
        args.copy_checkpoints = policy["checkpoint_storage"] == "copies"
    if policy and "configurations" in policy:
        args.configurations = policy["configurations"]
    print("Team configuration order: " + ", ".join(args.configurations), flush=True)
    print("Measurement phases: " + ", ".join(phases), flush=True)
    print("Primary results: " + str(output / "results"), flush=True)
    if not args.run:
        print("Provide --run inside the allocation to execute the sequence.")
        return
    require_compute_allocation()
    require_preparation_idle(output)
    if not port_free(args.port):
        raise ValueError("The team server port is occupied.")
    os.umask(0o077)
    output.mkdir(parents=True, exist_ok=True)
    attempt = output / ("sequence-" + uuid.uuid4().hex)
    attempt.mkdir()
    ledger = PhaseLog(attempt / "phases.jsonl", args.gpu_uuid)
    ledger.write({"event": "execution_policy", "phases": phases, "policy": policy})
    environment = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "HPML_DIAGNOSTIC_DIR"):
        environment.pop(name, None)
    environment.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, HPML_GPU_UUID=args.gpu_uuid,
                       HF_HOME=str(args.hf_home.resolve()), PYTHONDONTWRITEBYTECODE="1")
    previous = install_interrupt_handlers()
    statuses = []
    source = output / "source.ipynb"
    try:
        with ledger.phase("source_export") as phase:
            if source.exists():
                if sha256(source) != NOTEBOOK_SHA256:
                    raise ValueError("The existing source copy differs from the pinned notebook.")
            elif args.notebook:
                if sha256(args.notebook) != NOTEBOOK_SHA256:
                    raise ValueError("The supplied source copy differs from the pinned notebook.")
                with source.open("xb") as handle:
                    handle.write(args.notebook.read_bytes())
            else:
                export_notebook(source)
        if args.setup:
            with ledger.phase("runtime_setup") as phase:
                run_logged(["bash", str(base / "setup_team_peagle.sh")], attempt / "setup.log", environment, phase)
        if not Path(python).is_file():
            raise ValueError("Prepare the isolated team environment before executing models.")
        if args.import_from:
            imported = checkpoint_dir / "import-manifest.json"
            if imported.exists():
                if json.loads(imported.read_text())["source_directory"] != str(args.import_from.resolve()):
                    raise ValueError("The existing import names another shared directory.")
            else:
                command = [python, str(base / "team_peagle_import.py"), "--source", str(args.import_from),
                           "--output", str(checkpoint_dir), "--hf-home", str(args.hf_home),
                           "--prepared", str(prepared)]
                if args.original_vocabulary:
                    command += ["--vocabulary", str(args.original_vocabulary)]
                if args.reported_settings:
                    command += ["--reported-settings", str(args.reported_settings)]
                if args.copy_checkpoints:
                    command.append("--copy-files")
                with ledger.phase("original_checkpoint_import") as phase:
                    run_logged(command, attempt / "import.log", environment, phase)
        if args.train:
            manifest_path = checkpoint_dir / "training-manifest.json"
            complete = manifest_path.exists() and json.loads(manifest_path.read_text()).get("status") == "complete"
            if not complete:
                command = [python, str(base / "team_peagle_train.py"), "--notebook", str(source),
                           "--hf-home", str(args.hf_home), "--prepared", str(prepared), "--output", str(checkpoint_dir)]
                if args.resume_training:
                    command.append("--resume")
                with ledger.phase("data_generation_and_training") as phase:
                    run_logged(command, attempt / "training.log", environment, phase)
        manifest_path = checkpoint_dir / ("import-manifest.json" if (checkpoint_dir / "import-manifest.json").exists()
                                         else "training-manifest.json")
        manifest = json.loads(manifest_path.read_text())
        if (manifest["status"] != "complete" or manifest["evaluation_prepared_sha256"] != prepared_sha256
                or manifest["overlap_match_count"] != 0):
            raise ValueError("Complete the pilot and resolve the recorded evaluation-overlap check.")
        common = ["--notebook", str(source), "--hf-home", str(args.hf_home),
                  "--checkpoint-dir", str(checkpoint_dir), "--gpu-uuid", args.gpu_uuid]
        for mode in phases:
            results = output / ("results" if mode == "primary" else "diagnostic-results")
            results.mkdir(exist_ok=True)
            for configuration in args.configurations:
                destination = results / configuration
                if destination.exists():
                    validate_completed(destination, configuration, mode, args.gpu_uuid, prepared_sha256)
                    statuses.append({"configuration": configuration, "phase": mode, "status": "previously_completed"})
                    continue
                reuse_source = (policy or {}).get("reuse_completed_primary", {}).get(configuration)
                if mode == "primary" and reuse_source:
                    Path(reuse_source).resolve().relative_to(root)
                    reuse = reuse_completed_primary(reuse_source, destination, configuration, args.gpu_uuid, prepared_sha256)
                    ledger.write({"event": "completed_result_reuse", "configuration": configuration, **reuse})
                    if reuse["status"] == "reused_completed":
                        statuses.append({"configuration": configuration, "phase": mode, **reuse})
                        continue
                prefix = mode + "-" + configuration
                status = {"configuration": configuration, "phase": mode, "status": "started", "started_utc": utc()}
                statuses.append(status)
                status["source_equivalence_qualification"] = "omitted" if args.skip_source_qualification else "required"
                if not args.skip_source_qualification and (mode == "primary" or phases == ["diagnostic"]):
                    with ledger.phase("source_equivalence_qualification", configuration) as phase:
                        run_logged([python, str(adapter), "qualify", "--configuration", configuration, *common,
                                    "--prepared", str(prepared), "--output", str(attempt / (prefix + "-qualification.json"))],
                                   attempt / (prefix + "-qualification.log"), environment, phase)
                record = attempt / (prefix + "-server.json")
                command = [python, str(adapter), "serve", "--configuration", configuration, *common,
                           "--server-record", str(record), "--port", str(args.port)]
                if mode == "diagnostic":
                    command += ["--diagnostic-dir", str(output / "diagnostics" / configuration)]
                process = None
                try:
                    with (attempt / (prefix + "-server.log")).open("x") as log:
                        with ledger.phase("server_load", configuration) as phase:
                            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                       env=environment, start_new_session=True)
                            phase.update(process_pid=process.pid, command=command)
                            ready(process, args.port, args.startup_timeout)
                        print(f"{mode}: measuring {configuration}.", flush=True)
                        with ledger.phase(mode + "_measurement", configuration) as phase:
                            run_logged([python, str(adapter), "record", "--configuration", configuration,
                                        "--prepared", str(prepared), "--server-record", str(record),
                                        "--gpu-uuid", args.gpu_uuid, "--url", f"http://127.0.0.1:{args.port}",
                                        "--output", str(destination)], attempt / (prefix + "-recorder.log"),
                                       environment, phase)
                        validate_completed(destination, configuration, mode, args.gpu_uuid, prepared_sha256)
                    status["status"] = "completed"
                except BaseException as error:
                    status.update(status="failed", error_type=type(error).__name__, error=str(error))
                    raise
                finally:
                    if process is not None:
                        with ledger.phase("server_shutdown", configuration):
                            stop_server(process)
                    status["ended_utc"] = utc()
                if not port_free(args.port):
                    raise ValueError("The team server port remains occupied after shutdown.")
            for configuration in args.configurations:
                if configuration == "team-ar":
                    continue
                if not (results / "team-ar" / "summary.json").exists():
                    ledger.write({"event": "paired_comparison_pending", "configuration": configuration,
                                  "reason": "The matching Transformers AR reference is incomplete."})
                    continue
                comparison = attempt / (mode + "-" + configuration + "-comparison.json")
                with ledger.phase("paired_comparison", configuration) as phase:
                    run_logged([python, str(adapter), "compare", str(results / configuration),
                                str(results / "team-ar"), "--output", str(comparison)],
                               attempt / (mode + "-" + configuration + "-comparison.log"), environment, phase)
    finally:
        try:
            save_json(attempt / "sequence.json", {
                "configurations": statuses, "gpu_uuid": args.gpu_uuid, "prepared_sha256": prepared_sha256,
                "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
                "recorded_utc": utc(), "controller_sha256": sha256(__file__),
                "measurement_phases": phases, "execution_policy": policy,
                "configurations_requested": args.configurations,
                "source_equivalence_qualification": "omitted" if args.skip_source_qualification else "required",
            })
        finally:
            ledger.close()
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    print("The selected team measurement sequence is complete.", flush=True)


if __name__ == "__main__":
    main()
