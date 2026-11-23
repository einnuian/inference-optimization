#!/usr/bin/env python3
"""Run the prepared configurations sequentially inside one GPU allocation."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

from download_checkpoints import require_compute_allocation
from trial_recorder import CONFIGURATIONS, save_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def port_free(port):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def linux_group_members(group, proc_root=Path("/proc"), uid=None):
    """Return live processes owned by this user in the selected Linux group."""
    uid = os.getuid() if uid is None else uid
    members = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            fields = (entry / "stat").read_text().rpartition(")")[2].split()
            if len(fields) < 3:
                raise RuntimeError("A process state record is incomplete: " + entry.name)
            if int(fields[2]) == group and fields[0] not in {"Z", "X", "x"}:
                members.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def group_alive(group):
    if sys.platform.startswith("linux"):
        return bool(linux_group_members(group))
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise RuntimeError("The operating system denied the process-group state check.") from error


def stop_server(process):
    """Terminate only the process group created for this server."""
    group = process.pid
    for signum, seconds in ((signal.SIGINT, 20), (signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        process.poll()
        if not group_alive(group):
            process.wait(timeout=5)
            return
        try:
            os.killpg(group, signum)
        except ProcessLookupError:
            process.wait(timeout=5)
            return
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            process.poll()
            if not group_alive(group):
                process.wait(timeout=5)
                return
            time.sleep(0.5)
    raise RuntimeError("The owned server process group remains after shutdown.")


class SequenceInterrupted(KeyboardInterrupt):
    def __init__(self, signum):
        super().__init__("The sequence received " + signal.Signals(signum).name + ".")


def install_interrupt_handlers():
    previous = {}
    requested = False

    def interrupt(signum, frame):
        nonlocal requested
        if not requested:
            requested = True
            raise SequenceInterrupted(signum)

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.signal(signum, interrupt)
    return previous


def run_checked(command):
    """Reap an interrupted request client before shutting down its model server."""
    with subprocess.Popen(command) as process:
        try:
            returncode = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def preserve_partial(destination, attempts, configuration, error):
    """Move incomplete evidence into a failed attempt before a complete retry."""
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("The existing result path must be a regular configuration directory.")
    attempt = attempts / (configuration + "-partial-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    attempt.mkdir()
    archived = attempt / "partial-results"
    destination.rename(archived)
    status = {"configuration": configuration, "status": "failed", "recorded_utc": utc(),
              "error_type": type(error).__name__, "error": str(error),
              "preserved_result_rename": {"from": str(destination), "to": str(archived)}}
    save_json(attempt / "status.json", status)
    print(configuration + ": preserved the incomplete results in " + str(archived) + ".", flush=True)
    return status


def ready(process, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("The server exited during startup with code %s." % process.returncode)
        try:
            with urlopen("http://127.0.0.1:%d/health" % port, timeout=3) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("The server did not become ready within the startup timeout.")


def logit_probe_command(args, base, prepared, output, attempt, record, configuration):
    """Build the optional ancillary request for a separate SGLang diagnostic pass."""
    if not args.probe_sglang_logits or CONFIGURATIONS[configuration][0] != "sglang":
        return None
    if args.diagnostic_runtime is None:
        raise ValueError("The score probe requires an isolated diagnostic runtime.")
    return [sys.executable, str(base / "trial_probe_logits.py"),
            "--prepared", str(prepared), "--primary-results", str(output.parent),
            "--configuration", configuration, "--server-record", str(record),
            "--url", "http://127.0.0.1:%d" % args.port,
            "--output", str(attempt / "logit-probe" / "humaneval-source-index-2.json"), "--run"]


def run_ancillary_probe(command, attempt):
    """Preserve a probe failure while allowing ordinary diagnostic warmups to proceed."""
    status = {"scope": "ancillary_agreement_qualification", "started_utc": utc(),
              "command": command, "included_in_primary_timing": False}
    try:
        run_checked(command)
        status["status"] = "completed"
    except subprocess.CalledProcessError as error:
        status.update(status="requires_review", returncode=error.returncode,
            reason="The score probe failed or its output-preservation check requires review. Ordinary diagnostic requests continue.")
        print("The ancillary score probe requires review. Its saved evidence remains available; diagnostic warmups continue.", flush=True)
    status["ended_utc"] = utc()
    path = attempt / "logit-probe" / "status.json"
    path.parent.mkdir(exist_ok=True)
    save_json(path, status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS,
                        default=[name for name in CONFIGURATIONS if name != "ar-sglang"])
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--diagnostic-runtime", type=Path,
                        help="Use isolated engine packages for a separate diagnostic sequence.")
    parser.add_argument("--probe-sglang-logits", action="store_true",
                        help="Record one ancillary target-score request before SGLang diagnostic warmups.")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if len(set(args.configurations)) != len(args.configurations):
        parser.error("Each configuration must appear once.")
    if args.probe_sglang_logits and args.diagnostic_runtime is None:
        parser.error("--probe-sglang-logits requires --diagnostic-runtime.")
    root = Path(os.environ["HPML_ROOT"]).resolve(strict=True)
    output = args.output.resolve()
    output.relative_to(root / "results")
    prepared = args.prepared.resolve(strict=True)
    base = Path(__file__).resolve().parent
    print("Configurations: " + ", ".join(args.configurations), flush=True)
    print("Results directory: " + str(output), flush=True)
    if not args.run:
        print("Provide --run inside the allocation to execute these configurations.")
        return
    require_compute_allocation()
    os.umask(0o077)
    if not port_free(args.port):
        raise RuntimeError("Stop the existing model server before starting the sequence.")
    launch = output / "launch"
    launch.mkdir(parents=True, exist_ok=True)
    attempts = output / "attempts"
    attempts.mkdir(exist_ok=True)
    statuses = []
    previous_handlers = install_interrupt_handlers()
    try:
        for configuration in args.configurations:
            destination = output / configuration
            if destination.exists():
                from trial_wandb import validate_configuration
                try:
                    validate_configuration(output, configuration)
                except (ValueError, KeyError, TypeError, OSError) as error:
                    preserve_partial(destination, attempts, configuration, error)
                else:
                    print(configuration + ": the saved measurements are complete; skipping execution.", flush=True)
                    statuses.append({"configuration": configuration, "status": "previously_completed"})
                    continue
            attempt = attempts / (configuration + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
            attempt.mkdir()
            record = launch / (attempt.name + ".json")
            status = {"configuration": configuration, "started_utc": utc(), "attempt": str(attempt),
                      "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname()}
            statuses.append(status)
            command = [sys.executable, str(base / "trial_server.py"), configuration,
                       "--prepared", str(prepared), "--record", str(record),
                       "--port", str(args.port), "--launch"]
            if args.diagnostic_runtime is not None:
                backend = CONFIGURATIONS[configuration][0]
                package = (args.diagnostic_runtime / backend).resolve(strict=True)
                command.extend(["--diagnostic-package", str(package),
                                "--diagnostic-output", str(attempt / "trace")])
                status["scope"] = "separate_diagnostic"
            server = None
            try:
                print(configuration + ": starting the model server.", flush=True)
                with (attempt / "server.log").open("x") as server_log:
                    server = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT,
                                              start_new_session=True)
                    status["server_pid"] = server.pid
                    ready(server, args.port, args.startup_timeout)
                    print(configuration + ": the server is ready; checking responses.", flush=True)
                    qualification = [sys.executable, str(base / "trial_qualify_stream.py"),
                        "--prepared", str(prepared), "--configuration", configuration,
                        "--url", "http://127.0.0.1:%d" % args.port,
                        "--output", str(attempt / "qualification")]
                    run_checked(qualification)
                    probe = logit_probe_command(args, base, prepared, output, attempt, record, configuration)
                    if probe is not None:
                        print(configuration + ": recording the ancillary HumanEval target-score request.", flush=True)
                        status["ancillary_logit_probe"] = run_ancillary_probe(probe, attempt)
                    print(configuration + ": starting 90 timed requests.", flush=True)
                    recorder = [sys.executable, str(base / "trial_recorder.py"), "record",
                        "--prepared", str(prepared), "--configuration", configuration,
                        "--url", "http://127.0.0.1:%d" % args.port,
                        "--server-record", str(record), "--gpu-uuid", args.gpu_uuid,
                        "--output", str(destination)]
                    run_checked(recorder)
                    from trial_wandb import validate_configuration
                    validate_configuration(output, configuration)
                    status["status"] = "completed"
            except (Exception, KeyboardInterrupt) as error:
                status.update(status="failed", error_type=type(error).__name__, error=str(error))
                print(configuration + ": " + str(error), flush=True)
                if isinstance(error, KeyboardInterrupt):
                    raise
            finally:
                try:
                    if server is not None:
                        stop_server(server)
                finally:
                    status["ended_utc"] = utc()
                    save_json(attempt / "status.json", status)
            if not port_free(args.port):
                raise RuntimeError("The model port remains occupied after shutdown.")
            time.sleep(2)
    finally:
        try:
            path = output / ("sequence-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
            save_json(path, {"configurations": statuses, "recorded_utc": utc()})
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
    if any(item["status"] == "failed" for item in statuses):
        raise SystemExit("Some configurations need attention. Completed measurements are saved.")
    print("Every selected configuration has complete saved measurements.", flush=True)


if __name__ == "__main__":
    main()
