#!/usr/bin/env python3
"""Record visible GPU memory samples and completed Slurm allocation evidence."""

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import socket
import time

from trial_recorder import save_json


def gpu(args):
    from download_checkpoints import require_compute_allocation
    require_compute_allocation()
    command = ["nvidia-smi", "--id=" + args.device, "--query-gpu=timestamp,uuid,memory.used",
               "--format=csv,noheader,nounits", "--loop-ms=100"]
    os.umask(0o077)
    with args.output.open("x", encoding="utf-8") as handle:
        print("GPU memory monitor: " + " ".join(command), flush=True)
        print("Each line shows the current observation. Press Ctrl+C after server shutdown.", flush=True)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        try:
            for line in process.stdout:
                fields = next(csv.reader([line]))
                if len(fields) != 3:
                    raise ValueError("Unexpected nvidia-smi output: " + line)
                timestamp, device, memory = (field.strip() for field in fields)
                row = {"unix_ns": time.time_ns(), "monotonic_ns": time.perf_counter_ns(),
                       "device_timestamp": timestamp, "gpu_uuid": device,
                       "hostname": socket.gethostname(),
                       "memory_used_mib": float(memory), "slurm_job_id": os.environ["SLURM_JOB_ID"]}
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                print("%s | %s | %.3f GiB" % (timestamp, device, float(memory) / 1024), flush=True)
            if process.wait() != 0:
                raise RuntimeError("The GPU monitor exited with an error.")
        except KeyboardInterrupt:
            print("Stopping the GPU memory monitor.", flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)


def allocation(args):
    if not re.fullmatch(r"[0-9]+", args.job_id):
        raise ValueError("Use the parent Slurm job's numeric ID.")
    command = ["sacct", "-n", "-P", "-X", "-j", args.job_id,
               "--format=JobIDRaw,Start,End,ElapsedRaw,AllocTRES,State"]
    print("Reading allocation evidence: " + " ".join(command), flush=True)
    raw = subprocess.check_output(command, text=True)
    matches = [line.split("|") for line in raw.splitlines() if line.split("|")[0] == args.job_id]
    if len(matches) != 1:
        raise ValueError("The query must return one unambiguous parent allocation.")
    job, start, end, elapsed, tres, state = matches[0][:6]
    match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", tres)
    if not match or end in {"Unknown", "None", ""} or state in {"RUNNING", "PENDING"}:
        raise ValueError("Wait for a completed allocation with an explicit GPU count.")
    save_json(args.output, {"job_id": job, "start": start, "end": end, "elapsed_seconds": int(elapsed),
                           "gpu_count": int(match.group(1)), "allocated_gpu_seconds": int(elapsed) * int(match.group(1)),
                           "state": state, "alloc_tres": tres, "source_command": command, "raw_sacct": raw})
    print("Saved the complete parent-job allocation record.", flush=True)


def resource_ratios(output_tokens, gpu_seconds, net_charge=None):
    if output_tokens < 0 or gpu_seconds <= 0:
        raise ValueError("Use nonnegative output tokens and positive allocation time.")
    return {"allocated_gpu_seconds_per_million_output_tokens": gpu_seconds * 1e6 / output_tokens if output_tokens else None,
            "output_tokens_per_gpu_second": output_tokens / gpu_seconds,
            "gpu_cost_per_million_output_tokens": net_charge * 1e6 / output_tokens
            if output_tokens and net_charge is not None else None}


def report(args):
    meta = json.loads((args.run / "run.json").read_text())
    allocation_record = json.loads(args.allocation.read_text())
    if meta["slurm_job_id"] != allocation_record["job_id"]:
        raise ValueError("The run and allocation must belong to the same job.")
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()]
    if not samples or any(row["slurm_job_id"] != meta["slurm_job_id"] or row["gpu_uuid"] != meta["gpu_uuid"]
                          or row["hostname"] != meta["hostname"] for row in samples):
        raise ValueError("Memory samples must identify the same job, GPU and compute node.")
    warmups = json.loads((args.run / "warmup.json").read_text())
    blocks = [json.loads(path.read_text()) for path in sorted(args.run.glob("[0-9][0-9]-*.json"))]
    if len(blocks) != 30:
        raise ValueError("The resource report requires the complete trial.")
    total = sum(row["output_tokens"] for row in warmups) + sum(block["metrics"]["output_tokens"] for block in blocks)
    charge = None
    currency = None
    if args.billing:
        billing = json.loads(args.billing.read_text())
        if billing["job_id"] != meta["slurm_job_id"] or not billing.get("source"):
            raise ValueError("Billing evidence must identify the job and its source.")
        charge, currency = billing["net_charge"], billing["currency"]
    timed = []
    for block in blocks:
        observed = [row["memory_used_mib"] / 1024 for row in samples
                    if block["started_ns"] <= row["monotonic_ns"] <= block["ended_ns"]]
        gpu_seconds = block["metrics"]["elapsed_seconds"] * allocation_record["gpu_count"]
        timed.append({"benchmark": block["benchmark"], "repetition": block["repetition"],
                      "scope": "Timed benchmark interval", "sampled_device_memory_peak_gib": max(observed) if observed else None,
                      "memory_sample_count": len(observed),
                      **resource_ratios(block["metrics"]["output_tokens"], gpu_seconds)})
    save_json(args.output, {"shared_allocation": {
               "allocation_record": str(args.allocation),
               "device_samples": str(args.samples),
               "scope": "The allocation includes multiple configurations and ancillary requests.",
               "accounting_note": "The trial report records the allocation totals once. Each configuration report contains its timed benchmark results."},
               "timed_benchmarks": timed,
               "memory_note": "The device peak is sampled every 100 ms. Shorter peaks can occur between samples.",
               "framework_memory_peaks": None})
    print("Saved complete-job and timed-benchmark resource results separately.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    monitor = sub.add_parser("gpu")
    monitor.add_argument("--device", required=True, help="Assigned GPU UUID from nvidia-smi.")
    monitor.add_argument("--output", type=Path, required=True)
    account = sub.add_parser("allocation")
    account.add_argument("job_id")
    account.add_argument("--output", type=Path, required=True)
    result = sub.add_parser("report")
    result.add_argument("--run", type=Path, required=True)
    result.add_argument("--samples", type=Path, required=True)
    result.add_argument("--allocation", type=Path, required=True)
    result.add_argument("--billing", type=Path)
    result.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"gpu": gpu, "allocation": allocation, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
