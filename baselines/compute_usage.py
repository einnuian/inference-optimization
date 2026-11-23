#!/usr/bin/env python3
"""Save a readable, reproducible summary of the user's Slurm allocations."""
import argparse
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import re
import subprocess


FIELDS = ["JobIDRaw", "JobName", "Partition", "Account", "State", "Start", "End",
          "ElapsedRaw", "AllocCPUS", "AllocTRES", "NodeList"]


def gpu_count(tres):
    values = dict(item.split("=", 1) for item in tres.split(",") if "=" in item)
    if "gres/gpu" in values:
        return int(values["gres/gpu"])
    return sum(int(value) for key, value in values.items() if key.startswith("gres/gpu:"))


def parse_jobs(raw, include):
    jobs = []
    seen = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != len(FIELDS):
            raise ValueError("The Slurm record has an unexpected number of fields.")
        row = dict(zip(FIELDS, fields))
        if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", row["JobIDRaw"]):
            raise ValueError("The query must contain parent allocations only.")
        identity = (row["JobIDRaw"], row["Start"])
        if identity in seen:
            raise ValueError("A duplicate allocation would count resources twice.")
        seen.add(identity)
        seconds, cpus, gpus = int(row["ElapsedRaw"]), int(row["AllocCPUS"]), gpu_count(row["AllocTRES"])
        row.update(elapsed_seconds=seconds, allocated_cpus=cpus, allocated_gpus=gpus,
                   allocated_gpu_hours=seconds * gpus / 3600,
                   allocated_cpu_core_hours=seconds * cpus / 3600,
                   project_job=row["JobName"].startswith("hpml-") or row["JobIDRaw"] in include)
        jobs.append(row)
    return jobs


def totals(jobs):
    return {"jobs": len(jobs), "allocated_gpu_hours": sum(row["allocated_gpu_hours"] for row in jobs),
            "allocated_cpu_core_hours": sum(row["allocated_cpu_core_hours"] for row in jobs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", required=True, help="First date to include, in YYYY-MM-DD format.")
    parser.add_argument("--include-job", action="append", default=[], help="Also classify this job as project work.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    datetime.strptime(args.since, "%Y-%m-%d")
    command = ["sacct", "-n", "-P", "-X", "-u", getpass.getuser(), "-S", args.since,
               "--format=" + ",".join(FIELDS)]
    print("Reading the user's parent allocations: " + " ".join(command), flush=True)
    raw = subprocess.check_output(command, text=True)
    jobs = parse_jobs(raw, set(args.include_job))
    project = [row for row in jobs if row["project_job"]]
    record = {"observed_at": datetime.now(timezone.utc).isoformat(), "since": args.since,
              "user": getpass.getuser(), "source_command": command, "raw_sacct": raw,
              "project_rule": "Job names beginning hpml- plus the explicitly included job IDs.",
              "included_job_ids": args.include_job, "project_totals": totals(project),
              "all_user_totals": totals(jobs), "jobs": jobs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(args.output)
    print("Project allocations: %d jobs; %.4f GPU-hours; %.4f CPU-core-hours." %
          (len(project), totals(project)["allocated_gpu_hours"], totals(project)["allocated_cpu_core_hours"]), flush=True)
    print("Allocated time includes setup, model loading, generation and idle time within each allocation.", flush=True)
    print("Saved " + str(args.output), flush=True)


if __name__ == "__main__":
    main()
