#!/usr/bin/env python3
"""Print read-only Insomnia environment evidence without running a model."""

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command(args):
    if not shutil.which(args[0]):
        return {"command": args, "status": "unavailable"}
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {
            "command": args,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except subprocess.TimeoutExpired:
        return {"command": args, "status": "timeout"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("login", "gpu"), required=True)
    args = parser.parse_args()
    if args.mode == "gpu" and not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run GPU inspection inside a Slurm GPU allocation.")
    record = {
        "record_type": "environment_preflight",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "mode": args.mode,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID", "SLURM_JOB_ACCOUNT", "SLURM_JOB_PARTITION",
                "SLURM_JOB_NODELIST", "SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE",
                "SLURM_NTASKS", "SLURM_CONF", "LOADEDMODULES",
                "CUDA_VISIBLE_DEVICES", "HF_HOME", "HPML_ROOT",
            )
        },
        "checks": [],
    }
    commands = [["id", "-un"], ["uname", "-a"]]
    if args.mode == "login":
        commands += [
            ["sinfo", "-N", "-o", "%N|%P|%G|%f|%t"],
            ["sacctmgr", "-nP", "show", "assoc", f"user={os.environ.get('USER', '')}",
             "format=Account,Partition,QOS"],
        ]
        storage = Path(os.environ.get("HPML_ROOT", str(Path.home())))
        commands += [["df", "-h", str(storage)], ["df", "-i", str(storage)]]
    else:
        commands += [
            ["nvidia-smi"],
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
             "--format=csv,noheader"],
            ["nvcc", "--version"],
            ["apptainer", "--version"],
            ["scontrol", "show", "job", os.environ["SLURM_JOB_ID"]],
        ]
        try:
            import torch
            record["torch"] = {
                "version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "devices": [
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "capability": list(torch.cuda.get_device_capability(i)),
                        "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                    }
                    for i in range(torch.cuda.device_count())
                ],
            }
        except Exception as exc:
            record["torch"] = {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    record["checks"] = [command(item) for item in commands]
    print(json.dumps(record, indent=2))
    if args.mode == "gpu" and not record["torch"].get("cuda_available", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
