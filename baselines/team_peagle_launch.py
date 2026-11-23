#!/usr/bin/env python3
"""Apply a recorded launch policy while preserving the inference adapter source."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from trial_recorder import save_json


def save_omission(path, event, configuration, reason):
    save_json(path, {
        **event, "configuration": configuration, "status": "omitted",
        "source_equivalence_verified": False, "reason": reason,
        "checkpoint_checks": "The unchanged model loader validates checkpoint identity and compatibility.",
    })


def omission(arguments, job_id, policy):
    if not arguments or arguments[0] != "qualify":
        return None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--configuration")
    parser.add_argument("--output", type=Path)
    parsed, _ = parser.parse_known_args(arguments[1:])
    for exception in policy.get("qualification_omissions", []):
        if exception["slurm_job_id"] == job_id and exception["configuration"] == parsed.configuration:
            if parsed.output is None:
                raise ValueError("The omitted qualification needs an explicit evidence path.")
            return parsed, exception
    return None


def load_core(directory, policy):
    path = directory / policy["core_filename"]
    if path.parent != directory or hashlib.sha256(path.read_bytes()).hexdigest() != policy["core_sha256"]:
        raise ValueError("The preserved inference adapter differs from its recorded source.")
    name = "_hpml_preserved_team_adapter"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    directory = Path(__file__).resolve().parent
    policy_path = directory / "team-peagle-launch-policy.json"
    policy = json.loads(policy_path.read_text())
    core = load_core(directory, policy)
    skipped = omission(sys.argv[1:], os.environ.get("SLURM_JOB_ID"), policy)
    event = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_unix_ns": time.time_ns(),
             "command": sys.argv[1] if len(sys.argv) > 1 else None,
             "core_adapter_sha256": policy["core_sha256"],
             "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
             "source_equivalence_qualification": "omitted" if skipped else "unchanged"}
    with (directory / ("launch-events-" + str(event["slurm_job_id"]) + ".jsonl")).open("a") as handle:
        handle.write(json.dumps(event) + "\n")
    if skipped:
        parsed, exception = skipped
        save_omission(parsed.output, event, parsed.configuration, exception["reason"])
        print("The optional source-equivalence probe is omitted. The primary trial and checkpoint loader checks continue.", flush=True)
        return
    core.main()


if __name__ == "__main__":
    main()
elif (Path(__file__).resolve().parent / "team-peagle-launch-policy.json").exists():
    # Imports retain the original adapter API for the recorder and controller.
    _directory = Path(__file__).resolve().parent
    _core = load_core(_directory, json.loads((_directory / "team-peagle-launch-policy.json").read_text()))
    globals().update({name: value for name, value in vars(_core).items() if not name.startswith("_")})
