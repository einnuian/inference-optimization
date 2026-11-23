#!/usr/bin/env python3
"""Prepare the pinned team environment and verify original inputs on a CPU node."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import time
import uuid

from download_checkpoints import require_compute_allocation
from team_peagle_sequence import PhaseLog, run_logged
from team_peagle_trial import NOTEBOOK_SHA256, sha256
from trial_recorder import INPUT_SHA256, TARGET_REVISION, save_json


@contextmanager
def preparation_lock(output):
    lock = Path(output) / ".preparation.lock"
    lock.mkdir()
    try:
        (lock / "owner.json").write_text(json.dumps({
            "pid": os.getpid(), "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_unix_ns": time.time_ns(),
        }))
        yield
    finally:
        (lock / "owner.json").unlink(missing_ok=True)
        lock.rmdir()


def validate_import(directory, source, prepared):
    manifest = json.loads((Path(directory) / "import-manifest.json").read_text())
    expected = {
        "artifact_kind": "team_original_checkpoint_import", "status": "complete",
        "source_directory": str(Path(source).resolve()),
        "evaluation_prepared_sha256": sha256(prepared), "evaluation_input_sha256": INPUT_SHA256,
        "evaluation_target_revision": TARGET_REVISION, "evaluation_notebook_sha256": NOTEBOOK_SHA256,
        "overlap_match_count": 0,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError("The CPU preparation import differs in " + name)
    for field, filename in (("training_data_sha256", "train_data.jsonl"),
                            ("vocabulary_sha256", "draft_vocab.json"),
                            ("overlap_report_sha256", "evaluation-overlap.json")):
        if manifest.get(field) != sha256(Path(directory) / filename):
            raise ValueError("The verified input differs from the import manifest: " + filename)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--hf-home", required=True, type=Path)
    args = parser.parse_args()
    require_compute_allocation()
    root = Path(os.environ["HPML_ROOT"]).resolve(strict=True)
    output = args.output.resolve()
    output.relative_to(root)
    os.umask(0o077)
    output.mkdir(parents=True, exist_ok=True)
    attempt = output / ("preparation-" + uuid.uuid4().hex)
    attempt.mkdir()
    base = Path(__file__).resolve().parent
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "HPML_DIAGNOSTIC_DIR"):
        environment.pop(key, None)
    environment.update(CUDA_VISIBLE_DEVICES="", HF_HOME=str(args.hf_home.resolve()),
                       PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    ledger = PhaseLog(attempt / "phases.jsonl", None)
    record = {"status": "started", "slurm_job_id": os.environ["SLURM_JOB_ID"],
              "hostname": socket.gethostname(), "allocated_gpu_count": 0,
              "started_unix_ns": time.time_ns(), "prepared_sha256": sha256(args.prepared),
              "source_directory": str(args.source.resolve()), "preparer_sha256": sha256(__file__)}
    try:
        with preparation_lock(output):
            with ledger.phase("cpu_runtime_preparation") as phase:
                run_logged(["bash", str(base / "setup_team_peagle.sh"), "--prepare-only"],
                           attempt / "setup.log", environment, phase)
            checkpoint_dir = output / "checkpoints"
            if not (checkpoint_dir / "import-manifest.json").exists():
                with ledger.phase("cpu_checkpoint_verification") as phase:
                    run_logged([str(root / "envs/team-peagle-cu124-py310/bin/python"),
                                str(base / "team_peagle_import.py"), "--source", str(args.source),
                                "--output", str(checkpoint_dir), "--hf-home", str(args.hf_home),
                                "--prepared", str(args.prepared)], attempt / "import.log", environment, phase)
            manifest = validate_import(checkpoint_dir, args.source, args.prepared)
            record.update(status="complete", import_manifest_sha256=sha256(checkpoint_dir / "import-manifest.json"),
                          training_data_sha256=manifest["training_data_sha256"],
                          overlap_match_count=manifest["overlap_match_count"])
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record["ended_unix_ns"] = time.time_ns()
        save_json(attempt / "preparation.json", record)
        ledger.close()
    print("The pinned runtime and original checkpoint import are prepared for GPU evaluation.", flush=True)


if __name__ == "__main__":
    main()
