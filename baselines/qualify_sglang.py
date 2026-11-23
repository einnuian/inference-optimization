#!/usr/bin/env python3
"""Qualify ordinary generation, DFlash and EAGLE-3 on four shared setup prompts.

Run with the prepared EAGLE Python interpreter inside one Slurm GPU allocation.
The host tokenizer supplies canonical input IDs. Each engine stage starts a
fresh pinned SGLang container process. Performance benchmarking follows the
separate experiment protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback
import urllib.error
import urllib.request

from download_checkpoints import require_compute_allocation, require_scratch_cache
from qualify_eagle import PROMPTS, apply_output_policy, compare_output_records, write_record


ARTIFACTS = {
    "target": ("meta-llama/Llama-3.1-8B-Instruct", "0e9e39f249a16976918f6564b8830bc894c89659"),
    "dflash": ("z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat", "d3af30def9601abdd10810aba220d692f0e803f0"),
    "eagle3-sglang": ("jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B", "e5ed08d66f528a95ce89f5d4fd136a28f6def714"),
}


class ServerCleanupError(RuntimeError):
    """The owned server group has retained processes after shutdown."""


def require_snapshot(home: Path, name: str):
    repo, revision = ARTIFACTS[name]
    path = home / "hpml-provenance" / name / f"{revision}.json"
    raw = path.read_bytes()
    record = json.loads(raw)
    if record["repo_id"] != repo or record["revision"] != revision:
        raise RuntimeError(f"The {name} provenance identifies a different artifact.")
    if name == "target":
        acquisition = record.get("acquisition", {})
        if (acquisition.get("weight_source_repo") != repo
                or acquisition.get("weight_source_revision") != revision):
            raise RuntimeError("Complete the verified direct-Meta target import first.")
    snapshot = home / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if snapshot.resolve() != Path(record["snapshot_path"]).resolve():
        raise RuntimeError(f"The {name} snapshot path differs from its provenance.")
    for filename, size in record["required_file_bytes"].items():
        file = snapshot / filename
        if not file.is_file() or file.stat().st_size != size:
            raise RuntimeError(f"The {name} snapshot is incomplete: {filename}.")
    return snapshot, {"repo_id": repo, "revision": revision, "provenance_path": str(path),
                      "provenance_sha256": hashlib.sha256(raw).hexdigest()}


def native_finish_matches(row, stop_ids):
    if not row["raw_boundary_compliant"]:
        return False
    raw = row["raw_generated_token_ids"]
    finish = row["producer_finish_reason"]
    if finish.get("type") == "length":
        return (row["stop_reason"] == "length" and len(raw) == row["max_new_tokens"]
                and finish.get("length") == row["max_new_tokens"])
    if (finish.get("type") == "stop" and row["stop_reason"] == "stop_token"
            and raw and raw[-1] in stop_ids):
        matched = finish.get("matched")
        return raw[-1] in matched if isinstance(matched, list) else matched == raw[-1]
    return False


def request_json(base, path, payload=None, timeout=300):
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + path, data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def stop_server(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    def group_exists():
        process.poll()  # Reap the wrapper while observing its remaining children.
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False

    for timeout, action in ((60, signal.SIGKILL), (15, None)):
        deadline = time.monotonic() + timeout
        while group_exists() and time.monotonic() < deadline:
            time.sleep(0.25)
        if not group_exists():
            return
        if action is not None:
            try:
                os.killpg(process.pid, action)
            except ProcessLookupError:
                return
    raise ServerCleanupError("The server process group remains active; stop qualification and inspect the allocation.")


def execute_stage(stage, output, root, snapshots, provenance, tokenizer, stops, args):
    output.mkdir()
    container_path = lambda path: "/hpml/" + str(path.relative_to(root))
    common = ["--model-path", container_path(snapshots["target"]),
              "--host", "127.0.0.1", "--port", str(args.port), "--dtype", "bfloat16",
              "--context-length", "2048", "--mem-fraction-static", "0.7",
              "--max-running-requests", "1", "--attention-backend", "flashinfer",
              "--cuda-graph-backend-decode", "disabled", "--cuda-graph-backend-prefill", "disabled",
              "--disable-radix-cache", "--disable-overlap-schedule", "--random-seed", "0"]
    extra = []
    if stage == "dflash":
        extra = ["--speculative-algorithm", "DFLASH", "--speculative-draft-model-path",
                 container_path(snapshots["dflash"]), "--speculative-dflash-block-size", "10"]
    elif stage == "eagle3-sglang":
        extra = ["--speculative-algorithm", "EAGLE3", "--speculative-draft-model-path",
                 container_path(snapshots["eagle3-sglang"]), "--speculative-num-steps", "3",
                 "--speculative-eagle-topk", "4", "--speculative-num-draft-tokens", "16"]
    command = ["bash", str(Path(__file__).with_name("setup_sglang.sh")), "--exec",
               "python3", "-m", "sglang.launch_server", *common, *extra]
    write_record(output / "requested-settings.json", {
        "stage": stage, "command": command, "artifacts": provenance,
        "canonical_stop_token_ids": sorted(stops), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "host": socket.gethostname(), "scope": "Four greedy setup prompts; model loading and exact output checks",
        "container_origin": (root / "runs/sglang-setup/image-origin.txt").read_text().strip(),
        "container_sif_sha256_record": (root / "runs/sglang-setup/image-sha256.txt").read_text().strip(),
        "qualification_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    # A pre-existing listener belongs to another task and must remain untouched.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    base = f"http://127.0.0.1:{args.port}"
    with (output / "server.log").open("x") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"The {stage} server exited with code {process.returncode}; inspect server.log.")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=5) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError(f"The {stage} server did not become ready within the startup timeout.")
                time.sleep(2)
            write_record(output / "server-info.json", request_json(base, "/get_server_info"))
            rows = []
            for name, prompt, cap in PROMPTS:
                input_ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True)
                if not isinstance(input_ids, list) or not all(isinstance(t, int) for t in input_ids):
                    raise RuntimeError("The canonical chat template must supply token IDs.")
                payload = {"input_ids": input_ids, "stream": False,
                           "sampling_params": {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
                                               "max_new_tokens": cap, "min_new_tokens": 0,
                                               "stop_token_ids": sorted(stops), "ignore_eos": False,
                                               "skip_special_tokens": True}}
                write_record(output / f"{name}-request.json", payload)
                response = request_json(base, "/generate", payload)
                write_record(output / f"{name}-native-response.json", response)
                raw = response.get("output_ids")
                if not isinstance(raw, list) or not all(isinstance(t, int) for t in raw):
                    raise RuntimeError("The native SGLang response must contain output token IDs.")
                row = {"name": name, "prompt": prompt, "max_new_tokens": cap,
                       "input_token_ids": input_ids, "producer_returned_text": response.get("text"),
                       "producer_finish_reason": response.get("meta_info", {}).get("finish_reason", {}),
                       "producer_completion_tokens": response.get("meta_info", {}).get("completion_tokens"),
                       **apply_output_policy(raw, cap, stops)}
                row["returned_text"] = tokenizer.decode(row["returned_token_ids"], skip_special_tokens=True)
                row["native_finish_reason_compliant"] = native_finish_matches(row, stops)
                rows.append(row)
                print(f"{stage}: {name}, {len(raw)} native output IDs, {row['producer_finish_reason']}", flush=True)
            result = {"stage": stage, "prompts": rows, "artifacts": provenance,
                      "canonical_stop_token_ids": sorted(stops),
                      "raw_producer_boundary": "Native /generate output_ids after SGLang scheduler stopping, before the project's explicit output adapter; preserve finish_reason and completion_tokens separately.",
                      "output_policy": "Keep the first canonical stop token or strict cap, then exclude a terminal stop token from returned output tokens.",
                      "scope": "Four setup prompts; streaming, performance, resource and charge evaluation remain future work."}
            write_record(output / "output.json", result)
            return result
        finally:
            stop_server(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=("dflash", "eagle3-sglang"),
                        default=["dflash", "eagle3-sglang"])
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    args = parser.parse_args()
    require_compute_allocation()
    home = require_scratch_cache()
    root = Path(os.environ["HPML_ROOT"]).resolve()
    output = args.output_dir.resolve()
    if root / "runs" not in output.parents:
        raise RuntimeError("Use a fresh evidence directory under HPML_ROOT/runs.")
    os.umask(0o077)
    snapshots, provenance = {}, {}
    for name in ("target", *args.methods):
        snapshots[name], provenance[name] = require_snapshot(home, name)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshots["target"], local_files_only=True, use_fast=False)
    eos = json.loads((snapshots["target"] / "generation_config.json").read_text())["eos_token_id"]
    stops = set(eos if isinstance(eos, list) else [eos])
    output.mkdir(parents=True, exist_ok=False)
    stages, comparisons = {}, {}
    for stage in ("ordinary", *args.methods):
        try:
            stages[stage] = execute_stage(stage, output / stage, root, snapshots, provenance, tokenizer, stops, args)
        except Exception as error:
            write_record(output / f"{stage}-failure.json", {"error": f"{type(error).__name__}: {error}",
                                                           "traceback": traceback.format_exc()})
            if stage == "ordinary" or isinstance(error, ServerCleanupError):
                write_record(output / "summary.json", {
                    "failed_stage": stage, "comparisons": comparisons,
                    "scope": "Inspect the saved failure and server log before continuing qualification.",
                })
                raise
            comparisons[stage] = {"native_output_ready": False, "exit_code": 1, "reason": "Inspect the saved failure and server log."}
            continue
        if stage != "ordinary":
            comparison = compare_output_records(stages["ordinary"], stages[stage])
            native = all(row["native_finish_reason_compliant"] for row in stages["ordinary"]["prompts"] + stages[stage]["prompts"])
            finish_match = all(a["producer_finish_reason"] == b["producer_finish_reason"]
                               for a, b in zip(stages["ordinary"]["prompts"], stages[stage]["prompts"]))
            text_match = all(a["producer_returned_text"] == b["producer_returned_text"]
                             for a, b in zip(stages["ordinary"]["prompts"], stages[stage]["prompts"]))
            comparison["native_finish_reasons_compliant"] = native
            comparison["native_finish_reasons_match"] = finish_match
            comparison["native_returned_text_matches"] = text_match
            comparison["native_output_ready"] = comparison["native_output_ready"] and native and finish_match and text_match
            comparison["exit_code"] = 0 if comparison["native_output_ready"] else (3 if comparison["adapted_output_ready"] else 2)
            comparisons[stage] = comparison
            write_record(output / f"{stage}-comparison.json", comparison)
    codes = {item["exit_code"] for item in comparisons.values()}
    exit_code = next(code for code in (1, 2, 3, 0) if code in codes)
    write_record(output / "summary.json", {"comparisons": comparisons, "exit_code": exit_code,
                                           "scope": "Four-prompt pretrained setup qualification"})
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
