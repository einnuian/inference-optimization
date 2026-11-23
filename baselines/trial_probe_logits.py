#!/usr/bin/env python3
"""Inspect target scores for the third HumanEval prompt during separate qualification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler, ProxyHandler
import uuid

from trial_qualify_stream import loopback_url
from trial_recorder import INPUT_SHA256, TARGET_REVISION, ResponseTrace, request_payload, sse_events

CONFIGURATIONS = ("ar-sglang", "eagle3", "dflash")
PROMPT_ID = "humaneval/source-index/2"
POSITION = 3
CANDIDATE_IDS = [755, 475]
SETTING_FLAGS = (
    "--model-path", "--served-model-name", "--dtype", "--context-length",
    "--attention-backend", "--max-running-requests", "--stream-interval",
    "--cuda-graph-backend-decode", "--cuda-graph-backend-prefill", "--mem-fraction-static",
    "--random-seed", "--speculative-algorithm", "--speculative-draft-model-path",
    "--speculative-num-steps", "--speculative-eagle-topk", "--speculative-num-draft-tokens",
    "--speculative-dflash-block-size",
)
BOOLEAN_FLAGS = ("--disable-radix-cache", "--disable-overlap-schedule")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def flag(command, name):
    if command.count(name) > 1:
        raise ValueError("The launch command repeats " + name)
    if name not in command:
        return None
    index = command.index(name)
    if index + 1 == len(command):
        raise ValueError("The launch command omits the value for " + name)
    return command[index + 1]


def prepare(args):
    primary = args.primary_results.resolve(strict=True)
    bundle = read(args.prepared)
    run_path = primary / args.configuration / "run.json"
    run = read(run_path)
    launch = read(args.server_record)
    if (bundle.get("input_sha256") != INPUT_SHA256 or bundle.get("target_revision") != TARGET_REVISION or
            bundle.get("temperature") != 0 or bundle.get("max_new_tokens") != 2048 or
            run.get("prepared_sha256") != sha256(args.prepared)):
        raise ValueError("The probe requires the exact prepared primary inputs and generation settings.")
    if run.get("configuration") != args.configuration or launch.get("configuration") != args.configuration:
        raise ValueError("The probe and launch record must identify the selected primary configuration.")
    if run.get("sglang_incremental") is not False:
        raise ValueError("The probe requires the primary trial's cumulative SGLang stream format.")
    prior = run["server_settings"]
    if launch.get("target_provenance") != prior.get("target_provenance"):
        raise ValueError("The probe server must preserve the primary target provenance.")
    command = launch.get("command")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise ValueError("The probe requires a complete server launch command.")
    for name in SETTING_FLAGS:
        if flag(command, name) != flag(prior["command"], name):
            raise ValueError("The probe launch differs from the primary setting " + name)
    for name in BOOLEAN_FLAGS:
        if (name in command) != (name in prior["command"]):
            raise ValueError("The probe launch differs from the primary setting " + name)
    expected_algorithm = {"ar-sglang": None, "eagle3": "EAGLE3", "dflash": "DFLASH"}[args.configuration]
    if flag(command, "--speculative-algorithm") != expected_algorithm:
        raise ValueError("The selected speculative method must remain enabled for its probe.")
    if int(flag(command, "--port") or 30000) != (urlsplit(args.url).port or 80):
        raise ValueError("The probe URL must use the recorded server port.")
    rows = [row for row in bundle["prompts"] if row["id"] == PROMPT_ID]
    if len(rows) != 1:
        raise ValueError("The prepared bundle must identify the third HumanEval prompt exactly once.")
    prompt = rows[0]
    expected = []
    paths = []
    for repetition in range(1, 4):
        path = primary / args.configuration / ("%02d-humaneval.json" % repetition)
        selected = [row for row in read(path)["requests"] if row["prompt_id"] == PROMPT_ID]
        if len(selected) != 1 or selected[0]["input_token_ids"] != prompt["input_token_ids"]:
            raise ValueError("Each primary repetition must preserve the exact selected input tokens.")
        expected.append(selected[0])
        paths.append(path)
    fields = ("native_output_token_ids", "output_token_ids", "output_tokens", "output_text", "finish_reason", "native_stop_reason")
    if any(any(row[field] != expected[0][field] for field in fields) for row in expected[1:]):
        raise ValueError("The three primary repetitions differ; review their stability before interpreting this probe.")
    endpoint, payload = request_payload("sglang", prompt, bundle, "hpml-target", "hpml-probe-" + uuid.uuid4().hex)
    payload.update(return_logprob=True, top_logprobs_num=5, token_ids_logprob=CANDIDATE_IDS)
    sources = {str(path): sha256(path) for path in [args.prepared, run_path, args.server_record, *paths]}
    return bundle, prompt, run, launch, expected[0], endpoint, payload, sources


def compare_output(observed, expected):
    fields = ("native_output_token_ids", "output_token_ids", "output_tokens", "output_text", "finish_reason", "native_stop_reason")
    checks = {field: observed.get(field) == expected.get(field) for field in fields}
    return {"preserved": all(checks.values()), "checks": checks}


def logprob_entry(value):
    if (not isinstance(value, (list, tuple)) or len(value) < 2 or type(value[0]) not in (int, float) or
            not math.isfinite(value[0]) or type(value[1]) is not int):
        raise ValueError("The server score must contain a finite log probability and its token ID.")
    return {"log_probability": value[0], "token_id": value[1]}


def inspect_scores(observed, comparison, configuration=None):
    if not comparison["preserved"]:
        return {"eligible": False, "reason": "Requesting scores changed the primary output. Score interpretation requires a matching output."}
    try:
        metadata = observed["server_metadata"]
        if configuration in {"eagle3", "dflash"}:
            rounds = metadata.get("spec_verify_ct")
            if type(rounds) is not int or rounds <= 0:
                raise ValueError("The response must record speculative verification rounds for the selected method.")
        selected = logprob_entry(metadata["output_token_logprobs"][POSITION])
        candidates = [logprob_entry(value) for value in metadata["output_token_ids_logprobs"][POSITION]]
        top = [logprob_entry(value) for value in metadata["output_top_logprobs"][POSITION]]
        if selected["token_id"] != observed["native_output_token_ids"][POSITION]:
            raise ValueError("The returned score position does not identify the emitted token.")
        if {value["token_id"] for value in candidates} != set(CANDIDATE_IDS) or len(candidates) != 2:
            raise ValueError("The server must return both requested candidate scores exactly once.")
        if not top or len({row["token_id"] for row in top}) != len(top):
            raise ValueError("The server must return distinct highest-scoring token entries.")
        values = {row["token_id"]: row["log_probability"] for row in candidates}
        maximum = max(row["log_probability"] for row in top)
        return {"eligible": True, "zero_based_position": POSITION, "selected": selected,
                "candidate_scores": candidates, "top_scores": top,
                "def_minus_import_log_probability": values[755] - values[475],
                "reported_top_token_ids": [row["token_id"] for row in top if row["log_probability"] == maximum],
                "scope": "Recorded target log probabilities at the first known divergent output position. These scores do not establish the cause of a ranking difference."}
    except (KeyError, IndexError, TypeError, ValueError) as error:
        return {"eligible": False, "reason": str(error)}


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("The local model server must respond directly.")


def run_probe(args, prepared):
    from download_checkpoints import require_compute_allocation

    require_compute_allocation()
    bundle, prompt, run, launch, expected, endpoint, payload, sources = prepared
    if str(launch.get("slurm_job_id")) != os.environ.get("SLURM_JOB_ID") or launch.get("hostname") != socket.gethostname():
        raise ValueError("The diagnostic server must belong to this active allocation and compute host.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 1 or not visible[0].strip():
        raise ValueError("The probe requires the single visible GPU from the server allocation.")
    gpu_uuid = subprocess.check_output(["nvidia-smi", "--id=" + visible[0].strip(),
        "--query-gpu=uuid", "--format=csv,noheader,nounits"], text=True, timeout=30).strip()
    if gpu_uuid != run["gpu_uuid"]:
        raise ValueError("The probe must use the same physical GPU as the primary trial.")
    if args.output.exists():
        raise ValueError("Choose a fresh probe output file to preserve previous evidence.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start_unix = time.time_ns()
    submitted = time.perf_counter_ns()
    trace = ResponseTrace("sglang", submitted, bundle["stop_token_ids"], sglang_incremental=False)
    evidence = {"schema_version": 1, "scope": "ancillary_agreement_qualification",
                "configuration": args.configuration, "prompt_id": PROMPT_ID,
                "input_messages": prompt["messages"], "request_url": args.url + endpoint,
                "request_payload": payload, "source_sha256": sources,
                "probe_sha256": sha256(Path(__file__)), "server_launch": launch, "raw_events": [],
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "accounting": {"classification": "ancillary_qualification", "included_in_primary_timing": False,
                    "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
                    "gpu_uuid": gpu_uuid, "input_tokens": len(prompt["input_token_ids"]),
                    "submitted_unix_ns": start_unix, "submitted_monotonic_ns": submitted}}
    # Reserve the evidence filename before submitting the request.
    with args.output.open("x", encoding="utf-8") as output:
        try:
            request = Request(args.url + endpoint, data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
            opener = build_opener(ProxyHandler({}), NoRedirect())
            with opener.open(request, timeout=900) as response:
                if "text/event-stream" not in response.headers.get("Content-Type", ""):
                    raise ValueError("The probe requires the same streaming response format as the primary trial.")
                for received, event in sse_events(response):
                    evidence["raw_events"].append({"received_ns": received, "payload": event})
                    trace.accept(received, event)
            observed = trace.result()
            evidence["response"] = observed
            evidence["output_comparison"] = compare_output(observed, expected)
            evidence["scores_at_first_divergence"] = inspect_scores(observed, evidence["output_comparison"], args.configuration)
            evidence["status"] = "qualified" if evidence["scores_at_first_divergence"]["eligible"] else "requires_review"
            evidence["accounting"].update(output_tokens=observed["output_tokens"],
                native_completion_tokens=len(observed["native_output_token_ids"]), finish_reason=observed["finish_reason"])
        except BaseException as error:
            evidence.update(status="failed", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            evidence["accounting"].update(completed_unix_ns=time.time_ns(), completed_monotonic_ns=time.perf_counter_ns(),
                observed_native_token_count=len(trace.ids))
            json.dump(evidence, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--primary-results", type=Path, required=True)
    parser.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--server-record", type=Path, required=True)
    parser.add_argument("--url", type=loopback_url, default="http://127.0.0.1:31000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prepared = prepare(args)
    print("The qualification uses the third HumanEval prompt with its original input tokens and generation settings.")
    print("The request keeps the selected method enabled and asks for target log probabilities.")
    if not args.run:
        print(json.dumps(prepared[6], indent=2))
        print("Provide --run during the separate diagnostic pass to submit this one request.")
        return
    evidence = run_probe(args, prepared)
    print("Saved the complete raw response and ancillary token accounting to " + str(args.output))
    print("Output preservation: " + str(evidence["output_comparison"]["preserved"]))
    print(json.dumps(evidence["scores_at_first_divergence"], indent=2))
    if evidence["status"] != "qualified":
        raise SystemExit("Review the saved qualification before interpreting its scores.")


if __name__ == "__main__":
    main()
