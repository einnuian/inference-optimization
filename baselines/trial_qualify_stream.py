#!/usr/bin/env python3
"""Check native streaming and stopping behavior before a benchmark trial."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from download_checkpoints import require_compute_allocation
from trial_recorder import (
    CONFIGURATIONS,
    TARGET_REVISION,
    ResponseTrace,
    check_ids,
    request_payload,
    save_json,
    sse_events,
)


REQUEST_TIMEOUT_SECONDS = 900
NATURAL_STOP_TOKEN_LIMIT = 2048
TESTS = (
    ("one-token", 1, "The one-token check"),
    ("natural-stop", NATURAL_STOP_TOKEN_LIMIT, "The natural-stop check"),
)


class QualificationError(Exception):
    """Describe a request that fails the streaming qualification checks."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def loopback_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a valid local model URL.") from exc
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise argparse.ArgumentTypeError(
            "Use the model's loopback HTTP address, such as http://127.0.0.1:31000."
        )
    return value.rstrip("/")


def finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def load_prepared(path: Path) -> tuple[dict, dict, str]:
    raw = path.read_bytes()
    bundle = json.loads(raw)
    if not isinstance(bundle, dict):
        raise QualificationError("The prepared bundle must contain a JSON object.")
    if bundle.get("target_revision") != TARGET_REVISION:
        raise QualificationError(
            "The prepared bundle's target revision differs from the recorder's "
            "target revision. Prepare the inputs with the required target revision."
        )
    warmups = bundle.get("warmups")
    if not isinstance(warmups, list) or not warmups or not isinstance(warmups[0], dict):
        raise QualificationError("The prepared bundle must contain its first warmup prompt.")
    row = warmups[0]
    messages = row.get("messages")
    if (
        not isinstance(messages, list)
        or len(messages) != 1
        or not isinstance(messages[0], dict)
        or messages[0].get("role") != "user"
        or not isinstance(messages[0].get("content"), str)
        or not messages[0]["content"]
    ):
        raise QualificationError("The first warmup must contain one complete user prompt.")
    check_ids(row.get("input_token_ids"))
    check_ids(bundle.get("stop_token_ids"))
    if not row["input_token_ids"] or not bundle["stop_token_ids"]:
        raise QualificationError("The prepared prompt and stop-token lists must contain token IDs.")
    context_length = bundle.get("context_length")
    if (
        not isinstance(context_length, int)
        or isinstance(context_length, bool)
        or context_length < len(row["input_token_ids"]) + NATURAL_STOP_TOKEN_LIMIT
    ):
        raise QualificationError(
            "The prepared context length must fit the prompt and its 2,048-token output limit."
        )
    return bundle, row, hashlib.sha256(raw).hexdigest()


def perform_request(
    *,
    args: argparse.Namespace,
    backend: str,
    bundle: dict,
    row: dict,
    test_name: str,
    token_limit: int,
    request_id: str,
    output_path: Path,
    metadata: dict,
) -> dict:
    endpoint, payload = request_payload(backend, row, bundle, args.model, request_id)
    if backend == "sglang":
        payload["sampling_params"]["max_new_tokens"] = token_limit
    else:
        payload["max_tokens"] = token_limit
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    request = Request(
        args.url + endpoint,
        data=json.dumps(payload).encode(),
        headers=headers,
    )
    context = {
        **metadata,
        "test": test_name,
        "request_id": request_id,
        "prompt_id": row.get("id"),
        "messages": row["messages"],
        "input_token_ids": row["input_token_ids"],
        "output_token_limit": token_limit,
        "request_url": args.url + endpoint,
        "request_headers": headers,
        "request_payload": payload,
        "submitted_unix_ns": time.time_ns(),
    }
    wire_events: list[dict] = []
    submitted_ns = time.perf_counter_ns()
    trace = ResponseTrace(
        backend,
        submitted_ns,
        bundle["stop_token_ids"],
        sglang_incremental=args.sglang_incremental,
    )
    context["submitted_ns"] = submitted_ns
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            content_type = response.headers.get("Content-Type", "")
            context["response_content_type"] = content_type
            context["response_status"] = response.status
            if "text/event-stream" not in content_type.lower():
                raise QualificationError(
                    "The model response must use the text/event-stream content type."
                )
            for received_ns, event in sse_events(response, clock=time.perf_counter_ns):
                wire_events.append({"received_ns": received_ns, "payload": event})
                if received_ns - submitted_ns > REQUEST_TIMEOUT_SECONDS * 1_000_000_000:
                    raise QualificationError("The streaming request exceeded 900 seconds.")
                trace.accept(received_ns, event)
        result = trace.result()
    except (Exception, KeyboardInterrupt) as exc:
        save_json(
            output_path,
            {
                **context,
                "completion_status": "incomplete",
                "observed_unix_ns": time.time_ns(),
                "wire_events": wire_events,
                "error_type": type(exc).__name__,
                "error": str(exc) or "The request was interrupted.",
            },
        )
        raise
    record = {
        **context,
        **result,
        "completion_status": "completed",
        "observed_unix_ns": time.time_ns(),
        "wire_events": wire_events,
    }
    # Preserve the complete response before checking the qualification criteria.
    save_json(output_path, record)
    return record


def check_result(record: dict, test_name: str) -> list[str]:
    problems: list[str] = []
    native_ids = record.get("native_output_token_ids")
    check_ids(native_ids)
    if not native_ids:
        problems.append("The response did not include native output token IDs.")
    if not isinstance(record.get("output_text"), str) or not record["output_text"]:
        problems.append("The response did not include generated text.")
    if record.get("first_content_ns") is None or not finite_nonnegative(record.get("ttft_seconds")):
        problems.append("The response did not produce a valid time to first token.")
    if test_name == "one-token":
        if len(native_ids) > 1:
            problems.append("The one-token request returned more than one native output token.")
        if record.get("finish_reason") != "length":
            problems.append(
                "The one-token request must finish at its one-token limit. "
                "A natural stop before that limit leaves this setup check incomplete."
            )
    elif record.get("finish_reason") != "stop":
        problems.append(
            "The natural-stop request must reach a model stopping condition "
            "within its 2,048-token limit."
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check a running model with two qualification requests. "
            "The saved traces belong to qualification and are excluded from primary measurements."
        )
    )
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--configuration", required=True, choices=CONFIGURATIONS)
    parser.add_argument("--url", type=loopback_url, default="http://127.0.0.1:31000")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="hpml-target")
    parser.add_argument("--sglang-incremental", action="store_true")
    args = parser.parse_args()
    backend, reference = CONFIGURATIONS[args.configuration]
    if args.sglang_incremental and backend != "sglang":
        parser.error("The incremental streaming option applies to SGLang configurations.")
    output_created = False
    requests: list[dict] = []
    metadata: dict = {}
    started_at = utc_now()
    try:
        require_compute_allocation()
        bundle, row, prepared_sha256 = load_prepared(args.prepared)
        args.output = args.output.expanduser().resolve()
        args.output.mkdir(parents=True, exist_ok=False)
        output_created = True
        qualification_id = "qualification-" + uuid4().hex
        metadata = {
            "schema_version": 1,
            "kind": "streaming-qualification",
            "included_in_primary_measurements": False,
            "qualification_id": qualification_id,
            "configuration": args.configuration,
            "backend": backend,
            "reference_configuration": reference,
            "model": args.model,
            "target_revision": TARGET_REVISION,
            "prepared_sha256": prepared_sha256,
            "input_sha256": bundle.get("input_sha256"),
            "context_length": bundle["context_length"],
            "stop_token_ids": bundle["stop_token_ids"],
            "sglang_incremental": args.sglang_incremental,
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "started_at": started_at,
            "duration_clock": "time.perf_counter_ns",
            "event_timestamp_clock": "time.time_ns",
        }
        save_json(args.output / "run.json", {**metadata, "planned_request_count": 2})
        for ordinal, (test_name, token_limit, label) in enumerate(TESTS, start=1):
            request_id = f"{qualification_id}-{args.configuration}-{test_name}"
            filename = f"{ordinal:02d}-{test_name}.json"
            summary = {
                "test": test_name,
                "request_id": request_id,
                "trace_file": filename,
                "completion_status": "submitted",
            }
            requests.append(summary)
            print(f"{label} submits this exact prompt:", flush=True)
            print(row["messages"][0]["content"], flush=True)
            unit = "token" if token_limit == 1 else "tokens"
            print(f"The request allows up to {token_limit:,} generated {unit}.", flush=True)
            try:
                record = perform_request(
                    args=args, backend=backend, bundle=bundle, row=row,
                    test_name=test_name, token_limit=token_limit,
                    request_id=request_id, output_path=args.output / filename,
                    metadata=metadata,
                )
            except (Exception, KeyboardInterrupt):
                summary["completion_status"] = "incomplete"
                raise
            summary.update(
                completion_status="completed",
                output_tokens=record["output_tokens"],
                native_output_tokens=len(record["native_output_token_ids"]),
                finish_reason=record["finish_reason"],
                ttft_seconds=record["ttft_seconds"],
                end_to_end_seconds=record["end_to_end_seconds"],
            )
            problems = check_result(record, test_name)
            summary["passed"] = not problems
            summary["problems"] = problems
            if problems:
                raise QualificationError(" ".join(problems))
            print(f"{label} passed. The complete response trace is saved.", flush=True)
        save_json(
            args.output / "qualification.json",
            {
                **metadata,
                "status": "qualified",
                "qualified": True,
                "completed_at": utc_now(),
                "request_count": len(requests),
                "completed_request_count": len(requests),
                "total_returned_output_tokens": sum(item["output_tokens"] for item in requests),
                "total_native_output_tokens": sum(item["native_output_tokens"] for item in requests),
                "token_count_complete": True,
                "requests": requests,
            },
        )
        print("Both qualification requests passed. Their traces are saved for this configuration.", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if output_created:
            save_json(
                args.output / "qualification.json",
                {
                    **metadata,
                    "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                    "qualified": False,
                    "completed_at": utc_now(),
                    "request_count": len(requests),
                    "completed_request_count": sum(
                        item["completion_status"] == "completed" for item in requests
                    ),
                    "total_returned_output_tokens": sum(
                        item.get("output_tokens", 0) for item in requests
                    ),
                    "total_native_output_tokens": sum(
                        item.get("native_output_tokens", 0) for item in requests
                    ),
                    "token_count_complete": all(
                        item["completion_status"] == "completed" for item in requests
                    ),
                    "requests": requests,
                    "error_type": type(exc).__name__,
                    "error": str(exc) or "The qualification was interrupted.",
                },
            )
        print(
            "The qualification has stopped. "
            + (str(exc) or "The qualification was interrupted."),
            file=sys.stderr,
            flush=True,
        )
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    sys.exit(main())
