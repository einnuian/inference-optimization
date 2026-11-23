#!/usr/bin/env python3
"""Record the shared streaming trial and preserve the observations behind each metric."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import socket
import statistics
import subprocess
import threading
import time
import urllib.request
from urllib.parse import urlparse
import uuid

TARGET_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
INPUT_SHA256 = "b5ae4556936743902d455576aa0d71deb9005ac5f28b8b9a1fd4daa49b20fd1c"
BENCHMARKS = {"mt-bench", "alpaca", "arena-hard-v2", "cnn-dm", "gsm8k",
              "math500", "aime25", "humaneval", "mbpp", "livecodebench"}
CONFIGURATIONS = {
    "ar-sglang": ("sglang", None),
    "eagle3": ("sglang", "ar-sglang"),
    "dflash": ("sglang", "ar-sglang"),
    "ar-vllm-v1": ("vllm", None),
    "peagle": ("vllm", "ar-vllm-v1"),
    "ar-vllm-v2": ("vllm", None),
    "dspark": ("vllm", "ar-vllm-v2"),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


class LiveWriter:
    """Write small progress events outside the request thread."""

    def __init__(self, path):
        self.handle = Path(path).open("x", encoding="utf-8")
        self.queue = queue.Queue()
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._write, name="hpml-progress-writer")
        self.thread.start()

    def _write(self):
        while True:
            event = self.queue.get()
            try:
                if event is None:
                    return
                if self.error is None:
                    self.handle.write(json.dumps(dict(event), ensure_ascii=False, allow_nan=False) + "\n")
                    self.handle.flush()
            except Exception as error:
                if self.error is None:
                    self.error = error
            finally:
                self.queue.task_done()

    def put(self, event):
        if self.closed:
            raise RuntimeError("The progress writer is closed.")
        if any(value is not None and type(value) not in (str, int, float, bool)
               for value in event.values()):
            raise ValueError("Progress events contain scalar values only.")
        self.queue.put_nowait(tuple(event.items()))

    def check(self):
        if self.error is not None:
            raise RuntimeError("The trial progress record could not be saved.") from self.error

    def drain(self):
        self.queue.join()
        self.check()

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.put_nowait(None)
            self.thread.join()
            try:
                self.handle.close()
            except Exception as error:
                if self.error is None:
                    self.error = error
        self.check()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise
        return False


def load_inputs(path):
    raw = Path(path).read_bytes()
    if digest(raw) != INPUT_SHA256:
        raise ValueError("The input file differs from the verified 30-prompt selection.")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if Counter(row["benchmark"] for row in rows) != Counter({b: 3 for b in BENCHMARKS}):
        raise ValueError("Use exactly three prompts from each of the ten benchmarks.")
    if len({row["id"] for row in rows}) != 30:
        raise ValueError("Every prompt must have a unique ID.")
    for row in rows:
        messages = row["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError("Each prompt must contain one user message.")
        if digest(messages[0]["content"].encode()) != row["promptSha256"]:
            raise ValueError("Prompt text does not match its recorded checksum.")
    return rows


def prepare(args):
    rows = load_inputs(args.inputs)
    # Preparation tokenizes text. Model weights remain unloaded.
    from transformers import AutoTokenizer
    from qualify_hspec import require_snapshot, TARGET_ID
    target, provenance = require_snapshot(Path(args.hf_home), "target", TARGET_ID, TARGET_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True, use_fast=False)
    generation = json.loads((target / "generation_config.json").read_text())
    stops = generation["eos_token_id"]
    stops = stops if isinstance(stops, list) else [stops]
    warmup_text = ["What is 6 times 7? Give only the number.",
                   "Write a Python function that returns the square of its argument.",
                   "Explain in one sentence why leaves change color."]
    warmups = [{"id": "warmup-%d" % (i + 1), "benchmark": "warmup",
                "messages": [{"role": "user", "content": text}]}
               for i, text in enumerate(warmup_text)]
    for row in rows + warmups:
        row["input_token_ids"] = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=True, return_dict=False)
        check_ids(row["input_token_ids"])
    max_input = max(len(row["input_token_ids"]) for row in rows + warmups)
    context = math.ceil((max_input + 2048 + 32) / 256) * 256
    if context > 131072:
        raise ValueError("The longest input exceeds the target's context capacity.")
    save_json(args.output, {"schema": 1, "input_sha256": INPUT_SHA256,
                           "target_revision": TARGET_REVISION, "target_provenance": provenance,
                           "max_new_tokens": 2048, "temperature": 0, "stop_token_ids": stops,
                           "context_length": context, "max_input_tokens": max_input,
                           "prompts": rows, "warmups": warmups})
    print("Prepared 30 prompts and three warmups. Longest input: %d tokens. Context: %d tokens."
          % (max_input, context), flush=True)


def check_ids(ids):
    if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("The server must return original nonnegative integer token IDs.")


def percentile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    index = (len(values) - 1) * q
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def distribution(values):
    values = [value for value in values if value is not None]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": percentile(values, .5), "p95": percentile(values, .95)}


def sse_events(response, clock=time.perf_counter_ns):
    """Timestamp each complete SSE event before decoding its JSON payload."""
    parts = []
    while True:
        line = response.readline()
        if not line:
            raise ValueError("The stream ended before its [DONE] event.")
        line = line.rstrip(b"\r\n")
        if not line:
            if parts:
                received_ns = clock()
                data = b"\n".join(parts)
                parts = []
                if data == b"[DONE]":
                    yield received_ns, None
                    return
                yield received_ns, json.loads(data)
        elif line.startswith(b"data:"):
            parts.append(line[5:].lstrip(b" "))


class ResponseTrace:
    def __init__(self, backend, submitted_ns, stop_ids, sglang_incremental=False):
        self.backend = backend
        self.submitted_ns = submitted_ns
        self.stop_ids = set(stop_ids)
        self.incremental = sglang_incremental
        self.ids = []
        self.text = ""
        self.content_ns = []
        self.events = []
        self.finish_reason = None
        self.stop_reason = None
        self.completion_ns = None
        self.usage = None
        self.server_metadata = None

    def accept(self, received_ns, payload):
        if received_ns < self.submitted_ns or (self.events and received_ns < self.events[-1]["received_ns"]):
            raise ValueError("Stream timestamps must increase on the same monotonic clock.")
        if payload is None:
            self.completion_ns = received_ns
            return
        if "error" in payload:
            raise ValueError("The server returned an error event: %s" % payload["error"])
        if self.backend == "sglang":
            ids = payload.get("output_ids")
            check_ids(ids)
            text = payload.get("text") or ""
            metadata = payload.get("meta_info", {})
            self.server_metadata = metadata.copy()
            if self.incremental:
                delta_ids, delta_text = ids, text
            else:
                if ids[:len(self.ids)] != self.ids or not text.startswith(self.text):
                    raise ValueError("The cumulative SGLang response rewrote earlier output.")
                delta_ids, delta_text = ids[len(self.ids):], text[len(self.text):]
            reason = metadata.get("finish_reason")
            if reason:
                self.finish_reason = "length" if reason["type"] == "length" else "stop"
                self.stop_reason = reason.get("matched")
            self.usage = metadata.get("completion_tokens", self.usage)
        else:
            if payload.get("usage"):
                self.usage = payload["usage"].get("completion_tokens")
            choices = payload.get("choices", [])
            if not choices:
                self.events.append({"received_ns": received_ns, "metadata": payload})
                return
            if len(choices) != 1 or choices[0]["index"] != 0:
                raise ValueError("The trial requires one completion per request.")
            choice = choices[0]
            delta_ids, delta_text = choice.get("token_ids"), choice.get("text") or ""
            check_ids(delta_ids)
            if choice.get("finish_reason"):
                self.finish_reason, self.stop_reason = choice["finish_reason"], choice.get("stop_reason")
        self.ids.extend(delta_ids)
        self.text += delta_text
        if delta_text:
            self.content_ns.append(received_ns)
        self.events.append({"received_ns": received_ns, "text": delta_text,
                            "token_ids": delta_ids, "finish_reason": self.finish_reason,
                            "stop_reason": self.stop_reason, "completion_tokens": self.usage,
                            **({"meta_info": self.server_metadata} if self.backend == "sglang" else {})})

    def result(self):
        if self.completion_ns is None or self.finish_reason not in {"stop", "length"}:
            raise ValueError("A successful record requires a terminal reason and [DONE].")
        returned = self.ids[:]
        stop_positions = [i for i, token in enumerate(returned) if token in self.stop_ids]
        if stop_positions:
            if stop_positions != [len(returned) - 1]:
                raise ValueError("The engine returned tokens after a canonical stop token.")
            returned.pop()
        latency = (self.completion_ns - self.submitted_ns) / 1e9
        ttft = (self.content_ns[0] - self.submitted_ns) / 1e9 if self.content_ns else None
        gaps = [(b - a) / 1e9 for a, b in zip(self.content_ns, self.content_ns[1:])]
        count = len(returned)
        return {"submitted_ns": self.submitted_ns, "completed_ns": self.completion_ns,
                "first_content_ns": self.content_ns[0] if self.content_ns else None,
                "native_output_token_ids": self.ids, "output_token_ids": returned,
                "output_text": self.text, "output_tokens": count,
                "finish_reason": self.finish_reason, "native_stop_reason": self.stop_reason,
                "provider_completion_tokens": self.usage, "server_metadata": self.server_metadata,
                "events": self.events,
                "end_to_end_seconds": latency, "ttft_seconds": ttft,
                "tpot_seconds": (latency - ttft) / (count - 1) if count > 1 and ttft is not None else None,
                "output_rate": count / latency if latency > 0 else None,
                "inter_chunk_seconds": gaps, "max_inter_chunk_seconds": max(gaps) if gaps else None}


def request_payload(backend, row, bundle, model, request_id):
    common = {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
              "stop_token_ids": bundle["stop_token_ids"], "ignore_eos": False,
              "skip_special_tokens": True}
    if backend == "sglang":
        return "/generate", {"rid": request_id, "input_ids": row["input_token_ids"], "stream": True,
                             "sampling_params": {**common, "max_new_tokens": 2048}}
    return "/v1/completions", {**common, "model": model, "prompt": row["input_token_ids"],
                              "max_tokens": 2048, "seed": 0, "n": 1, "stream": True,
                              "stream_options": {"include_usage": True},
                              "return_token_ids": True, "request_id": request_id}


def record_request(args, row, bundle, backend):
    request_id = "hpml-" + uuid.uuid4().hex
    endpoint, payload = request_payload(backend, row, bundle, args.model, request_id)
    request = urllib.request.Request(args.url.rstrip("/") + endpoint,
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    submitted_ns = time.perf_counter_ns()
    live = getattr(args, "live_writer", None)
    progress = {"configuration": args.configuration, "prompt_id": row["id"],
                "benchmark": row["benchmark"], "repetition": getattr(args, "repetition", 0),
                "request_id": request_id} if live is not None else None
    if live is not None:
        live.put({**progress, "event": "submitted", "unix_ns": time.time_ns()})
    trace = ResponseTrace(backend, submitted_ns, bundle["stop_token_ids"], args.sglang_incremental)
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        if "text/event-stream" not in response.headers.get("Content-Type", ""):
            raise ValueError("The endpoint must return a streaming SSE response.")
        for received_ns, event in sse_events(response):
            if received_ns - submitted_ns > args.timeout * 1e9:
                raise TimeoutError("The request exceeded the complete-response timeout.")
            trace.accept(received_ns, event)
    result = trace.result()
    if len(result["native_output_token_ids"]) > 2048:
        raise ValueError("The engine exceeded the 2,048-token generation limit.")
    if live is not None:
        live.put({**progress, "event": "completed", "unix_ns": time.time_ns(),
                  "output_tokens": result["output_tokens"],
                  "ttft_seconds": result["ttft_seconds"],
                  "end_to_end_seconds": result["end_to_end_seconds"]})
    return {"prompt_id": row["id"], "benchmark": row["benchmark"],
            "request_id": request_id, "input_token_ids": row["input_token_ids"], **result}


def block_metrics(records, elapsed):
    if elapsed <= 0:
        raise ValueError("The measured interval must be positive.")
    return {"output_tokens": sum(row["output_tokens"] for row in records),
            "elapsed_seconds": elapsed,
            "otps": sum(row["output_tokens"] for row in records) / elapsed,
            **{key: distribution([row[key] for row in records]) for key in (
                "end_to_end_seconds", "ttft_seconds", "tpot_seconds", "output_rate", "max_inter_chunk_seconds")},
            "inter_chunk_seconds": distribution([gap for row in records for gap in row["inter_chunk_seconds"]])}


def record(args):
    from download_checkpoints import require_compute_allocation
    require_compute_allocation()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Run the client on the server's compute node using its loopback address.")
    bundle_bytes = Path(args.prepared).read_bytes()
    bundle = json.loads(bundle_bytes)
    if bundle["input_sha256"] != INPUT_SHA256 or bundle["target_revision"] != TARGET_REVISION:
        raise ValueError("Prepare inputs using the fixed prompt file and target revision.")
    if len(bundle["prompts"]) != 30 or len(bundle["warmups"]) != 3:
        raise ValueError("The trial requires 30 timed inputs and three warmups.")
    if Counter(row["benchmark"] for row in bundle["prompts"]) != Counter({b: 3 for b in BENCHMARKS}):
        raise ValueError("The prepared file must retain three inputs per benchmark.")
    for row in bundle["prompts"]:
        check_ids(row["input_token_ids"])
        if len(row["messages"]) != 1 or digest(row["messages"][0]["content"].encode()) != row["promptSha256"]:
            raise ValueError("Prepared messages differ from the selected prompts.")
    server_bytes = Path(args.server_record).read_bytes()
    server = json.loads(server_bytes)
    for key, expected in (("configuration", args.configuration), ("slurm_job_id", os.environ["SLURM_JOB_ID"]),
                          ("hostname", socket.gethostname())):
        if server[key] != expected:
            raise ValueError("The server launch record differs in " + key)
    device_record = subprocess.check_output(["nvidia-smi", "--id=" + args.gpu_uuid,
        "--query-gpu=uuid,name,driver_version", "--format=csv,noheader,nounits"], text=True).strip()
    if "L40S" not in device_record or len(device_record.splitlines()) != 1:
        raise ValueError("Select exactly one assigned L40S GPU UUID.")
    backend, reference = CONFIGURATIONS[args.configuration]
    output = Path(args.output)
    os.umask(0o077)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "run.json", {"configuration": args.configuration, "reference": reference,
              "backend": backend, "server_url": args.url, "model": args.model,
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
              "gpu_uuid": args.gpu_uuid, "gpu_record": device_record,
              "server_record_sha256": digest(server_bytes), "server_settings": server,
              "prepared_sha256": digest(bundle_bytes), "target_revision": TARGET_REVISION,
              "input_sha256": INPUT_SHA256, "clock": "time.perf_counter_ns",
              "clock_info": vars(time.get_clock_info("perf_counter")),
              "clock_anchor": {"unix_ns": time.time_ns(), "monotonic_ns": time.perf_counter_ns()},
              "recorder_sha256": digest(Path(__file__).read_bytes()),
              "sglang_incremental": args.sglang_incremental,
              "repetitions": 3, "temperature": 0, "max_new_tokens": 2048,
              "prompt_display": "Safari displays each prepared prompt. An asynchronous writer saves submission and completion events."})
    with LiveWriter(output / "live.jsonl") as live:
        args.live_writer = live
        try:
            measure(args, bundle, backend, output)
        finally:
            del args.live_writer


def measure(args, bundle, backend, output):
    print("Warming up with three separate prompts.", flush=True)
    warmups = []
    args.repetition = 0
    for row in bundle["warmups"]:
        warmups.append(record_request(args, row, bundle, backend))
        print("Completed %s." % row["id"], flush=True)
    args.live_writer.drain()
    save_json(output / "warmup.json", warmups)
    native_before = None
    if args.configuration in {"peagle", "dspark"}:
        from trial_native_stats import snapshot, delta
        print("Reading settled native acceptance counters after warmup.", flush=True)
        native_before = snapshot(args.url, gpu_uuid=args.gpu_uuid)
        if native_before["observed_completions"] is None:
            raise ValueError("The native metrics endpoint must expose the completed-request counter.")
        save_json(output / "native-after-warmup.json", native_before)
    names = list(dict.fromkeys(row["benchmark"] for row in bundle["prompts"]))
    blocks = []
    for repetition in range(1, 4):
        args.repetition = repetition
        for benchmark in names:
            rows = [row for row in bundle["prompts"] if row["benchmark"] == benchmark]
            print("Repetition %d/3: %s, three prompts. Capturing streaming events." % (repetition, benchmark), flush=True)
            records = []
            for row in rows:
                records.append(record_request(args, row, bundle, backend))
            started, ended = records[0]["submitted_ns"], records[-1]["completed_ns"]
            args.live_writer.drain()
            result = {"benchmark": benchmark, "repetition": repetition,
                      "started_ns": started, "ended_ns": ended,
                      "prompt_display_seconds_in_interval": 0,
                      "metrics": block_metrics(records, (ended - started) / 1e9), "requests": records}
            blocks.append(result)
            save_json(output / ("%02d-%s.json" % (repetition, benchmark)), result)
            if native_before is not None:
                native_after = snapshot(args.url,
                    expected_completions=native_before["observed_completions"] + len(records),
                    gpu_uuid=args.gpu_uuid)
                save_json(output / ("native-%02d-%s.json" % (repetition, benchmark)),
                          {"benchmark": benchmark, "repetition": repetition,
                           "before": native_before, "after": native_after,
                           "metrics": delta(native_before, native_after)})
                native_before = native_after
            for item in records:
                ttft = "unavailable" if item["ttft_seconds"] is None else "%.1f ms" % (item["ttft_seconds"] * 1000)
                print("  %s: %d tokens; TTFT %s; complete %.3f s."
                      % (item["prompt_id"], item["output_tokens"], ttft, item["end_to_end_seconds"]), flush=True)
            print("  OTPS: %.3f" % result["metrics"]["otps"], flush=True)
    summary = {}
    for name in names:
        selected = [block for block in blocks if block["benchmark"] == name]
        values = [block["metrics"]["otps"] for block in selected]
        estimates = {"otps": values}
        for metric, statistic in (("end_to_end_seconds", "median"), ("end_to_end_seconds", "p95"),
                                   ("ttft_seconds", "median"), ("ttft_seconds", "p95"),
                                   ("tpot_seconds", "mean"), ("output_rate", "mean"),
                                   ("inter_chunk_seconds", "p95"), ("max_inter_chunk_seconds", "p95")):
            estimates[metric + "_" + statistic] = [block["metrics"][metric][statistic] for block in selected]
        summary[name] = {"repetitions": [block["metrics"] for block in selected], "across_repetitions": {}}
        for metric, observations in estimates.items():
            eligible = [value for value in observations if value is not None]
            summary[name]["across_repetitions"][metric] = {
                "n": len(eligible), "mean": statistics.mean(eligible) if eligible else None,
                "sample_sd": statistics.stdev(eligible) if len(eligible) > 1 else None}
    save_json(output / "summary.json", summary)
    print("Completed three repetitions. Observations are saved in %s." % output, flush=True)


def show_prompt(configuration, row, phase):
    started = time.perf_counter_ns()
    print("\n%s | %s | %s | %s" % (phase, configuration, row["benchmark"], row["id"]), flush=True)
    print("Submitting this prompt:", flush=True)
    print(row["messages"][0]["content"], flush=True)
    print("The request begins after this line.", flush=True)
    return (time.perf_counter_ns() - started) / 1e9


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Check the 30 prompts without loading a model.")
    validate.add_argument("inputs", type=Path)
    prepare_parser = sub.add_parser("prepare", help="Tokenize the prompts using the existing target snapshot.")
    prepare_parser.add_argument("--inputs", type=Path, required=True)
    prepare_parser.add_argument("--hf-home", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("record", help="Run the timed trial against an already visible foreground server.")
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    run.add_argument("--url", default="http://127.0.0.1:31000")
    run.add_argument("--model", default="hpml-target")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--server-record", type=Path, required=True)
    run.add_argument("--gpu-uuid", required=True, help="Assigned physical L40S UUID recorded before launch.")
    run.add_argument("--timeout", type=float, default=900)
    run.add_argument("--sglang-incremental", action="store_true")
    args = parser.parse_args()
    if args.command == "validate":
        rows = load_inputs(args.inputs)
        print("Verified 30 unchanged, single-turn prompts: three per benchmark.")
        for name, count in Counter(row["benchmark"] for row in rows).items():
            print("  %s: %d" % (name, count))
    elif args.command == "prepare":
        prepare(args)
    else:
        record(args)


if __name__ == "__main__":
    main()
