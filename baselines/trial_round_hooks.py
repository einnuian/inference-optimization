"""Capture verification rounds during explicitly enabled diagnostic runs."""

from __future__ import annotations

import functools
import atexit
import json
import operator
import os
from pathlib import Path
import re
import queue
import socket
import threading
import time
import uuid

_LOCK = threading.RLock()
_ROUNDS = {}
_HANDLES = {}
_SG_PENDING = threading.local()
_FLUSH = object()


class _DiagnosticWriter:
    """Write immutable queued JSON lines through one persistent process-local handle."""
    def __init__(self, path):
        self.path = path
        self.queue = queue.Queue()
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="hpml-diagnostic-writer", daemon=True)
        self.thread.start()

    def _run(self):
        handle = None
        while True:
            item = self.queue.get()
            if item is None:
                try:
                    if handle is not None:
                        handle.close()
                except BaseException as error:
                    self.error = self.error or error
                finally:
                    self.queue.task_done()
                return
            try:
                if self.error is not None:
                    continue
                if handle is None:
                    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
                    handle = os.fdopen(os.open(self.path, flags, 0o600), "a", encoding="utf-8")
                if item is not _FLUSH:
                    handle.write(item)
                handle.flush()
                if item is _FLUSH:
                    os.fsync(handle.fileno())
            except BaseException as error:
                self.error = self.error or error
            finally:
                self.queue.task_done()

    def check(self):
        if self.error is not None:
            raise RuntimeError("The diagnostic evidence writer failed: " + str(self.path)) from self.error

    def put(self, line):
        self.check()
        if self.closed:
            raise RuntimeError("The diagnostic evidence writer is closed.")
        self.queue.put_nowait(line)

    def drain(self):
        if not self.closed:
            self.queue.put_nowait(_FLUSH)
            self.queue.join()
        self.check()

    def close(self):
        if not self.closed:
            try:
                self.drain()
            finally:
                self.closed = True
                self.queue.put_nowait(None)
                self.thread.join()
        self.check()


def _write_event(path, record):
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    key = (os.getpid(), str(path))
    with _LOCK:
        writer = _HANDLES.get(key)
        if writer is None:
            writer = _DiagnosticWriter(path)
            _HANDLES[key] = writer
        writer.put(line)


def drain_events():
    """Finish queued evidence writes at a request boundary outside recorded intervals."""
    for key, writer in list(_HANDLES.items()):
        if key[0] == os.getpid():
            writer.drain()


def _close_events():
    for key, writer in list(_HANDLES.items()):
        if key[0] == os.getpid():
            writer.close()


atexit.register(_close_events)


def canonical_request_id(request_id):
    """Reverse the pinned vLLM completion API's one-prompt, n=1 ID decoration."""
    value = str(request_id)
    match = re.fullmatch(r"cmpl-(hpml-[0-9a-f]{32})-0", value)
    return match.group(1) if match else value


def _integer(value, label):
    if isinstance(value, bool):
        raise ValueError(label + " must be an integer count.")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(label + " must be an integer count.") from error
    if result < 0:
        raise ValueError(label + " must be nonnegative.")
    return result


def _token_ids(value, label, allow_placeholders=False):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(label + " must contain the original token IDs.")
    if not allow_placeholders:
        return [_integer(token, label) for token in value]
    tokens = []
    for token in value:
        if isinstance(token, bool):
            raise ValueError(label + " must contain integer token IDs or scheduler placeholders.")
        try:
            tokens.append(operator.index(token))
        except TypeError as error:
            raise ValueError(label + " must contain integer token IDs or scheduler placeholders.") from error
    return tokens


def _settings(backend):
    selected = os.environ.get("HPML_DIAGNOSTIC_DIR")
    if not selected:
        return None
    directory = Path(selected).resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("HPML_DIAGNOSTIC_DIR must identify an existing directory.")
    configuration = os.environ.get("HPML_CONFIGURATION")
    allowed = {"sglang": {"ar-sglang", "eagle3", "dflash"},
               "vllm": {"ar-vllm-v1", "ar-vllm-v2", "peagle", "dspark"}}
    if configuration not in allowed[backend]:
        raise ValueError("HPML_CONFIGURATION must identify the diagnostic engine configuration.")
    return {"directory": directory, "configuration": configuration, "backend": backend}


def _source(method):
    code = getattr(method, "__code__", None)
    return {"callable": method.__module__ + "." + method.__qualname__,
            "file": code.co_filename if code is not None else None,
            "line": code.co_firstlineno if code is not None else None}


def _announce(cls, settings, source):
    print("HPML diagnostic round capture is active for %s.%s. Source: %s:%s."
          % (cls.__module__, cls.__qualname__, source["file"], source["line"]), flush=True)


def _emit(settings, source, row):
    native_id = row["request_id"]
    row = {**row, "request_id": canonical_request_id(native_id), "native_request_id": native_id}
    pid = os.getpid()
    key = (pid, settings["backend"], settings["configuration"], row["request_id"])
    with _LOCK:
        _ROUNDS[key] = _ROUNDS.get(key, 0) + 1
        event = {"schema": 1, "scope": "separate_diagnostic", "pid": pid,
                 "backend": settings["backend"], "configuration": settings["configuration"],
                 "clock": "time.perf_counter_ns", "observed_ns": time.perf_counter_ns(),
                 "clock_implementation": time.get_clock_info("perf_counter").implementation,
                 "hostname": socket.gethostname(), "gpu_uuid": os.environ.get("HPML_GPU_UUID"),
                 "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                 "round": _ROUNDS[key], "source": source, **row}
        name = "%s-rounds-%d.jsonl" % (settings["backend"], pid)
        _write_event(settings["directory"] / name, event)


def _counts(proposed, accepted, committed):
    proposed = _integer(proposed, "Proposed draft tokens")
    accepted = _integer(accepted, "Accepted draft tokens")
    if accepted > proposed or accepted > len(committed):
        raise ValueError("The diagnostic acceptance count exceeds its proposal or committed sequence.")
    return proposed, accepted, len(committed) - accepted


def install_sglang_processor(cls):
    """Wrap the pinned SGLang speculative-v2 result processor when diagnostics are enabled."""
    settings = _settings("sglang")
    if settings is None:
        return False
    if getattr(cls, "_hpml_sglang_round_hook", False):
        return True
    original = cls._resolve_spec_v2_tokens
    source = _source(original)
    decode_original = getattr(cls, "process_batch_result_decode", None)

    @functools.wraps(original)
    def resolve(self, result, batch, *args, **kwargs):
        active = []
        for index, request in enumerate(batch.reqs):
            if request.is_retracted or request.finished():
                continue
            previous = _token_ids(request.output_ids, "Previous SGLang output")
            active.append((index, request, previous,
                           _integer(request.spec_verify_ct, "Previous verification count")))
        started = time.perf_counter_ns()
        predictions = original(self, result, batch, *args, **kwargs)
        ended = time.perf_counter_ns()
        for index, request, previous, prior_rounds in active:
            current_rounds = _integer(request.spec_verify_ct, "Current verification count")
            if current_rounds == prior_rounds:
                continue
            if current_rounds != prior_rounds + 1:
                raise ValueError("The SGLang processor must increment the verification count once per captured round.")
            committed = _token_ids(predictions[index], "SGLang committed tokens")
            candidate_count = _integer(result.speculative_num_draft_tokens, "SGLang candidate tokens")
            if candidate_count < 1:
                raise ValueError("The SGLang candidate count must include its target token.")
            proposed, accepted, retained = _counts(
                candidate_count - 1, result.num_correct_drafts_per_req_cpu[index], committed)
            linear = settings["configuration"] == "dflash"
            if settings["configuration"] not in {"eagle3", "dflash"}:
                raise ValueError("A SGLang verification round requires an EAGLE-3 or DFlash diagnostic configuration.")
            row = {
                "request_id": str(request.rid), "native_verification_count": current_rounds,
                "proposed_draft_tokens": proposed, "accepted_draft_tokens": accepted,
                "retained_target_tokens": retained, "committed_tokens": len(committed),
                "committed_token_ids": committed, "previous_output_count": len(previous),
                "previous_output_tail": previous[-8:],
                "proposed_depth": proposed if linear else None,
                "accepted_prefix_depth": accepted if linear else None,
                "proposal_structure": "linear" if linear else "tree",
                "proposed_count_basis": "linear_tokens" if linear else "candidate_nodes",
                "depth_metrics_available": linear,
                "delivery_observation": "The client records delivered output after the engine applies stopping rules.",
                "cpu_call_started_ns": started, "cpu_call_finished_ns": ended,
                "state_update_completed": False,
            }
            pending = getattr(_SG_PENDING, "rows", None)
            if pending is None:
                _emit(settings, source, row)
            else:
                pending.append((row, request, previous))
        return predictions

    if callable(decode_original):
        @functools.wraps(decode_original)
        def process_decode(self, batch, result, *args, **kwargs):
            if getattr(_SG_PENDING, "rows", None) is not None:
                raise RuntimeError("SGLang result processing has overlapping diagnostic contexts.")
            pending = []
            _SG_PENDING.rows = pending
            try:
                returned = decode_original(self, batch, result, *args, **kwargs)
                ended = time.perf_counter_ns()
                for row, request, previous in pending:
                    current = _token_ids(request.output_ids, "SGLang output after state update")
                    if current[:len(previous)] != previous or current[len(previous):] != row["committed_token_ids"]:
                        raise ValueError("SGLang state update must append the captured committed token sequence.")
                    row.update(state_update_completed=True,
                        state_update_completed_ns=ended,
                        state_update_source=_source(decode_original),
                        state_update_boundary="Return from process_batch_result_decode after output append, stopping checks, streaming, and KV free-group completion.",
                        native_postupdate_token_ids=current[len(previous):],
                        native_finished=request.finished(),
                        native_finished_len=getattr(request, "finished_len", None))
                    _emit(settings, source, row)
                if any(request.finished() for _, request, _ in pending):
                    drain_events()
                return returned
            finally:
                _SG_PENDING.rows = None

        cls.process_batch_result_decode = process_decode

    cls._resolve_spec_v2_tokens = resolve
    cls._hpml_sglang_round_hook = True
    _announce(cls, settings, source)
    return True


def _request_finished(request):
    for name in ("is_finished", "finished"):
        state = getattr(request, name, None)
        if state is not None:
            return bool(state() if callable(state) else state)
    raise AttributeError("The vLLM request must expose its finished state.")


def install_vllm_scheduler(cls):
    """Wrap the pinned vLLM scheduler to preserve proposals and output before stopping trims."""
    settings = _settings("vllm")
    if settings is None:
        return False
    if getattr(cls, "_hpml_vllm_round_hook", False):
        return True
    original = cls.update_from_output
    source = _source(original)
    schedule_original = getattr(cls, "schedule", None)

    if callable(schedule_original):
        @functools.wraps(schedule_original)
        def schedule(self, *args, **kwargs):
            scheduled = schedule_original(self, *args, **kwargs)
            if not hasattr(scheduled, "hpml_diagnostic_step_id"):
                raise RuntimeError("The isolated SchedulerOutput requires its declared diagnostic step field.")
            scheduled.hpml_diagnostic_step_id = uuid.uuid4().hex
            return scheduled

        cls.schedule = schedule

    @functools.wraps(original)
    def update(self, scheduler_output, model_runner_output, *args, **kwargs):
        active = []
        invalid_counts = scheduler_output.num_invalid_spec_tokens or {}
        retained_per_step = _integer(self.num_sampled_tokens_per_step, "Target samples per step")
        for request_id in scheduler_output.num_scheduled_tokens:
            request = self.requests.get(request_id)
            if request is None or _request_finished(request) or request.async_tokens_to_discard != 0:
                continue
            scheduled = scheduler_output.scheduled_spec_decode_tokens.get(request_id)
            if scheduled is None or len(scheduled) == 0:
                continue
            index = model_runner_output.req_id_to_index[request_id]
            sampled = model_runner_output.sampled_token_ids
            generated = (_token_ids(sampled[index], "vLLM generated tokens")
                         if sampled is not None and len(sampled) else [])
            if not generated and retained_per_step != 0:
                continue
            proposed_ids = _token_ids(scheduled, "vLLM scheduled proposals", allow_placeholders=True)
            invalid = _integer(invalid_counts.get(request_id, 0), "Invalid speculative tokens")
            if invalid > len(proposed_ids):
                raise ValueError("The invalid speculative count exceeds the scheduled proposal count.")
            proposed, accepted, retained = _counts(
                len(proposed_ids) - invalid, max(len(generated) - retained_per_step, 0), generated)
            previous = _token_ids(request._output_token_ids, "Previous vLLM output")
            active.append((request_id, request, previous, proposed_ids, invalid,
                           proposed, accepted, retained, generated))
        started = time.perf_counter_ns()
        result = original(self, scheduler_output, model_runner_output, *args, **kwargs)
        ended = time.perf_counter_ns()
        for request_id, request, previous, proposed_ids, invalid, proposed, accepted, retained, generated in active:
            current = _token_ids(request._output_token_ids, "Current vLLM output")
            if current[:len(previous)] != previous:
                raise ValueError("The vLLM scheduler changed output from an earlier verification round.")
            native_delivered = current[len(previous):]
            if generated[:len(native_delivered)] != native_delivered:
                raise ValueError("The post-trim output must preserve a prefix of the sampled token sequence.")
            _emit(settings, source, {
                "request_id": str(request_id), "proposed_draft_tokens": proposed,
                "scheduler_step_id": getattr(scheduler_output, "hpml_diagnostic_step_id", None),
                "accepted_draft_tokens": accepted, "retained_target_tokens": retained,
                "committed_tokens": len(generated), "committed_token_ids": generated,
                "scheduled_proposal_token_ids": proposed_ids,
                "scheduled_proposal_contains_placeholders": any(token < 0 for token in proposed_ids),
                "scheduled_proposal_count": len(proposed_ids), "invalid_proposal_count": invalid,
                "previous_output_count": len(previous), "previous_output_tail": previous[-8:],
                "native_posttrim_token_ids": native_delivered,
                "native_posttrim_tokens": len(native_delivered),
                "native_posttrim_empty": not native_delivered,
                "proposed_depth": proposed, "accepted_prefix_depth": accepted,
                "proposal_structure": "linear", "proposed_count_basis": "valid_linear_tokens",
                "depth_metrics_available": True,
                "delivery_observation": "The native post-trim sequence preserves any engine stop tokens for later reconciliation.",
                "cpu_call_started_ns": started, "cpu_call_finished_ns": ended,
                "state_update_completed": True, "state_update_completed_ns": ended,
                "state_update_source": source,
                "state_update_boundary": "Return from update_from_output after sampled output, stopping checks, and request state updates.",
            })
        if any(_request_finished(request) for _, request, *_ in active):
            drain_events()
        return result

    cls.update_from_output = update
    cls._hpml_vllm_round_hook = True
    _announce(cls, settings, source)
    return True
