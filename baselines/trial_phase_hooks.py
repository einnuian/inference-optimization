"""Collect raw CUDA phase observations during a separate diagnostic execution.

Set HPML_DIAGNOSTIC_DIR to an absolute output directory before installing hooks.
HPML_CONFIGURATION identifies the configuration in each record. SLURM_JOB_ID and
HPML_GPU_UUID retain allocation context when supplied. The primary server uses its
ordinary, unpatched entry point. Imports of this helper load torch lazily.

Each phase describes the installed function or inline boundary. Nested phases have
inclusive durations. CUDA events describe their recorded stream. The outer boundary
synchronizes recorded events and the device before collecting process memory peaks.
Request memory peaks include allocations held at the first observed worker call.
"""

from collections.abc import Mapping
import functools
import importlib
import itertools
import json
import math
import numbers
import os
from pathlib import Path
import socket
import threading
import time
import uuid
import weakref

from trial_round_hooks import canonical_request_id, _write_event, drain_events


_TORCH = None
_LOCAL = threading.local()
_MEMORY_LOCK = threading.RLock()
_WRITE_LOCK = threading.Lock()
_MEMORY_SCOPES = {}
_ACTIVE_DEVICES = {}
_OBSERVATIONS = itertools.count(1)
_PROCESS_TOKEN = uuid.uuid4().hex


def _enabled():
    return bool(os.environ.get("HPML_DIAGNOSTIC_DIR"))


def _torch():
    global _TORCH
    if _TORCH is None:
        _TORCH = importlib.import_module("torch")
    if not _TORCH.cuda.is_available():
        raise RuntimeError("CUDA phase diagnostics require an available CUDA device.")
    return _TORCH


def _active():
    return getattr(_LOCAL, "active", None)


def _directory():
    path = Path(os.environ["HPML_DIAGNOSTIC_DIR"]).expanduser()
    if not path.is_absolute():
        raise ValueError("HPML_DIAGNOSTIC_DIR must identify an absolute directory.")
    return path


def _identifiers(request_ids):
    if request_ids is None:
        return []
    if isinstance(request_ids, str):
        return [request_ids]
    return [str(value) for value in request_ids if value is not None]


def _token_counts(tokens):
    if tokens is None:
        return None
    if not isinstance(tokens, Mapping):
        raise ValueError("Diagnostic token counts must be a host-side mapping.")
    result = {}
    for key, value in tokens.items():
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
            raise ValueError("Diagnostic token counts must be nonnegative host integers.")
        result[str(key)] = int(value)
    return result


def _reset_memory(device, request_ids):
    cuda = _torch().cuda
    cuda.synchronize(device)
    cuda.reset_peak_memory_stats(device)
    scope = {"reset_id": uuid.uuid4().hex, "request_ids": list(request_ids),
             "request_identity_available": bool(request_ids),
             "reset_monotonic_ns": time.perf_counter_ns(), "reset_unix_ns": time.time_ns()}
    _MEMORY_SCOPES[device] = scope
    return scope


def _memory(device, synchronized=False):
    cuda = _torch().cuda
    return {"pid": os.getpid(), "device_index": device,
            "memory_allocated_bytes": int(cuda.memory_allocated(device)),
            "memory_reserved_bytes": int(cuda.memory_reserved(device)),
            "max_memory_allocated_bytes": int(cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(cuda.max_memory_reserved(device)),
            "scope": dict(_MEMORY_SCOPES.get(device, {})),
            "device_synchronized_before_read": synchronized}


def reset_memory(request_ids=None):
    """Reset process peaks at an idle diagnostic boundary and retain their scope."""
    if not _enabled():
        return None
    if _active() is not None:
        raise RuntimeError("Reset memory before beginning the worker iteration.")
    device = int(_torch().cuda.current_device())
    with _MEMORY_LOCK:
        if device in _ACTIVE_DEVICES.values():
            raise RuntimeError("A diagnostic iteration still owns this device's memory scope.")
        _reset_memory(device, [canonical_request_id(value) for value in _identifiers(request_ids)])
        return _memory(device, synchronized=True)


def read_memory():
    """Read cumulative process peaks under the current recorded reset scope."""
    if not _enabled():
        return None
    context = _active()
    device = context["device_index"] if context is not None else int(_torch().cuda.current_device())
    return _memory(device)


def _event_start(name, device, sequence, parent, scope):
    if not isinstance(name, str) or not name:
        raise ValueError("Each diagnostic phase requires a name.")
    cuda = _torch().cuda
    stream = cuda.current_stream(device)
    event = cuda.Event(enable_timing=True)
    phase = {"name": name, "scope": scope, "sequence": sequence, "parent_sequence": parent,
             "stream_id": int(stream.cuda_stream), "device_index": device,
             "cpu_start_ns": time.perf_counter_ns(), "cpu_start_unix_ns": time.time_ns(),
             "_start": event, "_stream": stream}
    event.record(stream)
    return phase


def _event_end(phase):
    cuda = _torch().cuda
    if int(cuda.current_device()) != phase["device_index"]:
        raise RuntimeError("The CUDA device changed inside a diagnostic phase.")
    stream = cuda.current_stream(phase["device_index"])
    if int(stream.cuda_stream) != phase["stream_id"]:
        raise RuntimeError("The calling CUDA stream changed inside a diagnostic phase.")
    phase["_end"] = cuda.Event(enable_timing=True)
    phase["_end"].record(stream)
    phase["cpu_end_ns"] = time.perf_counter_ns()
    phase["cpu_end_unix_ns"] = time.time_ns()


def _resolved_phase(phase):
    seconds = float(phase["_start"].elapsed_time(phase["_end"])) / 1000
    if not math.isfinite(seconds) or seconds < 0:
        raise RuntimeError("CUDA events returned an invalid elapsed duration.")
    result = {key: value for key, value in phase.items() if not key.startswith("_")}
    result["gpu_elapsed_seconds"] = seconds
    result["cpu_elapsed_seconds"] = (phase["cpu_end_ns"] - phase["cpu_start_ns"]) / 1e9
    return result


def _metadata(context):
    return {"schema_version": 1, "diagnostic_only": True,
            "observation_id": context["observation_id"], "kind": context["kind"],
            "request_ids": context["request_ids"], "request_identity_available": bool(context["request_ids"]),
            "native_request_ids": context["native_request_ids"],
            "clock": "time.perf_counter_ns",
            "clock_implementation": time.get_clock_info("perf_counter").implementation,
            "instrumentation": "GPU synchronization, CPU event construction, and queued trace submission are included in diagnostic intervals. Persistent writer threads perform file writes. Nested intervals can overlap.",
            "verification_round_indices": context.get("verification_round_indices"),
            "scheduler_step_id": context.get("scheduler_step_id"),
            "produced_drafts": context.get("produced_drafts"),
            "verification_proposals": context.get("verification_proposals"),
            "tokens": context["tokens"], "pid": os.getpid(), "thread_id": threading.get_ident(),
            "hostname": socket.gethostname(), "configuration": os.environ.get("HPML_CONFIGURATION"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "gpu_uuid": os.environ.get("HPML_GPU_UUID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device_index": context["device_index"], "emitted_unix_ns": time.time_ns()}


def _emit(context, record):
    directory = context["directory"]
    path = directory / ("phase-%d-%s.jsonl" % (os.getpid(), _PROCESS_TOKEN))
    _write_event(path, record)


def _clear(context):
    if _active() is context:
        _LOCAL.active = None
    with _MEMORY_LOCK:
        _ACTIVE_DEVICES.pop(threading.get_ident(), None)


def _abort(error):
    context = _active()
    if context is None:
        return
    try:
        _emit(context, {**_metadata(context), "event": "diagnostic_capture_error",
                        "error_type": type(error).__name__, "error": str(error),
                        "open_phases": [phase["name"] for phase in context["stack"]]})
    finally:
        _clear(context)


def begin_round(request_ids, kind, phase="round", *, tokens=None):
    """Begin one raw worker observation and reset peaks for a newly observed request set."""
    if not _enabled():
        return None
    if _active() is not None:
        raise RuntimeError("Finish the current diagnostic worker iteration before beginning another.")
    native_ids = _identifiers(request_ids)
    ids = [canonical_request_id(value) for value in native_ids]
    counts = _token_counts(tokens)
    directory = _directory()
    device = int(_torch().cuda.current_device())
    context = {"observation_id": "%d-%d" % (os.getpid(), next(_OBSERVATIONS)),
               "kind": str(kind), "request_ids": ids, "native_request_ids": native_ids,
               "tokens": counts, "directory": directory,
               "device_index": device, "stack": [], "phases": []}
    with _MEMORY_LOCK:
        if device in _ACTIVE_DEVICES.values():
            raise RuntimeError("Concurrent worker observations would mix process memory scopes on this device.")
        previous = _MEMORY_SCOPES.get(device)
        if previous is None or set(previous["request_ids"]) != set(ids):
            drain_events()
            _reset_memory(device, ids)
        _ACTIVE_DEVICES[threading.get_ident()] = device
    _LOCAL.active = context
    try:
        context["outer"] = _event_start(phase, device, 0, None, "Inclusive installed worker boundary")
    except BaseException as error:
        _abort(error)
        raise
    return context["observation_id"]


def begin_phase(name, *, scope=None):
    """Begin a phase inside the active diagnostic worker observation."""
    context = _active()
    if context is None:
        if not _enabled():
            return None
        raise RuntimeError("Begin a diagnostic worker iteration before beginning a phase.")
    parent = context["stack"][-1]["sequence"] if context["stack"] else 0
    phase = _event_start(name, context["device_index"], len(context["phases"]) + 1, parent,
                         scope or "Explicit inline boundary")
    context["phases"].append(phase)
    context["stack"].append(phase)
    return phase["sequence"]


def end_phase(name):
    """Close the most recently opened phase and defer event synchronization."""
    context = _active()
    if context is None:
        if not _enabled():
            return None
        raise RuntimeError("A diagnostic phase has no active worker iteration.")
    if not context["stack"] or context["stack"][-1]["name"] != name:
        raise RuntimeError("Diagnostic phases must close in reverse opening order: " + str(name))
    _event_end(context["stack"][-1])
    context["stack"].pop()


def end_round():
    """Resolve all events at the outer boundary and emit one diagnostic record."""
    context = _active()
    if context is None:
        if not _enabled():
            return None
        raise RuntimeError("A diagnostic worker iteration has no active context.")
    try:
        if context["stack"]:
            raise RuntimeError("Close every diagnostic phase before finishing the worker iteration.")
        _event_end(context["outer"])
        phases = [context["outer"]] + context["phases"]
        for phase in phases:
            phase["_end"].synchronize()
        _torch().cuda.synchronize(context["device_index"])
        synchronized_ns = time.perf_counter_ns()
        record = {**_metadata(context), "event": "cuda_phase_observation",
                  "outer": _resolved_phase(context["outer"]),
                  "phases": [_resolved_phase(phase) for phase in context["phases"]],
                  "synchronized_cpu_end_ns": synchronized_ns,
                  "memory": _memory(context["device_index"], synchronized=True),
                  "timing_scope": "Inclusive CUDA event intervals on the recorded streams. Nested phase intervals can overlap."}
        _emit(context, record)
        return record
    except BaseException as error:
        _abort(error)
        raise
    finally:
        _clear(context)


def _tree_field(value, maximum=256):
    """Keep small raw values and explain each omitted tensor explicitly."""
    if hasattr(value, "numel") and hasattr(value, "detach"):
        count = int(value.numel())
        field = {
            "type": "tensor",
            "shape": [int(size) for size in value.shape],
            "dtype": str(value.dtype),
            "device": str(value.device),
            "value_count": count,
        }
        if count > maximum:
            field["omitted"] = "The tensor exceeds the raw capture limit."
        else:
            field["value"] = value.detach().cpu().tolist()
        return field
    if value is None or isinstance(value, (bool, str, numbers.Integral)):
        return {"value": int(value) if isinstance(value, numbers.Integral) and not isinstance(value, bool) else value}
    if isinstance(value, numbers.Real):
        number = float(value)
        return {"value": number} if math.isfinite(number) else {"omitted": "The value is nonfinite."}
    if isinstance(value, (list, tuple)):
        # Small Python sequences can contain a tensor for each request.
        if len(value) > maximum:
            return {"type": type(value).__name__, "length": len(value), "omitted": "The sequence exceeds the raw capture limit."}
        if all(item is None or isinstance(item, (bool, str, numbers.Integral)) for item in value):
            return {"value": [int(item) if isinstance(item, numbers.Integral) and not isinstance(item, bool) else item for item in value]}
        return {"type": type(value).__name__, "omitted": "The sequence requires a version-specific capture adapter."}
    return {"type": type(value).__name__, "omitted": "The value requires a version-specific capture adapter."}


def _named_raw_fields(subject, names):
    fields = {}
    for name in names:
        if isinstance(subject, Mapping):
            if name in subject:
                fields[name] = _tree_field(subject[name])
        elif subject is not None and hasattr(subject, name):
            fields[name] = _tree_field(getattr(subject, name))
    return fields


def _capture_eagle_tree(args, kwargs):
    """Copy pre-verification tree data during the diagnostic execution."""
    if os.environ.get("HPML_DIAGNOSTIC_EAGLE_TREE", "1") == "0":
        return None
    context = _active()
    if context is None:
        return None
    batch = args[1] if len(args) > 1 else None
    if batch is None:
        for name in ("batch", "forward_batch", "model_worker_batch"):
            if kwargs.get(name) is not None:
                batch = kwargs[name]
                break
    cpu_start = time.perf_counter_ns()
    names = ("draft_token", "positions", "retrieve_index", "retrieve_next_token", "retrieve_next_sibling", "draft_token_num")
    fields = _named_raw_fields(getattr(batch, "spec_info", None), names)
    capture_id = f"{context['observation_id']}:verify:{len(context['phases']) + 1}"
    _emit(context, {
        **_metadata(context),
        "event": "eagle_tree_before_verify",
        "capture_id": capture_id,
        "request_ids_from_verify_batch": _sglang_ids(batch) if batch is not None else [],
        "raw_tree": fields,
        "missing_tree_fields": [name for name in names if name not in fields],
        "cpu_capture_start_ns": cpu_start,
        "cpu_capture_end_ns": time.perf_counter_ns(),
        "scope": "Raw pre-verification fields. Tensor copies can synchronize CUDA during this diagnostic execution. Tree depth and position eligibility require analysis of these fields.",
    })
    return capture_id, batch


def _capture_eagle_result(capture, result):
    if capture is None:
        return
    context = _active()
    if context is None:
        raise RuntimeError("The EAGLE verification capture requires its active worker observation.")
    capture_id, batch = capture
    names = ("token_ids", "next_token_ids", "verified_id", "verified_token_ids", "accepted_token_ids", "accept_lens", "accept_length", "accept_lengths", "accept_length_cpu")
    cpu_start = time.perf_counter_ns()
    result_fields = _named_raw_fields(result, names)
    if isinstance(result, (tuple, list)):
        # Tuple positions retain the pinned implementation's ordering.
        result_fields["sequence_items"] = [
            {"index": index, "fields": _named_raw_fields(item, names), "raw": _tree_field(item)}
            for index, item in enumerate(result[:16])
        ]
        result_fields["sequence_length"] = len(result)
    elif hasattr(result, "numel"):
        result_fields["raw"] = _tree_field(result)
    _emit(context, {
        **_metadata(context),
        "event": "eagle_verification_result",
        "capture_id": capture_id,
        "result_type": type(result).__name__,
        "raw_result": result_fields,
        "spec_info_after_verify": _named_raw_fields(getattr(batch, "spec_info", None), names),
        "cpu_capture_start_ns": cpu_start,
        "cpu_capture_end_ns": time.perf_counter_ns(),
        "scope": "Raw returned fields and post-verification state. Tuple positions retain source ordering. Accepted-token semantics require the pinned implementation's definitions.",
    })


def _phase_call(function, name, scope, *args, **kwargs):
    # Initialization calls outside a worker observation retain their ordinary path.
    if _active() is None:
        return function(*args, **kwargs)
    tree_capture = _capture_eagle_tree(args, kwargs) if name == "target_verification" else None
    begin_phase(name, scope=scope)
    try:
        result = function(*args, **kwargs)
        end_phase(name)
        if tree_capture is not None:
            _capture_eagle_result(tree_capture, result)
        return result
    except BaseException as error:
        _abort(error)
        raise


def _flag(value, name):
    flag = getattr(value, name, False)
    return bool(flag() if callable(flag) else flag)


def _sglang_ids(batch):
    requests = getattr(batch, "reqs", None)
    if requests is not None:
        return _identifiers(getattr(request, "rid", getattr(request, "request_id", None)) for request in requests)
    return _identifiers(getattr(batch, "request_ids", None))


def install_sglang_worker(cls):
    """Wrap the supplied pinned SGLang class's forward_batch_generation method."""
    if not _enabled():
        return False
    original = getattr(cls, "forward_batch_generation", None)
    if not callable(original):
        raise ValueError("The supplied SGLang class has no forward_batch_generation method.")
    if getattr(original, "_hpml_phase_wrapper", False):
        return False

    @functools.wraps(original)
    def forward(self, *args, **kwargs):
        batch = args[0] if args else kwargs.get("batch", kwargs.get("model_worker_batch", kwargs.get("forward_batch")))
        mode = getattr(batch, "forward_mode", None)
        if not _enabled() or batch is None or _flag(mode, "is_idle"):
            return original(self, *args, **kwargs)
        if _active() is not None:
            return _phase_call(original, "nested_worker_forward", cls.__name__ + ".forward_batch_generation",
                               self, *args, **kwargs)
        kind = "prefill" if _flag(mode, "is_extend") or _flag(batch, "is_extend_in_batch") else "decode"
        input_ids = getattr(batch, "input_ids", None)
        tokens = {"batch_input_token_count": int(input_ids.numel())} if hasattr(input_ids, "numel") else None
        begin_round(_sglang_ids(batch), kind, phase="worker_iteration", tokens=tokens)
        try:
            if kind == "decode" and os.environ.get("HPML_CONFIGURATION") in {"eagle3", "dflash"}:
                counts = {}
                for request in batch.reqs:
                    if request.is_retracted or request.finished():
                        continue
                    count = request.spec_verify_ct
                    if isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 0:
                        raise ValueError("SGLang must expose a nonnegative native verification count.")
                    counts[canonical_request_id(request.rid)] = int(count) + 1
                _active()["verification_round_indices"] = counts
            result = original(self, *args, **kwargs)
            end_round()
            return result
        except BaseException as error:
            _abort(error)
            raise

    forward._hpml_phase_wrapper = True
    cls.forward_batch_generation = forward
    return True


def _install_phase_method(cls, method, phase):
    original = getattr(cls, method, None)
    if not callable(original) or getattr(original, "_hpml_phase_wrapper", False):
        return False

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        return _phase_call(original, phase, cls.__name__ + "." + method, self, *args, **kwargs)

    wrapped._hpml_phase_wrapper = True
    setattr(cls, method, wrapped)
    return True


def install_sglang_eagle_draft(cls):
    """Capture EAGLE drafting and draft extension within active worker observations."""
    if not _enabled():
        return False
    changed = _install_phase_method(cls, "draft", "draft_proposal")
    return _install_phase_method(cls, "_draft_extend_for_decode", "draft_extension") or changed


def install_sglang_eagle_verify(cls):
    """Capture the supplied EAGLE verify method within active worker observations."""
    if not _enabled():
        return False
    if not callable(getattr(cls, "verify", None)):
        raise ValueError("The supplied EAGLE class has no verify method.")
    return _install_phase_method(cls, "verify", "target_verification")


def _install_bound_draft(instance):
    speculator = getattr(instance, "speculator", None)
    target, method = ((speculator, "propose") if callable(getattr(speculator, "propose", None))
                      else (instance, "propose_draft_token_ids"))
    original = getattr(target, method, None)
    if not callable(original) or getattr(original, "_hpml_phase_wrapper", False):
        return

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        context = _active()
        peagle = context is not None and os.environ.get("HPML_CONFIGURATION") == "peagle"
        if peagle:
            _close_vllm_verification(context)
        result = _phase_call(original, "draft_proposal", type(target).__name__ + "." + method, *args, **kwargs)
        if peagle:
            native_ids = list(instance.input_batch.req_ids)
            rows = _draft_rows(result)
            if len(native_ids) != len(rows):
                raise RuntimeError("P-EAGLE draft rows must match the worker's request order.")
            if context.get("produced_drafts") is not None:
                raise RuntimeError("One P-EAGLE worker observation must produce at most one draft block.")
            produced = {canonical_request_id(rid): {"token_ids": rows[index], "row_index": index}
                        for index, rid in enumerate(native_ids)}
            context["produced_drafts"] = produced
            try:
                reference = weakref.ref(result)
            except TypeError:
                reference = lambda: result
            instance._hpml_proposal_snapshot = {"object_reference": reference, "native_ids": native_ids,
                "rows": rows, "observation_id": context["observation_id"],
                "scheduler_step_id": context["scheduler_step_id"]}
        return result

    wrapped._hpml_phase_wrapper = True
    setattr(target, method, wrapped)


def _draft_rows(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)):
        raise RuntimeError("P-EAGLE must expose its draft token matrix for diagnostic linkage.")
    rows = []
    for row in value:
        if not isinstance(row, (list, tuple)) or any(
                isinstance(token, bool) or not isinstance(token, numbers.Integral) or token < 0 for token in row):
            raise RuntimeError("P-EAGLE draft rows must contain actual nonnegative token IDs.")
        rows.append([int(token) for token in row])
    return rows


def _close_vllm_verification(context):
    phase = context.get("open_verification")
    if phase is not None:
        _event_end(phase)
        context["open_verification"] = None


def _install_vllm_input_link(cls):
    original = getattr(cls, "_prepare_input_ids", None)
    if not callable(original) or getattr(original, "_hpml_input_link", False):
        return

    @functools.wraps(original)
    def prepare(self, scheduler_output, num_reqs, *args, **kwargs):
        context = _active()
        proposals = scheduler_output.scheduled_spec_decode_tokens
        if context is not None and os.environ.get("HPML_CONFIGURATION") == "peagle" and proposals:
            snapshot = getattr(self, "_hpml_proposal_snapshot", None)
            if snapshot is None or self._draft_token_ids is not snapshot["object_reference"]():
                raise RuntimeError("P-EAGLE verification requires the recorded draft tensor that supplies its inputs.")
            asynchronous = self.input_batch.prev_sampled_token_ids is not None
            links = {}
            for current_index, request_id in enumerate(self.input_batch.req_ids[:num_reqs]):
                scheduled = proposals.get(request_id, [])
                if not scheduled:
                    continue
                if asynchronous:
                    index = int(self.prev_positions.np[current_index])
                    if index < 0 or index >= len(snapshot["native_ids"]) or snapshot["native_ids"][index] != request_id:
                        raise RuntimeError("P-EAGLE's previous draft row must identify the current verification request.")
                else:
                    index = snapshot["native_ids"].index(request_id)
                actual = snapshot["rows"][index][:len(scheduled)]
                if len(actual) != len(scheduled) or (not asynchronous and list(scheduled) != actual):
                    raise RuntimeError("P-EAGLE scheduled proposals must match their recorded producer.")
                links[canonical_request_id(request_id)] = {
                    "producer_observation_id": snapshot["observation_id"],
                    "producer_scheduler_step_id": snapshot["scheduler_step_id"],
                    "producer_row_index": index, "token_ids": actual,
                    "input_source": "previous_draft_tensor" if asynchronous else "scheduled_host_tokens",
                    "tensor_identity_verified": True}
            if set(links) != {canonical_request_id(rid) for rid, values in proposals.items() if values}:
                raise RuntimeError("Every P-EAGLE scheduled proposal needs an explicit producer link.")
            context["verification_proposals"] = links
        return original(self, scheduler_output, num_reqs, *args, **kwargs)

    prepare._hpml_input_link = True
    cls._prepare_input_ids = prepare


def install_vllm_runner(cls):
    """Capture vLLM execute_model through sample_tokens as one raw worker iteration."""
    if not _enabled():
        return False
    execute = getattr(cls, "execute_model", None)
    sample = getattr(cls, "sample_tokens", None)
    if not callable(execute) or not callable(sample):
        raise ValueError("The supplied vLLM runner must expose execute_model and sample_tokens.")
    if getattr(execute, "_hpml_phase_wrapper", False):
        return False
    _install_vllm_input_link(cls)

    @functools.wraps(execute)
    def execute_model(self, *args, **kwargs):
        scheduler = args[0] if args else kwargs.get("scheduler_output")
        scheduled = getattr(scheduler, "num_scheduled_tokens", None)
        if not _enabled() or not scheduled:
            if _enabled() and getattr(scheduler, "finished_req_ids", None):
                drain_events()
            return execute(self, *args, **kwargs)
        if getattr(self, "_hpml_pending_phase", None) is not None:
            raise RuntimeError("Finish the previous vLLM sample_tokens call before beginning another observation.")
        _install_bound_draft(self)
        kind = "speculative" if getattr(scheduler, "scheduled_spec_decode_tokens", None) else "prefill_or_ar"
        begin_round(list(scheduled), kind, phase="worker_iteration", tokens=scheduled)
        context = _active()
        context["scheduler_step_id"] = getattr(scheduler, "hpml_diagnostic_step_id", None)
        if os.environ.get("HPML_CONFIGURATION") == "peagle":
            if not context["scheduler_step_id"] or not callable(getattr(self, "_prepare_input_ids", None)):
                error = RuntimeError("P-EAGLE diagnostics require the declared scheduler step identity and input linkage hook.")
                _abort(error)
                raise error
            if kind == "speculative":
                phase = _event_start("verification_and_sampling", context["device_index"], 1, 0,
                    "Target input preparation, forward execution, and rejection sampling through the next draft call.")
                context["phases"].append(phase)
                context["open_verification"] = phase
        pending = {"thread_id": threading.get_ident(), "inside_execute": True, "sampled_inside_execute": False}
        self._hpml_pending_phase = pending
        try:
            result = _phase_call(execute, "target_execution", cls.__name__ + ".execute_model (inclusive)",
                                 self, *args, **kwargs)
            pending["inside_execute"] = False
            if result is not None or pending["sampled_inside_execute"]:
                _close_vllm_verification(context)
                end_round()
                self._hpml_pending_phase = None
            return result
        except BaseException as error:
            self._hpml_pending_phase = None
            _abort(error)
            raise

    @functools.wraps(sample)
    def sample_tokens(self, *args, **kwargs):
        pending = getattr(self, "_hpml_pending_phase", None)
        if pending is None:
            return sample(self, *args, **kwargs)
        if pending["thread_id"] != threading.get_ident() or _active() is None:
            raise RuntimeError("The vLLM sampling hook must finish on the thread that began the worker observation.")
        try:
            result = sample(self, *args, **kwargs)
            if pending["inside_execute"]:
                pending["sampled_inside_execute"] = True
            else:
                _close_vllm_verification(_active())
                end_round()
                self._hpml_pending_phase = None
            return result
        except BaseException as error:
            self._hpml_pending_phase = None
            _abort(error)
            raise

    execute_model._hpml_phase_wrapper = True
    sample_tokens._hpml_phase_wrapper = True
    cls.execute_model, cls.sample_tokens = execute_model, sample_tokens
    return True
