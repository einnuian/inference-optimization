"""Read and compare native vLLM speculative-decoding counters.

The reader only performs GET requests to a loopback /metrics endpoint. Collection
belongs outside the primary timing interval. The settling policy is recorded as
evidence of observed stability; the caller must verify the runtime's update cadence.
The physical GPU UUID comes from the caller's allocation evidence.
"""

import ipaddress
import math
import re
import socket
import time
import urllib.request
from urllib.parse import urlsplit, urlunsplit


CORE_NAMES = {
    "num_drafts": ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_drafts"),
    "num_draft_tokens": ("vllm:spec_decode_num_draft_tokens_total", "vllm:spec_decode_num_draft_tokens"),
    "num_accepted_tokens": ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_accepted_tokens"),
}
POSITION_NAMES = (
    "vllm:spec_decode_num_accepted_tokens_per_pos_total",
    "vllm:spec_decode_num_accepted_tokens_per_pos",
    "vllm:spec_decode_accepted_tokens_per_pos_total",
    "vllm:spec_decode_accepted_tokens_per_pos",
)
SUCCESS_NAMES = ("vllm:request_success_total", "vllm:request_success")
SPEC_NAMES = frozenset(name for names in CORE_NAMES.values() for name in names) | frozenset(POSITION_NAMES)
_NAME = re.compile(r"[a-zA-Z_:][a-zA-Z0-9_:]*")
_LABEL_NAME = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_NUMBER = re.compile(r"(?:[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|NaN|[-+]?Inf)\Z")
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class SnapshotTimeout(TimeoutError):
    """Retain the last observation when counters fail to settle before the deadline."""

    def __init__(self, message, last_observation):
        super().__init__(message)
        self.last_observation = last_observation


def _parse_labels(source):
    labels = {}
    index = 0
    while index < len(source):
        while index < len(source) and source[index].isspace():
            index += 1
        if index == len(source):
            break
        match = _LABEL_NAME.match(source, index)
        if match is None:
            raise ValueError("Invalid Prometheus label name.")
        name = match.group()
        index = match.end()
        while index < len(source) and source[index].isspace():
            index += 1
        if index == len(source) or source[index] != "=":
            raise ValueError("A Prometheus label requires an equals sign.")
        index += 1
        while index < len(source) and source[index].isspace():
            index += 1
        if index == len(source) or source[index] != '"':
            raise ValueError("A Prometheus label value must be quoted.")
        index += 1
        value = []
        closed = False
        while index < len(source):
            char = source[index]
            index += 1
            if char == '"':
                closed = True
                break
            if char == "\\":
                if index == len(source) or source[index] not in {'n', '"', "\\"}:
                    raise ValueError("Invalid Prometheus label escape.")
                char = {"n": "\n", '"': '"', "\\": "\\"}[source[index]]
                index += 1
            value.append(char)
        if not closed or name in labels:
            raise ValueError("Prometheus labels must have closed quotes and unique names.")
        labels[name] = "".join(value)
        while index < len(source) and source[index].isspace():
            index += 1
        if index < len(source):
            if source[index] != ",":
                raise ValueError("Prometheus labels must be separated by commas.")
            index += 1
    return labels


def parse_prometheus(raw):
    """Return samples with names, decoded labels, values, and their original lines.

    Nonfinite values remain strings so the complete observation can be saved as
    strict JSON. Counter validation rejects those values when they are relevant.
    """
    samples = []
    seen = set()
    for line_number, original in enumerate(raw.splitlines(), 1):
        line = original.strip()
        if not line or line.startswith("#"):
            continue
        match = _NAME.match(line)
        if match is None:
            raise ValueError("Invalid Prometheus sample on line %d." % line_number)
        name, index, labels = match.group(), match.end(), {}
        if index < len(line) and line[index] == "{":
            start = index + 1
            index = start
            quoted = escaped = False
            while index < len(line):
                char = line[index]
                if escaped:
                    escaped = False
                elif char == "\\" and quoted:
                    escaped = True
                elif char == '"':
                    quoted = not quoted
                elif char == "}" and not quoted:
                    break
                index += 1
            if index == len(line):
                raise ValueError("Unclosed Prometheus labels on line %d." % line_number)
            labels = _parse_labels(line[start:index])
            index += 1
        if index == len(line) or not line[index].isspace():
            raise ValueError("A Prometheus sample requires whitespace before its value.")
        remainder = re.split(r"\s+#", line[index:].strip(), maxsplit=1)
        tokens = remainder[0].split()
        if len(tokens) not in {1, 2} or any(_NUMBER.fullmatch(token) is None for token in tokens):
            raise ValueError("Invalid Prometheus value or timestamp on line %d." % line_number)
        number = float(tokens[0])
        value = number if math.isfinite(number) else tokens[0]
        sample = {"name": name, "labels": labels, "value": value,
                  "value_text": tokens[0], "raw": original, "line": line_number}
        if len(tokens) == 2:
            if not math.isfinite(float(tokens[1])):
                raise ValueError("A Prometheus sample timestamp must be finite.")
            sample["timestamp_text"] = tokens[1]
        if len(remainder) == 2:
            sample["exemplar_text"] = remainder[1].strip()
        identity = (name, tuple(sorted(labels.items())))
        if identity in seen:
            raise ValueError("The Prometheus response contains a duplicate sample.")
        seen.add(identity)
        samples.append(sample)
    return samples


def _metrics_url(url):
    parsed = urlsplit(url)
    host = parsed.hostname
    if parsed.scheme not in {"http", "https"} or not host or parsed.username is not None or parsed.password is not None:
        raise ValueError("Use an HTTP loopback server URL without credentials.")
    if host != "localhost":
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError("Native metrics collection requires a loopback server.")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/", "/metrics", "/v1", "/v1/"}:
        raise ValueError("Use the server base URL or its /metrics endpoint.")
    if parsed.port is not None and not 1 <= parsed.port <= 65535:
        raise ValueError("Use a valid server port.")
    return urlunsplit((parsed.scheme, parsed.netloc, "/metrics", "", ""))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("The loopback metrics endpoint must respond directly.")


def _fetch_metrics(url, timeout):
    # Explicitly disable environment proxies and redirects for local collection.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    request = urllib.request.Request(url, method="GET", headers={"Accept": "text/plain; version=0.0.4"})
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise ValueError("The metrics response exceeds the collection size limit.")
    return raw.decode("utf-8")


def _count(value):
    if isinstance(value, bool):
        raise ValueError("Native counters must contain finite, nonnegative integer counts.")
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError("A native counter has an invalid value.") from error
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError("Native counters must contain finite, nonnegative integer counts.")
    return int(number)


def _samples_by_key(samples, names):
    selected = {}
    for sample in samples:
        if sample["name"] not in names:
            continue
        if not isinstance(sample["labels"], dict) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in sample["labels"].items()
        ):
            raise ValueError("Counter labels must be string-to-string mappings.")
        key = (sample["name"], tuple(sorted(sample["labels"].items())))
        if key in selected:
            raise ValueError("A counter sample appears twice.")
        selected[key] = _count(sample["value"])
    return selected


def _check_aliases(values, aliases):
    present = {name for name, labels in values if name in aliases}
    if len(present) > 1:
        raise ValueError("Multiple aliases expose the same counter; summing them would double-count observations.")


def _success_count(samples):
    values = _samples_by_key(samples, SUCCESS_NAMES)
    _check_aliases(values, SUCCESS_NAMES)
    return sum(values.values()) if values else None


def _fingerprint(samples):
    names = SPEC_NAMES | frozenset(SUCCESS_NAMES)
    values = _samples_by_key(samples, names)
    for aliases in list(CORE_NAMES.values()) + [POSITION_NAMES, SUCCESS_NAMES]:
        _check_aliases(values, aliases)
    return tuple(sorted(values.items()))


def snapshot(url, expected_completions=None, timeout=15, *, settle_seconds=6.0,
             stable_seconds=1.0, poll_interval=0.25, gpu_uuid=None):
    """Read settled native counters without submitting a model request.

    The default policy waits six seconds, then requires matching relevant counters
    across observations at least one second apart. A changed counter restarts the
    stability window. An expected completion total must match exactly. The returned
    policy and raw observation permit independent inspection of that decision.
    """
    url = _metrics_url(url)
    parameters = (timeout, settle_seconds, stable_seconds, poll_interval)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in parameters):
        raise ValueError("Collection durations must be finite numbers.")
    if timeout <= 0 or settle_seconds < 0 or stable_seconds <= 0 or poll_interval <= 0 or timeout <= settle_seconds + stable_seconds:
        raise ValueError("The timeout must exceed the initial wait plus the positive stability window.")
    if expected_completions is not None and (type(expected_completions) is not int or expected_completions < 0):
        raise ValueError("The expected absolute completion count must be a nonnegative integer.")
    if gpu_uuid is not None and (not isinstance(gpu_uuid, str) or not gpu_uuid.startswith("GPU-")):
        raise ValueError("Use the assigned physical GPU UUID from the allocation record.")
    started = time.monotonic()
    deadline = started + timeout
    time.sleep(settle_seconds)
    first_stable = last = None
    prior_fingerprint = None
    observations = 0
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        fetch_started_ns = time.monotonic_ns()
        raw = _fetch_metrics(url, min(5.0, remaining))
        completed_ns = time.monotonic_ns()
        observed_unix_ns = time.time_ns()
        samples = parse_prometheus(raw)
        successes = _success_count(samples)
        last = {"schema_version": 1, "metrics_url": url, "gpu_uuid": gpu_uuid,
                "hostname": socket.gethostname(), "clock": "time.monotonic_ns",
                "observed_unix_ns": observed_unix_ns, "observed_monotonic_ns": completed_ns,
                "fetch_started_monotonic_ns": fetch_started_ns,
                "raw_prometheus": raw, "samples": samples,
                "expected_completions": expected_completions, "observed_completions": successes,
                "collection_policy": {"initial_wait_seconds": settle_seconds,
                                      "required_stability_seconds": stable_seconds,
                                      "poll_interval_seconds": poll_interval, "timeout_seconds": timeout}}
        observations += 1
        if time.monotonic() >= deadline:
            break
        fingerprint = _fingerprint(samples)
        if expected_completions is not None and successes is not None and successes > expected_completions:
            raise ValueError("The server reports %d completed requests. The trial expects %d. Verify the request scope."
                             % (successes, expected_completions))
        completion_matches = expected_completions is None or successes == expected_completions
        if not completion_matches:
            first_stable, prior_fingerprint = None, None
        elif first_stable is None or fingerprint != prior_fingerprint:
            first_stable, prior_fingerprint = last, fingerprint
        elif (completed_ns - first_stable["observed_monotonic_ns"]) / 1e9 >= stable_seconds:
            last["settling"] = {
                "initial_wait_seconds": settle_seconds, "required_stability_seconds": stable_seconds,
                "observed_stability_seconds": (completed_ns - first_stable["observed_monotonic_ns"]) / 1e9,
                "observation_count": observations,
                "first_matching_monotonic_ns": first_stable["observed_monotonic_ns"],
                "first_matching_unix_ns": first_stable["observed_unix_ns"],
                "first_matching_counter_samples": [sample for sample in first_stable["samples"]
                                                   if sample["name"] in SPEC_NAMES or sample["name"] in SUCCESS_NAMES],
                "scope": "Observed counter stability under the recorded collection policy.",
            }
            return last
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(poll_interval, remaining))
    raise SnapshotTimeout("The native counters did not meet the completion and stability checks before the deadline.", last)


def delta(before, after, *, gpu_uuid=None):
    """Aggregate spec-counter deltas from one recorded physical GPU server.

    Supply gpu_uuid here or in both snapshots. All engine/model label series are
    preserved in the result and summed for the three totals. Position observations
    remain raw deltas. The function derives acceptance and accepted tokens per
    counted draft round. It derives neither tau nor position acceptance rates.
    """
    if _metrics_url(before["metrics_url"]) != _metrics_url(after["metrics_url"]):
        raise ValueError("Both snapshots must describe the same loopback metrics endpoint.")
    if before.get("hostname") != after.get("hostname"):
        raise ValueError("Both snapshots must come from the same compute host.")
    if after["observed_monotonic_ns"] <= before["observed_monotonic_ns"]:
        raise ValueError("The second snapshot must follow the first on the same host clock.")
    identities = (before.get("gpu_uuid"), after.get("gpu_uuid"), gpu_uuid)
    if gpu_uuid is None and (identities[0] is None or identities[1] is None):
        raise ValueError("Supply the physical GPU UUID explicitly or record it in both snapshots.")
    if any(value is not None and (not isinstance(value, str) or not value.startswith("GPU-")) for value in identities):
        raise ValueError("Use the physical GPU UUID recorded for this one-GPU server.")
    supplied = {value for value in identities if value is not None}
    if len(supplied) != 1:
        raise ValueError("Provide one matching physical GPU UUID for both counter snapshots.")
    identity = next(iter(supplied))
    for sample_name in ("process_start_time_seconds",):
        old_start = {(sample["name"], tuple(sorted(sample["labels"].items()))): sample["value"]
                     for sample in before["samples"] if sample["name"] == sample_name}
        new_start = {(sample["name"], tuple(sorted(sample["labels"].items()))): sample["value"]
                     for sample in after["samples"] if sample["name"] == sample_name}
        if old_start != new_start:
            raise ValueError("The serving process identity changed between snapshots.")
    old = _samples_by_key(before["samples"], SPEC_NAMES)
    new = _samples_by_key(after["samples"], SPEC_NAMES)
    if old.keys() != new.keys():
        raise ValueError("Spec counter names and label sets changed between snapshots; preserve both observations for inspection.")
    for aliases in list(CORE_NAMES.values()) + [POSITION_NAMES]:
        _check_aliases(old, aliases)
    for aliases in CORE_NAMES.values():
        if not any(name in aliases for name, labels in old):
            raise ValueError("A required speculative counter is unavailable. Check the runtime's metrics configuration.")
    observations = []
    for key in sorted(old):
        if new[key] < old[key]:
            raise ValueError("A native counter decreased; check for a restart or counter reset.")
        observations.append({"name": key[0], "labels": dict(key[1]),
                             "before": old[key], "after": new[key], "delta": new[key] - old[key]})
    totals = {kind: sum(row["delta"] for row in observations if row["name"] in aliases)
              for kind, aliases in CORE_NAMES.items()}
    rounds, proposed, accepted = (totals[name] for name in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"))
    if accepted > proposed or (rounds == 0 and (accepted or proposed)):
        raise ValueError("The counter deltas are inconsistent; verify statistics update boundaries.")
    return {"schema_version": 1, "metrics_url": before["metrics_url"], "gpu_uuid": identity,
            "aggregation_scope": "All native spec-counter label series from the recorded one-GPU server.",
            "before_observed_monotonic_ns": before["observed_monotonic_ns"],
            "after_observed_monotonic_ns": after["observed_monotonic_ns"],
            "totals": totals, "acceptance_rate": accepted / proposed if proposed else None,
            "acceptance_rate_unit": "fraction",
            "accepted_draft_tokens_per_round": accepted / rounds if rounds else None,
            "counter_deltas": [row for row in observations if row["name"] not in POSITION_NAMES],
            "position_counter_deltas": [row for row in observations if row["name"] in POSITION_NAMES]}
