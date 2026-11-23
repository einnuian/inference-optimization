#!/usr/bin/env python3
"""Report one shared Slurm allocation and its saved request evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys


CONFIGURATIONS = (
    "ar-sglang", "eagle3", "dflash", "ar-vllm-v1", "peagle",
    "ar-vllm-v2", "dspark",
    "team-ar", "team-ffn-100", "team-ffn-75", "team-ffn-50",
)
SCOPES = {
    "primary_warmups": "These requests warm up each primary configuration.",
    "primary_timed_benchmarks": "These requests belong to the timed primary benchmark blocks.",
    "stream_qualification": "These requests check streaming and stopping behavior.",
    "diagnostic_warmups": "These requests warm up configurations during diagnostic checks.",
    "diagnostic_timed_benchmarks": "These requests belong to saved diagnostic benchmark blocks.",
    "ancillary_score_probes": "These requests inspect target scores during separate diagnostic checks.",
}
BLOCK_NAME = re.compile(r"[0-9]{2}-.+\.json")
QUALIFICATION_NAME = re.compile(r"[0-9]{2}-(one-token|natural-stop)\.json")


def reject_constant(value: str) -> None:
    raise ValueError(f"JSON contains an unsupported numeric value: {value}.")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def job_id(value: object) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    value = str(value).strip()
    return value or None


def allocation_summary(record: object) -> tuple[dict, list[dict]]:
    if not isinstance(record, dict) or job_id(record.get("job_id")) is None:
        raise ValueError("The allocation record must identify its Slurm job.")
    for field in ("elapsed_seconds", "gpu_count", "allocated_gpu_seconds"):
        if not nonnegative_integer(record.get(field)):
            raise ValueError(f"The allocation record must contain a valid {field} value.")
    expected_gpu_seconds = record["elapsed_seconds"] * record["gpu_count"]
    if record["allocated_gpu_seconds"] != expected_gpu_seconds:
        raise ValueError("Allocated GPU-seconds must equal elapsed seconds multiplied by GPU count.")
    cpu_count = None
    for entry in str(record.get("alloc_tres", "")).split(","):
        name, separator, value = entry.strip().partition("=")
        if name == "cpu" and separator and re.fullmatch(r"[0-9]+", value):
            cpu_count = int(value)
    missing = []
    if cpu_count is None:
        missing.append({
            "code": "missing_cpu_count",
            "message": "The allocation record's alloc_tres field does not contain a CPU count. Allocated CPU-hours remain unknown.",
        })
    return {
        "job_id": job_id(record["job_id"]),
        "start": record.get("start"),
        "end": record.get("end"),
        "state": record.get("state"),
        "elapsed_seconds": record["elapsed_seconds"],
        "gpu_count": record["gpu_count"],
        "allocated_gpu_seconds": record["allocated_gpu_seconds"],
        "allocated_gpu_hours": record["allocated_gpu_seconds"] / 3600,
        "cpu_count": cpu_count,
        "allocated_cpu_hours": (
            record["elapsed_seconds"] * cpu_count / 3600
            if cpu_count is not None else None
        ),
        "alloc_tres": record.get("alloc_tres"),
        "source_command": record.get("source_command"),
        "scope": "The totals cover the complete Slurm allocation, including model loading, requests, and time between requests.",
        "gpu_hours_definition": "Allocated GPU-hours multiply the allocated GPU count by elapsed allocation hours.",
        "cpu_hours_definition": "Allocated CPU-hours multiply the allocated CPU count by elapsed allocation hours.",
    }, missing


class RequestEvidence:
    def __init__(self, trial: Path, allocation_job_id: str, configurations=CONFIGURATIONS):
        self.trial = trial.resolve()
        self.job_id = allocation_job_id
        self.configurations = tuple(configurations)
        self.missing: list[dict] = []
        self.duplicates: list[dict] = []
        self.requests: dict[str, dict] = {}
        self.probe_interpretations: list[dict] = []
        self.probe_tokens: dict[str, dict] = {}
        self.timed: dict[str, list[dict]] = {name: [] for name in CONFIGURATIONS}
        self.diagnostic_timed: dict[str, list[dict]] = {name: [] for name in CONFIGURATIONS}

    def source(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.trial))
        except ValueError:
            return str(path)

    def issue(self, scope: str | list[str], configuration: str, path: Path, code: str, message: str) -> None:
        self.missing.append({
            "scopes": [scope] if isinstance(scope, str) else scope,
            "configuration": configuration,
            "source": self.source(path),
            "code": code,
            "message": message,
        })

    def load(self, path: Path, scope: str | list[str], configuration: str) -> object:
        if not path.resolve().is_relative_to(self.trial):
            self.issue(scope, configuration, path, "outside_trial", "The record resolves outside the trial folder and is excluded.")
            return None
        try:
            return read_json(path)
        except (OSError, ValueError) as exc:
            self.issue(scope, configuration, path, "unavailable_record", f"The record could not be read: {exc}")
            return None

    def matches_job(self, record: dict, scope: str | list[str], configuration: str, path: Path, *, required: bool) -> bool:
        recorded_job = job_id(record.get("slurm_job_id"))
        if recorded_job is None and not required and "slurm_job_id" not in record:
            return True
        if recorded_job != self.job_id:
            self.issue(
                scope, configuration, path, "job_id_mismatch",
                f"The record identifies Slurm job {recorded_job!r}. The report covers job {self.job_id!r}, so this record is excluded.",
            )
            return False
        return True

    def run_matches(self, folder: Path, scopes: str | list[str], configuration: str) -> bool:
        path = folder / "run.json"
        record = self.load(path, scopes, configuration)
        if not isinstance(record, dict):
            if record is not None:
                self.issue(scopes, configuration, path, "invalid_run", "The run record must contain a JSON object.")
            return False
        if record.get("configuration", configuration) != configuration:
            self.issue(scopes, configuration, path, "configuration_mismatch", "The run record identifies a different configuration and is excluded.")
            return False
        return self.matches_job(record, scopes, configuration, path, required=True)

    def add(self, record: object, scope: str, configuration: str, path: Path) -> None:
        if not isinstance(record, dict):
            self.issue(scope, configuration, path, "invalid_request", "The saved request must contain a JSON object.")
            return
        if not self.matches_job(record, scope, configuration, path, required=False):
            return
        request_id = record.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            self.issue(scope, configuration, path, "missing_request_id", "The request has no usable request ID. Its tokens cannot be counted without risking duplicate counting.")
            return
        tokens = record.get("output_tokens")
        if record.get("completion_status") in ("incomplete", "submitted"):
            tokens = None
            self.issue(scope, configuration, path, "incomplete_request", f"Request {request_id} has an incomplete response record. Its final returned token count remains unknown.")
        elif not nonnegative_integer(tokens):
            tokens = None
            self.issue(scope, configuration, path, "missing_token_count", f"Request {request_id} has no valid returned output token count.")
        observation = {
            "request_id": request_id,
            "configuration": configuration,
            "scope": scope,
            "output_tokens": tokens,
            "submitted_ns": record.get("submitted_ns"),
            "completed_ns": record.get("completed_ns"),
            "sources": [self.source(path)],
            "conflicting_evidence": False,
        }
        previous = self.requests.get(request_id)
        if previous is None:
            self.requests[request_id] = observation
            return
        disagreements = [
            field for field in ("configuration", "output_tokens", "submitted_ns", "completed_ns")
            if previous.get(field) is not None and observation.get(field) is not None
            and previous[field] != observation[field]
        ]
        self.duplicates.append({
            "request_id": request_id,
            "retained_scope": previous["scope"],
            "additional_scope": scope,
            "source": self.source(path),
            "conflicting_fields": disagreements,
        })
        previous["sources"].append(self.source(path))
        if disagreements:
            previous["conflicting_evidence"] = True
            previous["output_tokens"] = None
            self.issue(scope, configuration, path, "conflicting_request", f"Copies of request {request_id} disagree on {', '.join(disagreements)}. Its token count is excluded from the total.")
        elif not previous["conflicting_evidence"]:
            for field in ("output_tokens", "submitted_ns", "completed_ns"):
                if previous.get(field) is None:
                    previous[field] = observation[field]

    def collect_run(self, folder: Path, configuration: str, *, diagnostic: bool) -> None:
        warmup_scope = "diagnostic_warmups" if diagnostic else "primary_warmups"
        timed_scope = "diagnostic_timed_benchmarks" if diagnostic else "primary_timed_benchmarks"
        if not self.run_matches(folder, [warmup_scope, timed_scope], configuration):
            return
        warmup_path = folder / "warmup.json"
        warmups = self.load(warmup_path, warmup_scope, configuration)
        if isinstance(warmups, list):
            if not diagnostic and len(warmups) != 3:
                self.issue(warmup_scope, configuration, warmup_path, "primary_warmup_count", f"The file contains {len(warmups)} warmup records. The complete primary trial contains three warmup requests per configuration.")
            for record in warmups:
                self.add(record, warmup_scope, configuration, warmup_path)
        elif warmups is not None:
            self.issue(warmup_scope, configuration, warmup_path, "invalid_warmups", "The warmup record must contain a list of requests.")
        paths = sorted(path for path in folder.iterdir() if BLOCK_NAME.fullmatch(path.name))
        if not diagnostic and len(paths) != 30:
            self.issue(timed_scope, configuration, folder, "primary_block_count", f"The folder contains {len(paths)} benchmark block files. The complete primary trial contains 30 blocks per configuration.")
        table = self.diagnostic_timed if diagnostic else self.timed
        for path in paths:
            block = self.load(path, timed_scope, configuration)
            if not isinstance(block, dict):
                if block is not None:
                    self.issue(timed_scope, configuration, path, "invalid_block", "The benchmark block must contain a JSON object.")
                continue
            if not self.matches_job(block, timed_scope, configuration, path, required=False):
                continue
            records = block.get("requests")
            if not isinstance(records, list):
                self.issue(timed_scope, configuration, path, "missing_block_requests", "The benchmark block has no request list. Its aggregate token count is excluded to avoid duplicate counting.")
                continue
            for record in records:
                self.add(record, timed_scope, configuration, path)
            metrics = block.get("metrics")
            if not isinstance(metrics, dict):
                self.issue(timed_scope, configuration, path, "missing_block_metrics", "The benchmark block has no saved metric dictionary.")
                metrics = None
            table[configuration].append({
                "benchmark": block.get("benchmark"),
                "repetition": block.get("repetition"),
                "source": self.source(path),
                "request_ids": [row.get("request_id") for row in records if isinstance(row, dict)],
                "metrics": metrics,
            })

    def collect_qualification(self, folder: Path, configuration: str) -> None:
        scope = "stream_qualification"
        if not self.run_matches(folder, scope, configuration):
            return
        manifest_path = folder / "qualification.json"
        manifest = self.load(manifest_path, scope, configuration)
        entries: dict[str, dict] = {}
        if isinstance(manifest, dict) and self.matches_job(manifest, scope, configuration, manifest_path, required=True):
            if manifest.get("token_count_complete") is False:
                self.issue(scope, configuration, manifest_path, "incomplete_qualification_counts", "The qualification manifest identifies incomplete token evidence for its requests.")
            records = manifest.get("requests")
            if not isinstance(records, list):
                self.issue(scope, configuration, manifest_path, "missing_qualification_requests", "The qualification manifest has no request list.")
            else:
                declared_count = manifest.get("request_count")
                if nonnegative_integer(declared_count) and declared_count != len(records):
                    self.issue(scope, configuration, manifest_path, "qualification_request_count", "The qualification request count differs from the number of saved request summaries.")
                for record in records:
                    if not isinstance(record, dict):
                        self.issue(scope, configuration, manifest_path, "invalid_qualification_request", "The qualification summary must contain a request object.")
                        continue
                    name = record.get("trace_file")
                    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".json"):
                        self.issue(scope, configuration, manifest_path, "invalid_trace_path", "The qualification trace must identify a JSON file directly inside its configuration folder.")
                        self.add(record, scope, configuration, manifest_path)
                        continue
                    if name in entries:
                        self.issue(scope, configuration, manifest_path, "duplicate_trace_reference", f"The qualification manifest references {name} more than once.")
                        self.add(record, scope, configuration, manifest_path)
                    else:
                        entries[name] = record
        elif manifest is not None and not isinstance(manifest, dict):
            self.issue(scope, configuration, manifest_path, "invalid_qualification_manifest", "The qualification manifest must contain a JSON object.")
        names = set(entries)
        names.update(path.name for path in folder.iterdir() if QUALIFICATION_NAME.fullmatch(path.name))
        if not names:
            self.issue(scope, configuration, folder, "missing_qualification_traces", "The qualification folder contains no identifiable request traces.")
        for name in sorted(names):
            path = folder / name
            trace = self.load(path, scope, configuration)
            summary = entries.get(name)
            if isinstance(trace, dict):
                if not self.matches_job(trace, scope, configuration, path, required=True):
                    continue
                if summary and summary.get("request_id") != trace.get("request_id"):
                    self.issue(scope, configuration, path, "qualification_request_mismatch", "The trace request ID differs from its manifest entry. The trace supplies the request evidence.")
                self.add(trace, scope, configuration, path)
                if summary and summary.get("request_id") == trace.get("request_id"):
                    self.add(summary, scope, configuration, manifest_path)
            else:
                if trace is not None:
                    self.issue(scope, configuration, path, "invalid_qualification_trace", "The qualification trace must contain a request object.")
                if summary:
                    self.add(summary, scope, configuration, manifest_path)

    def collect(self) -> None:
        for configuration in self.configurations:
            self.collect_run(self.trial / configuration, configuration, diagnostic=False)
            qualification = self.trial / "qualification" / configuration
            if qualification.exists():
                self.collect_qualification(qualification, configuration)
            diagnostics = self.trial / "diagnostics" / configuration
            if diagnostics.exists():
                self.collect_run(diagnostics, configuration, diagnostic=True)
        for parent in (self.trial / "attempts", self.trial / "diagnostics" / "attempts"):
            for manifest in sorted(parent.glob("*/qualification/run.json")):
                record = read_json(manifest)
                configuration = record.get("configuration") if isinstance(record, dict) else None
                if configuration in self.configurations:
                    self.collect_qualification(manifest.parent, configuration)
        for folder in sorted((self.trial / "diagnostics" / "attempts").glob("*/logit-probe")):
            for configuration in ("ar-sglang", "eagle3", "dflash"):
                if configuration in self.configurations and re.fullmatch(
                        re.escape(configuration) + r"-[0-9]{8}T[0-9]{12}Z", folder.parent.name):
                    self.collect_probe(folder, configuration)

    def collect_probe(self, folder: Path, configuration: str) -> None:
        scope = "ancillary_score_probes"
        path = folder / "humaneval-source-index-2.json"
        record = self.load(path, scope, configuration)
        interpretation = {"configuration": configuration, "source": self.source(path),
                          "request_id": None, "status": "unavailable", "score_eligible": None}
        self.probe_interpretations.append(interpretation)
        if not isinstance(record, dict):
            if record is not None:
                self.issue(scope, configuration, path, "invalid_probe", "The score probe must contain a JSON object.")
            return
        payload = record.get("request_payload")
        accounting = record.get("accounting")
        scores = record.get("scores_at_first_divergence")
        score_metadata_valid = isinstance(scores, dict) and isinstance(scores.get("eligible"), bool)
        interpretation.update(status=record.get("status"), score_metadata_valid=score_metadata_valid,
            score_eligible=scores.get("eligible") if score_metadata_valid else None,
            score_reason=scores.get("reason") if isinstance(scores, dict) else None,
            output_preservation=record.get("output_comparison"),
            error_type=record.get("error_type"), error=record.get("error"))
        if (record.get("scope") != "ancillary_agreement_qualification" or
                record.get("configuration") != configuration or not isinstance(accounting, dict) or
                accounting.get("classification") != "ancillary_qualification" or
                accounting.get("included_in_primary_timing") is not False):
            self.issue(scope, configuration, path, "invalid_probe_scope", "The probe lacks its separate diagnostic accounting scope.")
            return
        if not self.matches_job(accounting, scope, configuration, path, required=True):
            return
        request_id = payload.get("rid") if isinstance(payload, dict) else None
        interpretation["request_id"] = request_id
        if not isinstance(request_id, str) or re.fullmatch(r"hpml-probe-[0-9a-f]{32}", request_id) is None:
            self.issue(scope, configuration, path, "missing_request_id", "The score probe requires its original submitted request ID for deduplication.")
            return
        response = record.get("response")
        response = response if isinstance(response, dict) else {}
        returned = response.get("output_tokens")
        complete = nonnegative_integer(returned)
        delivered_ids = response.get("output_token_ids")
        if complete and (("output_token_ids" in response and (
                not isinstance(delivered_ids, list) or len(delivered_ids) != returned or
                not all(nonnegative_integer(token) for token in delivered_ids))) or
                ("output_tokens" in accounting and accounting["output_tokens"] != returned)):
            complete = False
            self.issue(scope, configuration, path, "conflicting_probe_returned_tokens", "The probe's returned token sequence and accounting count disagree. Its returned token total remains unknown.")
        ids = response.get("native_output_token_ids")
        native = len(ids) if isinstance(ids, list) and all(nonnegative_integer(token) for token in ids) else None
        observed = accounting.get("observed_native_token_count")
        observed = observed if nonnegative_integer(observed) else None
        declared_native = accounting.get("native_completion_tokens")
        token_conflict = (native is not None and (
            (declared_native is not None and (not nonnegative_integer(declared_native) or declared_native != native)) or
            (observed is not None and observed != native)))
        if token_conflict:
            self.issue(scope, configuration, path, "conflicting_probe_native_tokens", "The probe's native token sequence and accounting counts disagree. Its native token totals remain unknown.")
            native, observed = None, None
        elif native is not None:
            observed = native
        self.add({"request_id": request_id, "slurm_job_id": self.job_id,
                  "output_tokens": returned if complete else None,
                  "completion_status": "completed" if complete else "incomplete",
                  "submitted_ns": accounting.get("submitted_monotonic_ns"),
                  "completed_ns": accounting.get("completed_monotonic_ns")}, scope, configuration, path)
        current = {"request_id": request_id, "native_completion_tokens": native,
                   "observed_native_token_lower_bound": observed, "sources": [self.source(path)],
                   "conflicting_evidence": token_conflict}
        previous = self.probe_tokens.get(request_id)
        if previous is None:
            self.probe_tokens[request_id] = current
        else:
            previous["sources"].extend(current["sources"])
            count_conflict = (previous["native_completion_tokens"] is not None and native is not None and
                              previous["native_completion_tokens"] != native)
            previous["conflicting_evidence"] |= token_conflict or count_conflict
            if previous["conflicting_evidence"]:
                previous["native_completion_tokens"] = None
                previous["observed_native_token_lower_bound"] = None
                self.issue(scope, configuration, path, "conflicting_probe_native_tokens", "Copies of the probe disagree on their native token count. Its native token totals remain unknown.")
            else:
                previous["native_completion_tokens"] = native if native is not None else previous["native_completion_tokens"]
                available = [value for value in (previous["observed_native_token_lower_bound"], observed) if value is not None]
                previous["observed_native_token_lower_bound"] = max(available) if available else None
                if (previous["native_completion_tokens"] is not None and available and
                        max(available) > previous["native_completion_tokens"]):
                    previous.update(conflicting_evidence=True, native_completion_tokens=None,
                                    observed_native_token_lower_bound=None)
                    self.issue(scope, configuration, path, "conflicting_probe_native_tokens", "A saved partial token count exceeds the final count for the same probe. Its native token totals remain unknown.")

    def probe_accounting(self) -> dict:
        rows = []
        for request_id, saved in self.probe_tokens.items():
            row = dict(saved)
            if self.requests[request_id]["conflicting_evidence"]:
                row.update(conflicting_evidence=True, native_completion_tokens=None,
                           observed_native_token_lower_bound=None)
            rows.append(row)
        return {"requests": rows, "unique_request_count": len(rows),
                "native_completion_tokens_with_evidence": sum(row["native_completion_tokens"] or 0 for row in rows),
                "requests_with_unknown_final_native_counts": sum(row["native_completion_tokens"] is None for row in rows),
                "observed_native_token_lower_bound": sum(row["observed_native_token_lower_bound"] or 0 for row in rows),
                "definition": "Native completion tokens include engine stop tokens. Observed counts include partial responses and provide a lower bound. Each probe request ID contributes once. Score eligibility has a separate interpretation record."}

    def totals(self) -> tuple[dict, dict]:
        scopes = {
            name: {"request_count": 0, "returned_output_tokens_with_evidence": 0, "requests_with_missing_token_counts": 0}
            for name in SCOPES
        }
        for record in self.requests.values():
            summary = scopes[record["scope"]]
            summary["request_count"] += 1
            if record["output_tokens"] is None:
                summary["requests_with_missing_token_counts"] += 1
            else:
                summary["returned_output_tokens_with_evidence"] += record["output_tokens"]
        total = {field: sum(scope[field] for scope in scopes.values()) for field in next(iter(scopes.values()))}
        return scopes, total


def build_report(trial: Path, allocation_record: object, configurations=CONFIGURATIONS) -> dict:
    trial = trial.resolve()
    if not trial.is_dir():
        raise ValueError("The trial path must identify an existing results directory.")
    allocation, missing_resources = allocation_summary(allocation_record)
    evidence = RequestEvidence(trial, allocation["job_id"], configurations)
    evidence.collect()
    scopes, totals = evidence.totals()
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trial_directory": str(trial),
        "allocation": allocation,
        "scope_definitions": SCOPES,
        "requests_by_scope": scopes,
        "recorded_request_totals": totals,
        "token_accounting_note": "Token totals cover saved request evidence. Allocation totals cover all time assigned by Slurm. Each request ID contributes at most once across all scopes.",
        "configuration_timed_benchmarks": evidence.timed,
        "diagnostic_timed_benchmarks": evidence.diagnostic_timed,
        "ancillary_probe_token_accounting": evidence.probe_accounting(),
        "ancillary_probe_interpretations": evidence.probe_interpretations,
        "missing_token_evidence": evidence.missing,
        "missing_resource_evidence": missing_resources,
        "duplicate_records": evidence.duplicates,
        "requests": list(evidence.requests.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Report the shared allocation once and count saved requests across the trial.")
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument("--allocation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=CONFIGURATIONS)
    args = parser.parse_args()
    try:
        report = build_report(args.trial.expanduser(), read_json(args.allocation.expanduser()), args.configurations)
        report["allocation_record"] = str(args.allocation.expanduser().resolve())
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"The shared allocation report is saved in {output}.")
        if report["missing_token_evidence"] or report["missing_resource_evidence"]:
            print("The report lists the missing evidence alongside the recorded totals.")
        return 0
    except (OSError, ValueError) as exc:
        print(f"The allocation report could not be completed. {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
