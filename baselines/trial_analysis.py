#!/usr/bin/env python3
"""Calculate paired results and explicit round diagnostics from saved observations."""

from collections import defaultdict
import argparse
import json
from pathlib import Path
import statistics

from trial_recorder import CONFIGURATIONS, distribution, save_json


def ratio(numerator, denominator, scale=1):
    return numerator * scale / denominator if denominator > 0 else None


def round_metrics(rows):
    """Consume engine records after their token meanings have been validated."""
    if not rows:
        raise ValueError("Provide measured verification rounds.")
    per_request = defaultdict(list)
    proposed = accepted = committed = delivered = 0
    for row in rows:
        for name in ("proposed_draft_tokens", "accepted_draft_tokens", "retained_target_tokens",
                     "committed_tokens", "delivered_tokens", "proposed_depth", "accepted_prefix_depth"):
            if type(row[name]) is not int or row[name] < 0:
                raise ValueError("Round counts must be nonnegative integers: " + name)
        if row["accepted_draft_tokens"] > row["proposed_draft_tokens"]:
            raise ValueError("Accepted draft count exceeds the proposal count.")
        if row["accepted_prefix_depth"] > row["proposed_depth"]:
            raise ValueError("Accepted depth exceeds the proposed depth.")
        if row["committed_tokens"] != row["accepted_draft_tokens"] + row["retained_target_tokens"]:
            raise ValueError("Committed tokens must reconcile with accepted drafts and retained target tokens.")
        if row["delivered_tokens"] > row["committed_tokens"]:
            raise ValueError("Delivered round tokens exceed committed round tokens.")
        proposed += row["proposed_draft_tokens"]
        accepted += row["accepted_draft_tokens"]
        committed += row["committed_tokens"]
        delivered += row["delivered_tokens"]
        per_request[row["request_id"]].append(row)
    prefix = []
    for position in range(1, max(row["proposed_depth"] for row in rows) + 1):
        eligible = [row for row in rows if row["proposed_depth"] >= position]
        conditional = [row for row in eligible if row["accepted_prefix_depth"] >= position - 1]
        passed = sum(row["accepted_prefix_depth"] >= position for row in eligible)
        prefix.append({"position": position, "proposed_rounds": len(eligible),
                       "conditional_eligible_rounds": len(conditional), "accepted_rounds": passed,
                       "prefix_acceptance_probability": ratio(passed, len(eligible)),
                       "conditional_acceptance_percent": ratio(passed, len(conditional), 100)})
    return {"round_count": len(rows), "proposed_draft_tokens": proposed,
            "accepted_draft_tokens": accepted, "committed_tokens": committed, "delivered_tokens": delivered,
            "draft_token_acceptance_percent": ratio(accepted, proposed, 100),
            "accepted_draft_tokens_per_round": accepted / len(rows),
            "mean_acceptance_length": committed / len(rows),
            "mean_acceptance_length_across_requests": statistics.mean(
                sum(row["committed_tokens"] for row in group) / len(group) for group in per_request.values()),
            "delivered_output_tokens_per_round": delivered / len(rows), "positions": prefix,
            "timing": {name: distribution([row.get(name) for row in rows]) for name in
                       ("draft_seconds", "verification_seconds", "round_seconds")}}


def compare(method_dir, reference_dir):
    method_dir, reference_dir = Path(method_dir), Path(reference_dir)
    method_meta = json.loads((method_dir / "run.json").read_text())
    reference_meta = json.loads((reference_dir / "run.json").read_text())
    expected = CONFIGURATIONS[method_meta["configuration"]][1]
    if expected is None or reference_meta["configuration"] != expected:
        raise ValueError("Choose the matching ordinary-generation reference.")
    for field in ("prepared_sha256", "target_revision", "input_sha256", "repetitions", "temperature", "max_new_tokens"):
        if method_meta[field] != reference_meta[field]:
            raise ValueError("Paired runs differ in " + field)
    if method_meta["gpu_record"].partition(",")[2].strip() != reference_meta["gpu_record"].partition(",")[2].strip():
        raise ValueError("Paired GPU models or driver versions differ. Review those settings before comparing them.")
    summaries = defaultdict(list)
    files = sorted(method_dir.glob("[0-9][0-9]-*.json"))
    if len(files) != 30:
        raise ValueError("The method must contain all ten benchmarks across three repetitions.")
    for path in files:
        method = json.loads(path.read_text())
        reference = json.loads((reference_dir / path.name).read_text())
        method_rows = {row["prompt_id"]: row for row in method["requests"]}
        reference_rows = {row["prompt_id"]: row for row in reference["requests"]}
        if method_rows.keys() != reference_rows.keys() or len(method_rows) != 3:
            raise ValueError("Paired benchmark blocks must contain the same three prompts.")
        checks = []
        for key, item in method_rows.items():
            control = reference_rows[key]
            if item["input_token_ids"] != control["input_token_ids"]:
                raise ValueError("Paired input token IDs differ.")
            # Delivered output and the common stop/length reason define agreement.
            delivered_ids_match = item["output_token_ids"] == control["output_token_ids"]
            output_length_match = item["output_tokens"] == control["output_tokens"]
            stopping_reason_match = item["finish_reason"] == control["finish_reason"]
            checks.append({
                "prompt_id": key,
                "delivered_token_ids_match": delivered_ids_match,
                "output_length_match": output_length_match,
                "stopping_reason_match": stopping_reason_match,
                "exact_output_agreement": delivered_ids_match and output_length_match and stopping_reason_match,
                "native_diagnostics": {
                    "output_token_ids_match": item["native_output_token_ids"] == control["native_output_token_ids"],
                    "stop_reason_match": item["native_stop_reason"] == control["native_stop_reason"],
                },
            })
        summaries[method["benchmark"]].append({
            "repetition": method["repetition"], "checks": checks,
            "exact_output_agreement_percent": sum(row["exact_output_agreement"] for row in checks) / 3 * 100,
            "speedup_over_ar": ratio(method["metrics"]["otps"], reference["metrics"]["otps"])})
    valid = lambda rows: [row["speedup_over_ar"] for row in rows if row["speedup_over_ar"] is not None]
    return {name: {"repetitions": rows, "same_physical_gpu": method_meta["gpu_uuid"] == reference_meta["gpu_uuid"],
                   "speedup_mean": statistics.mean(valid(rows)) if valid(rows) else None,
                   "speedup_sample_sd": statistics.stdev(valid(rows)) if len(valid(rows)) > 1 else None,
                   "exact_output_agreement_percent": statistics.mean(row["exact_output_agreement_percent"] for row in rows)}
            for name, rows in summaries.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pair = sub.add_parser("compare")
    pair.add_argument("method", type=Path)
    pair.add_argument("reference", type=Path)
    pair.add_argument("--output", type=Path, required=True)
    rounds = sub.add_parser("rounds")
    rounds.add_argument("trace", type=Path)
    rounds.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        result = compare(args.method, args.reference)
    else:
        result = round_metrics([json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()])
    save_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
