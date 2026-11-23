#!/usr/bin/env python3
"""Qualify released P-EAGLE, H-Spec and DSpark with four greedy setup prompts.

Run after the basic H-Spec GPU check and verified Meta target import:
    python baselines/qualify_hspec.py --output-dir "$HPML_ROOT/runs/hspec-models-$SLURM_JOB_ID"

Each ordinary/speculative stage starts a separate process. The script records
checkpoint loading, native vLLM output, the common output policy and token/stop
comparisons. Performance benchmarking uses the common baseline measurement protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import traceback

from qualify_eagle import PROMPTS, apply_output_policy, compare_output_records, write_record

TARGET_ID = "meta-llama/Llama-3.1-8B-Instruct"
TARGET_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
METHODS = {
    "peagle": {"repo": "weifanjiang/llama3.1-8b-it.speculators.peagle",
               "revision": "0db2ae706f7a314286516835006183efdcb5bb39", "runner": "0",
               "settings": {"method": "eagle3", "parallel_drafting": True, "num_speculative_tokens": 7}},
    "hspec": {"repo": "weifanjiang/llama3.1-8b-it.speculators.hspec",
              "revision": "a43cf8a437816ddf894270ef03b23520dedcde88", "runner": "1",
              "settings": {"method": "mamba_attn_hybrid", "enable_kv_sharing": True, "num_speculative_tokens": 7}},
    "dspark": {"repo": "weifanjiang/llama3.1-8b-it.speculators.dspark",
               "revision": "110d99b89501e3007a893da937d9560e087243c0", "runner": "1",
               "settings": {"method": "dspark", "num_speculative_tokens": 7}},
}
STAGES = ("target-v1", "peagle", "target-v2", "hspec", "dspark")


def require_snapshot(home: Path, name: str, repo: str, revision: str):
    provenance = home / "hpml-provenance" / name / f"{revision}.json"
    record = json.loads(provenance.read_text())
    if record["repo_id"] != repo or record["revision"] != revision:
        raise RuntimeError(f"The {name} provenance names another artifact.")
    if name == "target":
        acquisition = record.get("acquisition", {})
        if acquisition.get("weight_source_repo") != TARGET_ID or acquisition.get("weight_source_revision") != TARGET_REVISION:
            raise RuntimeError("Complete the verified direct-Meta target import before loading.")
    expected = home / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    if expected.resolve() != Path(record["snapshot_path"]).resolve():
        raise RuntimeError(f"The {name} snapshot path differs from its provenance.")
    for filename, size in record["required_file_bytes"].items():
        if not (expected / filename).is_file() or (expected / filename).stat().st_size != size:
            raise RuntimeError(f"The {name} snapshot is incomplete: {filename}.")
    return expected, {"path": str(provenance), "sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
                      "repo_id": repo, "revision": revision, "verification": record["verification"]}


def checkpoint_inventory(path: Path):
    tensors = []
    for filename in sorted(path.glob("*.safetensors")):
        with filename.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            if size > 16 * 1024 * 1024:
                raise RuntimeError("Unexpected safetensors metadata size.")
            header = json.loads(handle.read(size))
        tensors.extend({"file": filename.name, "name": key, "shape": value["shape"], "dtype": value["dtype"]}
                       for key, value in header.items() if key != "__metadata__")
    return tensors


def native_finish_matches(record):
    if not record["producer_finished"] or not record["raw_boundary_compliant"]:
        return False
    if record["stop_reason"] == "length":
        return record["producer_finish_reason"] == "length" and record["producer_stop_reason"] is None
    if record["stop_reason"] == "stop_token":
        return (record["producer_finish_reason"] == "stop"
                and record["producer_stop_reason"] in (None, record["generated_token_ids"][-1]))
    return False


def execute_stage(stage: str, output: Path, root: Path, home: Path):
    method = METHODS.get(stage)
    runner = method["runner"] if method else ("0" if stage == "target-v1" else "1")
    os.environ.update(VLLM_USE_V2_MODEL_RUNNER=runner, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
                      HPML_HSPEC_AUDIT_DIR=str(output), PYTHONNOUSERSITE="1",
                      PYTHONPATH=str(Path(__file__).resolve().parent))
    os.environ["TRITON_CACHE_DIR"] = str(root / "cache/hspec/triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(root / "cache/hspec/torchinductor")
    target, target_provenance = require_snapshot(home, "target", TARGET_ID, TARGET_REVISION)
    draft = None
    draft_provenance = None
    if method:
        draft, draft_provenance = require_snapshot(home, stage, method["repo"], method["revision"])
    if metadata.version("vllm") != "0.25.1+cu129.hspec":
        raise RuntimeError("Use the recorded patched vLLM environment.")
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Allocate exactly one visible GPU for model qualification.")
    if torch.__version__ != "2.11.0+cu129":
        raise RuntimeError("Use the recorded Torch CUDA 12.9 package.")
    tokenizer = AutoTokenizer.from_pretrained(target, use_fast=False, local_files_only=True)
    generation = json.loads((target / "generation_config.json").read_text())
    stops = generation["eos_token_id"]
    stop_ids = set(stops if isinstance(stops, list) else [stops])
    settings = dict(
        model=str(target), tokenizer=str(target), dtype="bfloat16", seed=0,
        tensor_parallel_size=1, max_model_len=512, max_num_seqs=1,
        max_num_batched_tokens=512, gpu_memory_utilization=0.65,
        enable_prefix_caching=False, enable_chunked_prefill=True,
        enforce_eager=True, compilation_config={"mode": 0},
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
        load_format="safetensors", generation_config="auto",
        worker_cls="hspec_audit_worker.AuditedWorker", disable_log_stats=True,
    )
    if method:
        settings["speculative_config"] = {"model": str(draft), **method["settings"]}
    write_record(output / "requested-settings.json", {
        "stage": stage, "runner_environment_value": runner, "engine_arguments": settings,
        "canonical_stop_token_ids": sorted(stop_ids), "target_provenance": target_provenance,
        "draft_provenance": draft_provenance,
        "source_environment_record": json.loads((root / "envs/hspec-cu129-py313/hpml-setup.json").read_text()),
    })
    inventories = {"target": checkpoint_inventory(target)}
    if draft:
        inventories["draft"] = checkpoint_inventory(draft)
    write_record(output / "checkpoint-keys.json", inventories)
    print(f"Loading {stage} in a separate process with Model Runner {'V1' if runner == '0' else 'V2'}.", flush=True)
    engine = LLM(**settings)
    states = engine.collective_rpc("hpml_model_state", timeout=180)
    write_record(output / "loaded-model-state.json", {"workers": states})
    prompt_records = []
    for name, prompt, cap in PROMPTS:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_dict=False)
        if not isinstance(input_ids, list) or not all(isinstance(value, int) for value in input_ids):
            raise RuntimeError("The canonical chat template must return a token-ID list.")
        parameters = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1, max_tokens=cap,
                                    min_tokens=0, seed=0, stop_token_ids=sorted(stop_ids),
                                    ignore_eos=False, skip_special_tokens=True)
        request = engine.generate({"prompt_token_ids": input_ids}, parameters, use_tqdm=False)[0]
        if request.prompt_token_ids != input_ids or len(request.outputs) != 1:
            raise RuntimeError("The producer changed the input IDs or completion count.")
        completion = request.outputs[0]
        raw = list(completion.token_ids)
        row = {
            "name": name, "prompt": prompt, "max_new_tokens": cap, "input_token_ids": input_ids,
            "producer_finished": request.finished, "producer_finish_reason": completion.finish_reason,
            "producer_stop_reason": completion.stop_reason, "producer_returned_text": completion.text,
            **apply_output_policy(raw, cap, stop_ids),
        }
        row["returned_text"] = tokenizer.decode(row["returned_token_ids"], skip_special_tokens=True)
        row["native_finish_reason_compliant"] = native_finish_matches(row)
        prompt_records.append(row)
        print(f"{stage}: {name}, {len(raw)} native engine tokens, {completion.finish_reason}.", flush=True)
    audits = [json.loads(path.read_text()) for path in sorted(output.glob("load-audit-*.json"))]
    coverage = []
    for role, checkpoint in inventories.items():
        model_path = str(target if role == "target" else draft)
        matching = [record for audit in audits for record in audit["models"] if record["model_path"] == model_path]
        observed = [item["name"] for record in matching for item in record["checkpoint_inputs"]]
        saved = [item["name"] for item in checkpoint]
        coverage.append({
            "role": role, "saved_key_count": len(saved), "observed_key_count": len(observed),
            "all_checkpoint_key_names_observed": sorted(saved) == sorted(observed),
            "unobserved_checkpoint_keys": sorted(set(saved) - set(observed)),
            "unexpected_observed_keys": sorted(set(observed) - set(saved)),
            "native_loader_completed": bool(matching) and all(record["completed"] for record in matching),
            "native_full_parameter_return_available": bool(matching) and all(record["model_load_weights_return"] is not None for record in matching),
        })
    write_record(output / "output.json", {
        "stage": stage, "scope": "Four setup prompts and model-loading evidence; performance measurements use the common baseline protocol.",
        "target_revision": TARGET_REVISION, "draft_revision": method["revision"] if method else None,
        "model_runner": "V1" if runner == "0" else "V2", "runner_environment_value": runner,
        "requested_attention_backend": "FLASH_ATTN", "requested_flash_attention_version": 2,
        "actual_worker_state_file": "loaded-model-state.json", "load_key_coverage": coverage,
        "tensor_value_verification": "The audit records all observed names, shapes, dtypes and loader destination sets. Exact loaded-value comparison remains a separate check.",
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda, "dtype": "bfloat16",
        "vllm": metadata.version("vllm"), "transformers": metadata.version("transformers"),
        "speculators": metadata.version("speculators"), "python": sys.version,
        "gpu": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability(0)),
        "node": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "canonical_stop_token_ids": sorted(stop_ids), "prompts": prompt_records,
        "raw_producer_boundary": "Unmodified vLLM CompletionOutput.token_ids after the engine's native scheduler stopping, before the project output adapter; the token list includes a terminal stop token.",
        "internal_verifier_count_scope": "Speculative proposals and verifier tokens discarded inside the engine require separate cycle instrumentation.",
        "output_policy": "Keep the first canonical stop token or strict output cap; exclude a terminal stop token from returned tokens/text. Preserve all native engine token IDs separately.",
    })
    engine.llm_engine.engine_core.shutdown(timeout=60)


def compare_stage(ordinary, speculative):
    result = compare_output_records(ordinary, speculative)
    native_stops = all(row["native_finish_reason_compliant"] for row in ordinary["prompts"] + speculative["prompts"])
    result["native_engine_finish_reasons_compliant"] = native_stops
    result["native_engine_finish_reasons_match"] = all(
        left["producer_finish_reason"] == right["producer_finish_reason"]
        and left["producer_stop_reason"] == right["producer_stop_reason"]
        for left, right in zip(ordinary["prompts"], speculative["prompts"]))
    result["native_returned_text_matches"] = all(
        left["producer_returned_text"] == right["producer_returned_text"]
        for left, right in zip(ordinary["prompts"], speculative["prompts"]))
    key_coverage = all(row["all_checkpoint_key_names_observed"] and row["native_loader_completed"]
                       for row in ordinary["load_key_coverage"] + speculative["load_key_coverage"])
    result["checkpoint_input_key_coverage_complete"] = key_coverage
    result["native_output_ready"] = (result["native_output_ready"] and native_stops
        and result["native_engine_finish_reasons_match"] and result["native_returned_text_matches"])
    result["adapted_output_ready"] = (result["adapted_output_ready"]
        and all(row["native_finish_reason_compliant"] for row in ordinary["prompts"])
        and all(row["producer_finished"] for row in speculative["prompts"]))
    result["exit_code"] = 0 if result["native_output_ready"] and key_coverage else (3 if result["adapted_output_ready"] and key_coverage else 2)
    result["load_value_scope"] = "Complete checkpoint-input key coverage is recorded separately from exact tensor-value identity and review of the released loader's explicit skip rules."
    result["scope"] = "These four prompts establish observed setup behavior. The common baseline comparison adds broader correctness, streaming, latency, GPU resource and actual-charge measurements."
    return result


def overall_exit_code(comparisons, ordinary_runner_comparison=None):
    codes = {item["exit_code"] for item in comparisons.values()}
    if ordinary_runner_comparison is not None:
        codes.add(ordinary_runner_comparison["exit_code"])
    return next((code for code in (1, 2, 3) if code in codes), 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHODS), default=list(METHODS))
    args = parser.parse_args()
    from download_checkpoints import require_compute_allocation, require_scratch_cache
    require_compute_allocation()
    home = require_scratch_cache()
    root = Path(os.environ["HPML_ROOT"]).resolve()
    output = args.output_dir.resolve()
    if root / "runs" not in output.parents:
        raise RuntimeError("Use a fresh evidence directory under HPML_ROOT/runs.")
    os.umask(0o077)
    if args.stage:
        output.mkdir(parents=True, exist_ok=False)
        try:
            execute_stage(args.stage, output, root, home)
        except Exception as error:
            write_record(output / "failure.json", {"stage": args.stage, "error": f"{type(error).__name__}: {error}",
                                                   "traceback": traceback.format_exc()})
            raise
        return 0
    # Require verified Meta provenance before launching any worker process.
    require_snapshot(home, "target", TARGET_ID, TARGET_REVISION)
    basic = json.loads((root / "envs/hspec-cu129-py313/hpml-gpu-check.json").read_text())
    if not basic.get("checks") or any(value != "passed" for value in basic["checks"].values()):
        raise RuntimeError("Complete the recorded basic GPU runtime checks first.")
    output.mkdir(parents=True, exist_ok=False)
    planned = [stage for stage in STAGES if stage in args.methods or (
        stage == "target-v1" and "peagle" in args.methods) or (
        stage == "target-v2" and any(method in args.methods for method in ("hspec", "dspark")))]
    stages = {}
    for stage in planned:
        reference = "target-v1" if stage == "peagle" else "target-v2"
        if stage in METHODS and stages.get(reference, {}).get("returncode") != 0:
            stages[stage] = {"status": "ordinary generation failed", "returncode": None}
            continue
        with (output / f"{stage}.log").open("x") as log:
            process = subprocess.run([sys.executable, __file__, "--output-dir", str(output / stage), "--stage", stage],
                                     stdout=log, stderr=subprocess.STDOUT)
        stages[stage] = {"status": "completed" if process.returncode == 0 else "failed", "returncode": process.returncode}
    comparisons = {}
    for method in args.methods:
        reference = "target-v1" if method == "peagle" else "target-v2"
        if stages.get(reference, {}).get("returncode") == 0 and stages.get(method, {}).get("returncode") == 0:
            comparison = compare_stage(json.loads((output / reference / "output.json").read_text()),
                                       json.loads((output / method / "output.json").read_text()))
            write_record(output / f"{method}-comparison.json", comparison)
            comparisons[method] = comparison
        else:
            comparisons[method] = {"native_output_ready": False, "adapted_output_ready": False,
                                   "exit_code": 1, "reason": "Inspect the ordinary and method stage records."}
    ordinary_runner_comparison = None
    if all(stages.get(name, {}).get("returncode") == 0 for name in ("target-v1", "target-v2")):
        ordinary_runner_comparison = compare_stage(
            json.loads((output / "target-v1/output.json").read_text()),
            json.loads((output / "target-v2/output.json").read_text()))
        ordinary_runner_comparison.update(reference_stage="ordinary Model Runner V1", comparison_stage="ordinary Model Runner V2")
        write_record(output / "ordinary-runner-comparison.json", ordinary_runner_comparison)
    exit_code = overall_exit_code(comparisons, ordinary_runner_comparison)
    result = {"stages": stages, "comparisons": comparisons, "ordinary_runner_comparison": ordinary_runner_comparison,
              "exit_code": exit_code,
              "scope": "Pretrained inference setup qualification on four prompts. Performance and quality conclusions require the common baseline measurements."}
    write_record(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
