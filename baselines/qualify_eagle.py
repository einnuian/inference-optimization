#!/usr/bin/env python3
"""Check pretrained EAGLE-3 loading and greedy outputs on four setup prompts.

Run inside one allocated GPU using the prepared EAGLE environment:
    python baselines/qualify_eagle.py --output-dir "$HPML_ROOT/runs/eagle-check-$SLURM_JOB_ID"

The target-only and EAGLE-3 stages execute in separate processes. The output
records preserve raw producer tokens and the common returned-output policy.
These small checks establish setup behavior; performance and quality benchmarks
require the common baseline measurement protocol.
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

SOURCE_REVISION = "cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b"
TARGET_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
DRAFT_REVISION = "ada412b672e293d682423de84a095447bf38a637"
RUNTIME_PYTHON = (3, 10, 22)
RUNTIME_PACKAGES = {"torch": "2.6.0+cu124", "transformers": "4.53.1", "accelerate": "0.26.0"}
PROMPTS = (
    ("short_stop", "Reply with exactly the word READY.", 32),
    ("length_cap", "List the integers from one through one hundred, separated by commas.", 16),
    ("arithmetic", "What is 7 multiplied by 8? Answer briefly.", 32),
    ("code", "Write a Python function that adds two numbers.", 32),
)


def check_runtime_versions(python_version: tuple[int, ...], packages: dict[str, str]) -> None:
    """Require the interpreter and core packages used by the prepared EAGLE stack."""
    if tuple(python_version[:3]) != RUNTIME_PYTHON:
        raise RuntimeError("EAGLE qualification requires the prepared Python 3.10.22 interpreter.")
    for name, expected in RUNTIME_PACKAGES.items():
        if packages.get(name) != expected:
            raise RuntimeError(f"EAGLE qualification requires {name} {expected}; found {packages.get(name)}.")


def apply_output_policy(raw: list[int], cap: int, stop_ids: set[int]) -> dict:
    """Apply a strict cap and first-stop boundary while preserving raw evidence."""
    if cap < 1:
        raise ValueError("The output cap must be positive.")
    generated = []
    stop_reason = "producer_ended"
    for token in raw:
        if len(generated) == cap:
            stop_reason = "length"
            break
        generated.append(token)
        if token in stop_ids:
            stop_reason = "stop_token"
            break
    if stop_reason == "producer_ended" and len(generated) == cap:
        stop_reason = "length"
    returned = generated[:-1] if generated and generated[-1] in stop_ids else generated
    return {
        "raw_generated_token_ids": raw,
        "generated_token_ids": generated,
        "returned_token_ids": returned,
        "stop_reason": stop_reason,
        "raw_tokens_after_stop_or_cap": len(raw) - len(generated),
        "raw_boundary_compliant": len(raw) == len(generated) and stop_reason in {"stop_token", "length"},
        "raw_exceeds_output_cap": len(raw) > cap,
    }


def compare_output_records(ordinary: dict, speculative: dict) -> dict:
    """Separate the common output adapter's result from native producer stopping."""
    rows = []
    if len(ordinary["prompts"]) != len(speculative["prompts"]):
        raise ValueError("The ordinary and speculative records require the same prompt count.")
    for expected, actual in zip(ordinary["prompts"], speculative["prompts"]):
        left, right = expected["generated_token_ids"], actual["generated_token_ids"]
        first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
        if first is None and len(left) != len(right):
            first = min(len(left), len(right))
        rows.append({
            "prompt": expected["name"],
            "prompt_identity_matches": expected["name"] == actual["name"],
            "input_ids_match": expected["input_token_ids"] == actual["input_token_ids"],
            "output_caps_match": expected["max_new_tokens"] == actual["max_new_tokens"],
            "normalized_generated_token_ids_match": left == right,
            "normalized_returned_token_ids_match": expected["returned_token_ids"] == actual["returned_token_ids"],
            "normalized_stop_reason_matches": expected["stop_reason"] == actual["stop_reason"],
            "ordinary_raw_boundary_compliant": expected["raw_boundary_compliant"],
            "speculative_raw_boundary_compliant": actual["raw_boundary_compliant"],
            "first_normalized_generated_divergence_index": first,
            "raw_producer_ids_match": expected["raw_generated_token_ids"] == actual["raw_generated_token_ids"],
            "ordinary_raw_tokens_after_stop_or_cap": expected["raw_tokens_after_stop_or_cap"],
            "speculative_raw_tokens_after_stop_or_cap": actual["raw_tokens_after_stop_or_cap"],
        })
    adapted = bool(rows) and all(row[key] for row in rows for key in (
        "prompt_identity_matches", "input_ids_match", "output_caps_match",
        "normalized_generated_token_ids_match", "normalized_returned_token_ids_match",
        "normalized_stop_reason_matches", "ordinary_raw_boundary_compliant"))
    native = adapted and all(row["speculative_raw_boundary_compliant"] and row["raw_producer_ids_match"] for row in rows)
    return {
        "native_output_ready": native,
        "adapted_output_ready": adapted,
        "normalized_returned_token_identity": bool(rows) and all(row["normalized_returned_token_ids_match"] for row in rows),
        "normalized_stopping_equality": bool(rows) and all(row["normalized_stop_reason_matches"] for row in rows),
        "ordinary_raw_boundary_compliance": bool(rows) and all(row["ordinary_raw_boundary_compliant"] for row in rows),
        "speculative_raw_boundary_compliance": bool(rows) and all(row["speculative_raw_boundary_compliant"] for row in rows),
        "output_adapter": "canonical stop-token and strict output-cap clipping",
        "checks": rows,
        "exit_code": 0 if native else (3 if adapted else 2),
    }


def check_output_policy() -> None:
    """Exercise actual boundary and readiness decisions without loading a model."""
    stops = {128001, 128008, 128009}
    for stop in stops:
        exact = apply_output_policy([7, stop], 4, stops)
        assert exact["generated_token_ids"] == [7, stop]
        assert exact["returned_token_ids"] == [7]
        assert exact["stop_reason"] == "stop_token" and exact["raw_boundary_compliant"]
        extra = apply_output_policy([7, stop, 8], 4, stops)
        assert extra["raw_tokens_after_stop_or_cap"] == 1 and not extra["raw_boundary_compliant"]
        at_cap = apply_output_policy([7, stop], 2, stops)
        assert at_cap["stop_reason"] == "stop_token" and at_cap["raw_boundary_compliant"]
    assert apply_output_policy([7, 8, 9], 2, stops)["generated_token_ids"] == [7, 8]
    assert apply_output_policy([7, 8], 2, stops)["raw_boundary_compliant"]
    for short in ([], [7]):
        assert apply_output_policy(short, 2, stops)["stop_reason"] == "producer_ended"
        assert not apply_output_policy(short, 2, stops)["raw_boundary_compliant"]
    try:
        apply_output_policy([7], 0, stops)
    except ValueError:
        pass
    else:
        raise AssertionError("A zero output cap must fail.")

    def fixture(raw, cap=2):
        return {"prompts": [{"name": "policy-fixture", "max_new_tokens": cap,
                             "input_token_ids": [1], **apply_output_policy(raw, cap, stops)}]}
    identical = compare_output_records(fixture([7, 8]), fixture([7, 8]))
    assert identical["native_output_ready"] and identical["exit_code"] == 0
    for reference, raw, cap in [([7, 8], [7, 8, 9], 2), ([7, 128009], [7, 128009, 9], 4)]:
        clipped = compare_output_records(fixture(reference, cap), fixture(raw, cap))
        assert clipped["adapted_output_ready"] and not clipped["native_output_ready"] and clipped["exit_code"] == 3
    for reference, raw in [([7, 8], [7]), ([7, 8], [7, 9]), ([7], [7])]:
        failed = compare_output_records(fixture(reference), fixture(raw))
        assert not failed["adapted_output_ready"] and failed["exit_code"] == 2
    print("Output-policy checks pass: all canonical stops, exact caps, extra tokens, premature endings and separate readiness results.")


def write_record(path: Path, data: dict) -> None:
    # Each invocation uses a fresh evidence directory; preserve earlier results.
    with path.open("x") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def compare_tensor_values(saved, loaded, *, initial_float_dtype=None) -> bool:
    """Compare in bounded chunks after the released loader's dtype conversions."""
    import torch
    if saved.shape != loaded.shape:
        return False
    left, right = saved.detach().reshape(-1), loaded.detach().reshape(-1)
    for start in range(0, left.numel(), 1_048_576):
        expected = left[start:start + 1_048_576].to(device="cpu")
        if initial_float_dtype is not None and expected.is_floating_point():
            expected = expected.to(dtype=initial_float_dtype)
        expected = expected.to(dtype=loaded.dtype)
        actual = right[start:start + 1_048_576].to(device="cpu")
        if not torch.equal(actual, expected):
            return False
    return True


def verify_released_drafter(model, draft: Path, output: Path) -> dict:
    """Verify the checkpoint independently of the upstream strict=False load."""
    import torch
    config = json.loads((draft / "config.json").read_text())
    checkpoint = draft / "pytorch_model.bin"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(saved, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in saved.items()):
        raise RuntimeError("Expected the released EAGLE-3 tensor state dictionary.")
    final = model.ea_layer.state_dict()
    saved_keys, final_keys = set(saved), set(final)
    full_vocabulary = config["vocab_size"] == config["draft_vocab_size"]
    permitted_unused = {"d2t", "t2d"} if full_vocabulary else set()
    missing = final_keys - saved_keys
    target_embedding = {"embed_tokens.weight"} if "target_hidden_size" not in config else set()
    unexplained_missing = sorted(missing - target_embedding)
    unexplained_extra = sorted(saved_keys - final_keys - permitted_unused)
    shape_failures = []
    dtype_failures = []
    value_failures = []
    verified = []
    for name in sorted(saved_keys & final_keys):
        expected, actual = saved[name], final[name]
        if expected.shape != actual.shape:
            shape_failures.append({"key": name, "saved_shape": list(expected.shape), "loaded_shape": list(actual.shape)})
            continue
        expected_dtype = torch.bfloat16 if expected.is_floating_point() else expected.dtype
        if actual.dtype != expected_dtype:
            dtype_failures.append({"key": name, "expected_dtype": str(expected_dtype), "loaded_dtype": str(actual.dtype)})
        # cnets.Model constructs floating parameters in FP32, then EaModel casts
        # the loaded layer to the target dtype. Integer vocabulary buffers retain
        # their registered integer/Boolean dtype.
        matches = compare_tensor_values(expected, actual, initial_float_dtype=torch.float32)
        if not matches:
            value_failures.append(name)
        verified.append({"key": name, "shape": list(actual.shape),
                         "saved_dtype": str(expected.dtype), "loaded_dtype": str(actual.dtype),
                         "values_match_after_loader_conversion": matches})
    embedding_check = None
    if "embed_tokens.weight" in missing & target_embedding:
        expected = model.base_model.model.embed_tokens.weight
        actual = final["embed_tokens.weight"]
        matches = actual.dtype == torch.bfloat16 and compare_tensor_values(expected, actual, initial_float_dtype=torch.float32)
        embedding_check = {"source": "canonical Meta target model.embed_tokens.weight",
                           "destination": "ea_layer.embed_tokens.weight",
                           "shape": list(actual.shape), "values_match_after_loader_conversion": matches}
        if not matches:
            value_failures.append("target-sourced embed_tokens.weight")
    result = {
        "drafter_load_checks_pass": not (unexplained_missing or unexplained_extra or shape_failures or dtype_failures or value_failures),
        "source_revision": SOURCE_REVISION, "draft_revision": DRAFT_REVISION,
        "checkpoint_filename": checkpoint.name,
        "saved_key_count": len(saved_keys), "final_state_key_count": len(final_keys),
        "unexplained_missing_keys": unexplained_missing, "unexplained_unused_saved_keys": unexplained_extra,
        "shape_failures": shape_failures, "dtype_failures": dtype_failures, "value_failures": value_failures,
        "checkpoint_tensors": verified, "target_embedding_check": embedding_check,
        "intended_checkpoint_key_renames": [],
        "intended_unused_saved_keys": sorted((saved_keys - final_keys) & permitted_unused),
        "loader_rules": [
            "cnets.Model loads embed_tokens.weight from the target when target_hidden_size is absent; a saved draft embedding would subsequently overwrite it.",
            "EaModel deletes d2t/t2d only when draft_vocab_size equals vocab_size. The selected checkpoint has a reduced vocabulary, so both saved mapping buffers must match.",
            "RoPE caches use nonpersistent buffers; tree initialization creates runtime attributes. These values are outside the saved state_dict.",
            "Checkpoint key names are loaded directly. Floating parameters pass through the constructed FP32 module and the target's BF16 dtype; vocabulary mapping buffers preserve integer/Boolean semantics.",
        ],
        "source_links": [
            f"https://github.com/SafeAILab/EAGLE/blob/{SOURCE_REVISION}/eagle/model/ea_model.py#L74",
            f"https://github.com/SafeAILab/EAGLE/blob/{SOURCE_REVISION}/eagle/model/cnets.py#L479",
        ],
        "scope": "Saved-key, tensor-shape and exact post-conversion value checks; token output identity has separate evidence.",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
    }
    write_record(output / "drafter-load.json", result)
    if not result["drafter_load_checks_pass"]:
        raise RuntimeError("EAGLE-3 drafter loading verification failed; inspect drafter-load.json.")
    print(f"Verified {len(verified)} saved drafter tensors and the target-sourced embedding.", flush=True)
    return result


def execute_stage(stage: str, output: Path, target: Path, draft: Path) -> None:
    packages = {name: metadata.version(name) for name in RUNTIME_PACKAGES}
    check_runtime_versions(tuple(sys.version_info[:3]), packages)
    import torch
    from transformers import AutoTokenizer
    from eagle.model.ea_model import EaModel
    from eagle.model.modeling_llama_kv import LlamaForCausalLM
    from eagle.model.kv_cache import initialize_past_key_values

    if torch.version.cuda != "12.4":
        raise RuntimeError("EAGLE qualification requires the prepared CUDA 12.4 PyTorch wheel.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Allocate exactly one visible GPU for this check.")
    torch.manual_seed(0)
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("The reviewed drafter construction requires the default FP32 dtype.")
    tokenizer = AutoTokenizer.from_pretrained(target, use_fast=False, local_files_only=True)
    generation = json.loads((target / "generation_config.json").read_text())
    values = generation["eos_token_id"]
    stop_ids = set(values if isinstance(values, list) else [values])
    records = []
    drafter_load = None
    loading = dict(torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
                   low_cpu_mem_usage=True, local_files_only=True)
    print(f"Loading {stage} in a separate process.", flush=True)
    if stage == "target":
        model = LlamaForCausalLM.from_pretrained(str(target), **loading).eval()
        model.model.tree_mask = None
        cache, cache_data, current_length = initialize_past_key_values(model, max_length=512)
    else:
        model = EaModel.from_pretrained(
            use_eagle3=True, base_model_path=str(target), ea_model_path=str(draft),
            total_token=60, depth=7, top_k=10, threshold=1.0, **loading,
        ).eval()
        drafter_load = verify_released_drafter(model, draft, output)
    with torch.inference_mode():
        for name, prompt, cap in PROMPTS:
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, return_tensors="pt",
            ).to("cuda:0")
            if input_ids.shape[-1] + cap + 80 >= 512:
                raise RuntimeError("The setup prompt exceeds the fixed cache capacity.")
            if stage == "target":
                current_length.zero_()
                output_state = model(input_ids, past_key_values=cache, use_cache=True)
                raw = []
                for position in range(cap):
                    token = int(output_state.logits[0, -1].argmax().item())
                    raw.append(token)
                    if token in stop_ids or position == cap - 1:
                        break
                    next_ids = torch.tensor([[token]], device="cuda:0")
                    output_state = model(next_ids, past_key_values=cache, use_cache=True)
            else:
                full_output = model.eagenerate(
                    input_ids.clone(), temperature=0.0, max_new_tokens=cap,
                    max_length=512, is_llama3=True,
                )
                if not torch.equal(full_output[:, :input_ids.shape[-1]], input_ids):
                    raise RuntimeError("The released producer changed the prompt prefix.")
                raw = full_output[0, input_ids.shape[-1]:].tolist()
            record = {"name": name, "prompt": prompt, "max_new_tokens": cap,
                      "input_token_ids": input_ids[0].tolist(),
                      **apply_output_policy(raw, cap, stop_ids)}
            record["returned_text"] = tokenizer.decode(record["returned_token_ids"], skip_special_tokens=True)
            records.append(record)
            print(f"{stage}: {name}, {len(raw)} raw tokens, {record['stop_reason']}.", flush=True)
    torch.cuda.synchronize()
    write_record(output / f"{stage}.json", {
        "stage": stage, "scope": "Setup loading and four greedy output checks; no performance measurement.",
        "source_revision": SOURCE_REVISION,
        "target_revision": TARGET_REVISION, "draft_revision": DRAFT_REVISION if stage == "eagle3" else None,
        "drafter_load_checks_pass": drafter_load["drafter_load_checks_pass"] if drafter_load else None,
        "target_path": str(target), "draft_path": str(draft) if stage == "eagle3" else None,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "python": sys.version, "transformers": packages["transformers"], "accelerate": packages["accelerate"],
        "cuda_runtime": torch.version.cuda, "dtype": "bfloat16",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "canonical_stop_token_ids": sorted(stop_ids), "prompts": records,
        "ordinary_implementation": "Pinned EAGLE KVLlamaForCausalLM with one-token greedy decoding and no loaded drafter.",
        "output_policy": "Keep the first canonical stop token or strict output cap; exclude a terminal stop token from returned text/tokens. Preserve all producer tokens separately.",
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check-policy", action="store_true", help="Run pure boundary checks without model loading.")
    parser.add_argument("--stage", choices=("target", "eagle3"))
    args = parser.parse_args()
    if args.check_policy:
        check_output_policy()
        return 0
    if args.output_dir is None:
        parser.error("--output-dir is required for model qualification.")
    from download_checkpoints import require_compute_allocation, require_scratch_cache
    require_compute_allocation()
    home = require_scratch_cache()
    root = Path(os.environ["HPML_ROOT"]).resolve()
    output = args.output_dir.resolve()
    source = root / "source/hpml-project/external/eagle"
    actual_source = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    source_changes = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain=v1"], text=True).strip()
    if actual_source != SOURCE_REVISION or source_changes:
        raise RuntimeError("Use the clean reviewed EAGLE source revision.")
    sys.path.insert(0, str(source))
    if root / "runs" not in output.parents:
        raise RuntimeError("Use a fresh result directory under HPML_ROOT/runs.")
    os.umask(0o077)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    target = home / "hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots" / TARGET_REVISION
    draft = home / "hub/models--yuhuili--EAGLE3-LLaMA3.1-Instruct-8B/snapshots" / DRAFT_REVISION
    if not (home / "hpml-provenance/target" / f"{TARGET_REVISION}.json").is_file():
        raise RuntimeError("Complete and verify target staging first.")
    if args.stage:
        execute_stage(args.stage, output, target, draft)
        return 0
    output.mkdir(parents=True, exist_ok=False)
    for stage in ("target", "eagle3"):
        subprocess.run([sys.executable, __file__, "--output-dir", str(output), "--stage", stage], check=True)
    ordinary = json.loads((output / "target.json").read_text())
    speculative = json.loads((output / "eagle3.json").read_text())
    result = compare_output_records(ordinary, speculative)
    result["drafter_load_checks_pass"] = speculative["drafter_load_checks_pass"]
    result.update({
        "scope": "Four setup prompts establish these observed behaviors. The common baseline comparison requires broader correctness, streaming and performance measurements.",
        "stopping_note": "The unchanged released EAGLE producer checks max_new_tokens after a generation cycle. Native readiness requires raw boundary compliance. Adapted readiness identifies the explicit output-clipping adapter. Future timing must include all executed work.",
    })
    write_record(output / "comparison.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
