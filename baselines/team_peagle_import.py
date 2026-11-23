#!/usr/bin/env python3
"""Verify shared original pilot checkpoints and record their available provenance."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import socket
import sys
import time

from team_peagle_trial import NOTEBOOK_COMMIT, NOTEBOOK_SHA256, PILOT, draft_vocabulary, sha256, training_rows
from team_peagle_train import overlap_report, read_evaluation, recover_prompt_texts, write_manifest


def expected_shapes(width):
    ffn = int(round(14336 * width / 128)) * 128
    shapes = {"norm_h.weight": (4096,), "norm_e.weight": (4096,), "fc.weight": (4096, 8192),
              "h_shared": (4096,), "e_mask": (4096,), "norm_out.weight": (4096,)}
    for index in range(4):
        prefix = f"layers.{index}."
        shapes.update({
            prefix + "norm1.weight": (4096,), prefix + "norm2.weight": (4096,),
            prefix + "q.weight": (4096, 4096), prefix + "k.weight": (1024, 4096),
            prefix + "v.weight": (1024, 4096), prefix + "o.weight": (4096, 4096),
            prefix + "gate.weight": (ffn, 4096), prefix + "up.weight": (ffn, 4096),
            prefix + "down.weight": (4096, ffn),
        })
    return shapes


def validate_state(state, width):
    if not isinstance(state, dict) or type(state.get("update")) is not int or state["update"] != 50:
        raise ValueError("The supplied checkpoint must record the saved pilot's 50 updates.")
    parameters = state.get("drafter")
    shapes = expected_shapes(width)
    if not isinstance(parameters, dict) or set(parameters) != set(shapes):
        raise ValueError("The checkpoint parameter names differ from the pinned four-layer drafter.")
    for name, shape in shapes.items():
        value = parameters[name]
        if tuple(value.shape) != shape or not value.is_floating_point():
            raise ValueError("The checkpoint parameter shape or type differs: " + name)
    return {
        "updates": state["update"], "feed_forward_width": width,
        "ffn_dimension": int(round(14336 * width / 128)) * 128,
        "parameter_count": sum(math.prod(shape) for shape in shapes.values()),
        "parameter_shapes": {name: list(shape) for name, shape in shapes.items()},
        "parameter_dtypes": dict(Counter(str(value.dtype) for value in parameters.values())),
        "optimizer_state_present": "opt" in state,
    }


def reported_settings(path):
    if path is None:
        return None, []
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Reported settings must be a JSON object.")
    allowed = {
        "PILOT", "TARGET", "K", "WIDTHS", "DRAFT_VOCAB", "N_TRAIN", "N_VAL", "EPOCHS",
        "MAX_PROMPT", "MAX_ANSWER", "ANCHORS", "ACCUM", "LR", "WARMUP", "N_EVAL", "MAX_NEW",
        "N_CHECK", "N_TASK", "ROPE_BASE", "D", "FFN", "HEADS", "KV_HEADS", "HEAD_DIM", "EOS", "PAD",
        "target_revision", "dataset_revision", "torch_version", "transformers_version", "notebook_commit", "seed",
    }
    return {key: item for key, item in value.items() if key in allowed}, sorted(set(value) - allowed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="The shared directory containing the original files.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, help="An original JSON list or target_token_ids mapping, when available.")
    parser.add_argument("--reported-settings", type=Path, help="The teammate's supplied JSON settings, retained with attribution.")
    parser.add_argument("--copy-files", action="store_true", help="Copy the source files into the import directory.")
    args = parser.parse_args()
    settings, unrecorded_settings = reported_settings(args.reported_settings)
    from download_checkpoints import require_compute_allocation
    from qualify_hspec import require_snapshot, TARGET_ID
    from trial_recorder import TARGET_REVISION
    require_compute_allocation()
    import torch
    import transformers
    from transformers import AutoTokenizer
    if transformers.__version__ != "4.57.6":
        raise ValueError("Use the isolated team runtime with Transformers 4.57.6.")
    if args.output.exists():
        raise FileExistsError("Use a new import directory so existing evidence remains intact.")
    source = args.source.resolve(strict=True)
    data_path = source / "train_data.jsonl"
    rows = training_rows(data_path, 200)
    evaluation = read_evaluation(args.prepared)
    snapshot, target_provenance = require_snapshot(Path(args.hf_home), "target", TARGET_ID, TARGET_REVISION)
    vocab_size = json.loads((snapshot / "config.json").read_text())["vocab_size"]
    if any(token >= vocab_size for row in rows for token in row["prompt"] + row["answer"]):
        raise ValueError("The supplied data contains IDs outside the evaluation target vocabulary.")
    generation = json.loads((snapshot / "generation_config.json").read_text())
    stops = generation["eos_token_id"]
    stops = stops if isinstance(stops, list) else [stops]
    keep = draft_vocabulary(rows[:200], stops)
    if args.vocabulary:
        original = json.loads(args.vocabulary.read_text())
        original = original if isinstance(original, list) else original["target_token_ids"]
        if not isinstance(original, list) or any(type(token) is not int for token in original) or original != keep:
            raise ValueError("The supplied original vocabulary differs from the reconstructed token order.")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=False)
    namespace = {
        "tok": tokenizer,
        "chat_ids": lambda content: tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True),
    }
    texts = recover_prompt_texts(namespace, rows[:250])
    overlap = overlap_report(texts, evaluation)
    overlap.update(evaluation_prepared_sha256=sha256(args.prepared),
                   evaluation_input_sha256=evaluation["input_sha256"], training_data_sha256=sha256(data_path))
    checkpoints = {}
    for width in PILOT["WIDTHS"]:
        path = source / f"drafter_{int(width * 100)}.pt"
        print(f"Checking the shared {int(width * 100)}% checkpoint.", flush=True)
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        record = validate_state(state, width)
        for name, value in state["drafter"].items():
            flat = value.reshape(-1)
            for start in range(0, flat.numel(), 1_048_576):
                if not torch.isfinite(flat[start:start + 1_048_576]).all().item():
                    raise ValueError("The checkpoint contains nonfinite weights: " + name)
        del state
        record.update(path=path.name, source_path=str(path), sha256=sha256(path),
                      size_bytes=path.stat().st_size, finite_weights_verified=True)
        checkpoints[str(int(width * 100))] = record
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=False)
    for filename in ("train_data.jsonl", "drafter_100.pt", "drafter_75.pt", "drafter_50.pt"):
        destination = args.output / filename
        if args.copy_files:
            shutil.copyfile(source / filename, destination)
        else:
            destination.symlink_to(source / filename)
    vocabulary = {
        "target_token_ids": keep, "reconstruction_rows": 200,
        "reconstruction_stop_token_ids": stops, "evaluation_target_revision": TARGET_REVISION,
        "original_mapping_supplied": args.vocabulary is not None,
        "original_mapping_sha256": sha256(args.vocabulary) if args.vocabulary else None,
    }
    write_manifest(args.output / "draft_vocab.json", vocabulary)
    write_manifest(args.output / "training-prompt-texts.json", {
        "text_origin": "Recovered from supplied prompt tokens with an exact chat-template roundtrip.",
        "prompts": texts,
    })
    write_manifest(args.output / "evaluation-overlap.json", overlap)
    manifest = {
        "artifact_kind": "team_original_checkpoint_import",
        "status": "overlap_detected" if overlap["matches"] else "complete",
        "source_directory": str(source), "file_mode": "copies" if args.copy_files else "references",
        "created_unix_ns": time.time_ns(), "hostname": socket.gethostname(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "importer_sha256": sha256(__file__),
        "inspection_runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                               "python": sys.version},
        "evaluation_notebook_commit": NOTEBOOK_COMMIT, "evaluation_notebook_sha256": NOTEBOOK_SHA256,
        "evaluation_target_revision": TARGET_REVISION, "evaluation_target_provenance": target_provenance,
        "evaluation_prepared_sha256": sha256(args.prepared), "evaluation_input_sha256": evaluation["input_sha256"],
        "training_data_sha256": sha256(data_path), "supplied_data_rows": len(rows),
        "vocabulary_sha256": sha256(args.output / "draft_vocab.json"),
        "vocabulary_reconstruction_rows": 200, "vocabulary_size": len(keep),
        "original_vocabulary_supplied": args.vocabulary is not None,
        "overlap_report_sha256": sha256(args.output / "evaluation-overlap.json"),
        "overlap_match_count": len(overlap["matches"]), "checkpoints": checkpoints,
        "training_target_revision": None, "training_notebook_commit": None,
        "training_dataset_revision": None, "training_torch_version": None,
        "training_transformers_version": None, "training_seed": None,
        "training_provenance": "The checkpoints record weights and update counts. Additional training provenance remains unverified.",
        "reported_settings": settings, "unrecorded_settings_names": unrecorded_settings,
        "reported_settings_sha256": sha256(args.reported_settings) if args.reported_settings else None,
    }
    write_manifest(args.output / "import-manifest.json", manifest)
    if overlap["matches"]:
        raise ValueError("The supplied training or held-out prompts overlap the fixed evaluation. Inspect evaluation-overlap.json.")
    print(f"Verified all three original pilot checkpoints in {args.output}.", flush=True)


if __name__ == "__main__":
    main()
