#!/usr/bin/env python3
"""Reproduce the team's saved pilot while recording data, vocabulary, and model provenance."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import time
import unicodedata

from team_peagle_trial import PILOT, NOTEBOOK_SHA256, attach_vocabulary, load_runtime, sha256, training_rows

DATASET_REVISION = "8049631c405ae6576f93f445c6b8166f76f5505a"


def normalized_text(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def overlap_report(prompt_texts, evaluation):
    selected = {}
    for row in evaluation["prompts"]:
        selected.setdefault(normalized_text(row["messages"][0]["content"]), []).append(row)
    matches = []
    for row in prompt_texts:
        normalized = normalized_text(row["text"])
        for candidate in selected.get(normalized, []):
            matches.append({
                "selection_index": row["selection_index"], "split": row["split"],
                "evaluation_prompt_id": candidate["id"], "evaluation_benchmark": candidate["benchmark"],
                "normalized_text_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            })
    return {
        "normalization": "Unicode NFC, case folding, and whitespace collapse.",
        "training_prompts": sum(r["split"] == "training" for r in prompt_texts),
        "held_out_prompts": sum(r["split"] == "held_out" for r in prompt_texts),
        "evaluation_prompts": len(evaluation["prompts"]), "matches": matches,
    }


def recover_prompt_texts(namespace, rows):
    tokenizer = namespace["tok"]
    marker = "HPML_UNIQUE_PROMPT_CONTENT_MARKER_7c3d"
    template = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}], add_generation_prompt=True, tokenize=False)
    if template.count(marker) != 1:
        raise ValueError("The pinned chat template must retain the content marker exactly once.")
    prefix, suffix = template.split(marker)
    records = []
    for index, row in enumerate(rows):
        decoded = tokenizer.decode(row["prompt"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if not decoded.startswith(prefix) or not decoded.endswith(suffix):
            raise ValueError("A training prompt differs from the pinned single-user chat template.")
        end = len(decoded) - len(suffix) if suffix else len(decoded)
        content = decoded[len(prefix):end]
        if namespace["chat_ids"](content) != row["prompt"]:
            raise ValueError("Recovered training text fails the exact token-ID roundtrip.")
        records.append({
            "selection_index": index, "split": "training" if index < PILOT["N_TRAIN"] else "held_out",
            "text": content, "prompt_token_ids_sha256": hashlib.sha256(json.dumps(row["prompt"]).encode()).hexdigest(),
        })
    return records


def read_evaluation(path):
    from trial_recorder import INPUT_SHA256, TARGET_REVISION, BENCHMARKS
    evaluation = json.loads(Path(path).read_text())
    if (evaluation["input_sha256"] != INPUT_SHA256 or evaluation["target_revision"] != TARGET_REVISION
            or Counter(r["benchmark"] for r in evaluation["prompts"]) != Counter({name: 3 for name in BENCHMARKS})):
        raise ValueError("Use the shared prepared 30-prompt evaluation.")
    for row in evaluation["prompts"]:
        if (len(row["messages"]) != 1 or row["messages"][0]["role"] != "user"
                or hashlib.sha256(row["messages"][0]["content"].encode()).hexdigest() != row["promptSha256"]):
            raise ValueError("An evaluation prompt differs from its recorded text.")
    return evaluation


def write_manifest(path, value):
    temporary = path.with_suffix(".partial")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION, help="The full UltraChat dataset commit SHA.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.dataset_revision):
        raise ValueError("Pin the UltraChat dataset to its complete commit SHA.")
    if importlib.metadata.version("datasets") != "3.6.0":
        raise ValueError("The saved notebook requires datasets==3.6.0.")
    evaluation = read_evaluation(args.prepared)
    os.umask(0o077)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["WANDB_MODE"] = "disabled"
    if args.output.exists() and not args.resume:
        raise ValueError("Choose a new output directory or explicitly resume its recorded pilot.")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "training-manifest.json"
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if args.resume and prior is None:
        raise ValueError("Resuming requires the saved training manifest.")
    namespace, provenance = load_runtime(args.notebook, args.hf_home)
    identity = {**provenance, "settings": PILOT, "dataset": "HuggingFaceH4/ultrachat_200k",
                "dataset_revision": args.dataset_revision, "dataset_split": "train_sft",
                "dataset_shuffle_seed": 0, "torch_seed_per_width": 0,
                "initialization_scope": "Each width resets seed zero. Different matrix sizes consume different random draws.",
                "epoch_order_seed": 0, "anchor_seed": "update * ACCUM + micro",
                "data_generation_batch_size": 32,
                "evaluation_input_sha256": evaluation["input_sha256"],
                "evaluation_prepared_sha256": sha256(args.prepared),
                "training_runner_sha256": sha256(__file__)}
    if prior:
        for key in ("target_revision", "notebook_sha256", "settings", "dataset_revision", "training_runner_sha256",
                    "torch_version", "transformers_version", "evaluation_prepared_sha256"):
            if prior[key] != identity[key]:
                raise ValueError("Resume provenance differs in " + key)
        manifest = prior
    else:
        manifest = {**identity, "status": "preparing_data", "created_unix_ns": time.time_ns(),
                    "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
                    "classification": "Local reproduction of the team's P-EAGLE-style FFN-width pilot",
                    "checkpoints": {}}
        write_manifest(manifest_path, manifest)
    import datasets
    import wandb

    def pinned_dataset(name, *positional, **kwargs):
        if name != identity["dataset"] or kwargs.get("split") != "train_sft":
            raise ValueError("The pilot requested an unexpected training dataset.")
        result = datasets.load_dataset(name, *positional, revision=args.dataset_revision, **kwargs)
        manifest["dataset_fingerprint"] = result._fingerprint
        write_manifest(manifest_path, manifest)
        return result

    data_path = args.output / "train_data.jsonl"
    namespace.update(OUT=str(args.output.resolve()), DATA=str(data_path.resolve()),
                     USE_WANDB=False, wandb=wandb, load_dataset=pinned_dataset)
    count = PILOT["N_TRAIN"] + PILOT["N_VAL"]
    if len(namespace["load_rows"](str(data_path))) < count:
        prompts = namespace["training_prompts"](count)
        if len(prompts) != count:
            raise ValueError("The fixed source selection supplies too few short prompts.")
        namespace["write_answers"](prompts)
    rows = training_rows(data_path, count)
    if len(rows) != count:
        raise ValueError("The pilot requires exactly 200 training and 50 held-out examples.")
    if prior and prior.get("training_data_sha256") and prior["training_data_sha256"] != sha256(data_path):
        raise ValueError("The saved training examples changed after preparation.")
    prompt_texts = recover_prompt_texts(namespace, rows)
    overlap = overlap_report(prompt_texts, evaluation)
    overlap.update(evaluation_input_sha256=evaluation["input_sha256"],
                   evaluation_prepared_sha256=sha256(args.prepared), training_data_sha256=sha256(data_path))
    write_manifest(args.output / "training-prompt-texts.json", {
        "text_origin": "Recovered from the pinned chat template with exact token-ID roundtrip checks.",
        "prompts": prompt_texts,
    })
    write_manifest(args.output / "evaluation-overlap.json", overlap)
    manifest["overlap_report_sha256"] = sha256(args.output / "evaluation-overlap.json")
    manifest["overlap_match_count"] = len(overlap["matches"])
    if overlap["matches"]:
        manifest["status"] = "overlap_detected"
        write_manifest(manifest_path, manifest)
        raise ValueError("Training or held-out prompts overlap the fixed evaluation. Inspect evaluation-overlap.json before training.")
    namespace.update(data=rows[:PILOT["N_TRAIN"]], val=rows[PILOT["N_TRAIN"]:])
    keep = attach_vocabulary(namespace, namespace["data"])
    vocabulary_path = args.output / "draft_vocab.json"
    vocabulary = {"target_token_ids": keep, "target_revision": provenance["target_revision"],
                  "training_data_sha256": sha256(data_path), "training_examples": PILOT["N_TRAIN"]}
    if vocabulary_path.exists() and json.loads(vocabulary_path.read_text()) != vocabulary:
        raise ValueError("The saved draft vocabulary changed.")
    write_manifest(vocabulary_path, vocabulary)
    manifest.update(status="training", training_data_sha256=sha256(data_path),
                    vocabulary_sha256=sha256(vocabulary_path), vocabulary_size=len(keep),
                    training_examples=200, held_out_examples=50, expected_updates=50)
    write_manifest(manifest_path, manifest)
    torch = namespace["torch"]
    for width in PILOT["WIDTHS"]:
        print(f"Training the saved pilot at feed-forward width {width:.2f}.", flush=True)
        model = namespace["train"](width)
        del model
        torch.cuda.empty_cache()
        checkpoint = args.output / f"drafter_{int(width * 100)}.pt"
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        if state["update"] != 50:
            raise ValueError("The drafter did not finish the saved pilot's 50 updates.")
        del state
        manifest["checkpoints"][str(int(width * 100))] = {
            "path": checkpoint.name, "sha256": sha256(checkpoint), "updates": 50,
            "feed_forward_width": width, "ffn_dimension": int(round(14336 * width / 128)) * 128,
        }
        write_manifest(manifest_path, manifest)
    manifest.update(status="complete", completed_unix_ns=time.time_ns())
    write_manifest(manifest_path, manifest)
    print(f"Saved all three pilot checkpoints and their provenance in {args.output}.", flush=True)


if __name__ == "__main__":
    main()
