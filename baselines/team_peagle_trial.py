#!/usr/bin/env python3
"""Evaluate the pinned team notebook with the shared streaming trial procedure."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
from urllib.parse import urlparse

NOTEBOOK_COMMIT = "64b24c42f1859e25b0ffb69293e7e9f793a697d3"
NOTEBOOK_SHA256 = "401a612174145e4eb55e324223045804bc7840514944152d2f0ecdb5cb642490"
DEFINITIONS = {
    "target_step", "chat_ids", "rope", "RMSNorm", "Layer", "Drafter",
    "logits_of", "training_view", "new_cache", "cache_len", "crop", "draft",
    "load_rows", "training_prompts", "write_answers", "anchors_for", "lr_at",
    "train", "sync", "standard", "speculative",
}
CONFIGURATIONS = {
    "team-ar": None, "team-ffn-100": 1.0, "team-ffn-75": .75, "team-ffn-50": .5,
}
PILOT = {"PILOT": True, "K": 8, "WIDTHS": [1.0, .75, .5], "DRAFT_VOCAB": 32000,
         "N_TRAIN": 200, "N_VAL": 50, "EPOCHS": 1, "MAX_PROMPT": 256,
         "MAX_ANSWER": 512, "ANCHORS": 64, "ACCUM": 4, "LR": 3e-4, "WARMUP": 50}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def notebook_definitions(path, expected_sha256=NOTEBOOK_SHA256):
    """Select definitions while excluding cell execution, training loops, and login."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Use the notebook bytes from the recorded origin/main commit.")
    notebook = json.loads(raw)
    nodes, found = [], set()
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        module = ast.parse("".join(cell["source"]), filename=f"{path}:cell-{index}")
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in DEFINITIONS:
                if node.name in found:
                    raise ValueError("Duplicate notebook definition: " + node.name)
                nodes.append(node)
                found.add(node.name)
    if found != DEFINITIONS:
        raise ValueError("Missing notebook definitions: " + ", ".join(sorted(DEFINITIONS - found)))
    return compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec")


def export_notebook(path):
    raw = subprocess.check_output(
        ["git", "show", f"{NOTEBOOK_COMMIT}:peagle_ffn.ipynb"],
        cwd=Path(__file__).resolve().parents[1])
    if hashlib.sha256(raw).hexdigest() != NOTEBOOK_SHA256:
        raise ValueError("The requested Git blob differs from the pinned notebook.")
    with Path(path).open("xb") as handle:
        handle.write(raw)


def training_rows(path, count):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) < count:
        raise ValueError("The saved training data contains too few examples.")
    for row in rows:
        for field in ("prompt", "answer"):
            ids = row.get(field)
            if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
                raise ValueError("Training data must preserve nonempty original token-ID arrays.")
    return rows


def draft_vocabulary(rows, stops, count=32000):
    """Reproduce the notebook's frequency ordering and EOS append order."""
    counts = Counter(token for row in rows for token in row["answer"])
    if not counts:
        raise ValueError("Training answers contain no tokens.")
    keep = [token for token, _ in counts.most_common(count)]
    keep += [token for token in set(stops) if token not in keep]
    return keep


def retained_chunk(tokens, remaining, stops):
    """Apply the shared length limit and first-EOS boundary before transmission."""
    if remaining < 1:
        return [], "length", None
    result = list(tokens[:remaining])
    for index, token in enumerate(result):
        if token in stops:
            return result[:index + 1], "stop", token
    return result, "length" if len(result) == remaining else None, None


def validate_runtime_versions(torch_version, cuda_version, transformers_version, python_version):
    if (str(torch_version), cuda_version, transformers_version, tuple(python_version[:2])) != (
            "2.6.0+cu124", "12.4", "4.57.6", (3, 10)):
        raise ValueError("Use the pinned team Python 3.10, Torch 2.6.0+cu124, CUDA 12.4, and Transformers 4.57.6 runtime.")


def load_runtime(notebook, hf_home):
    from download_checkpoints import require_compute_allocation
    from qualify_hspec import require_snapshot, TARGET_ID
    from trial_recorder import TARGET_REVISION
    require_compute_allocation()
    code = notebook_definitions(notebook)
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    validate_runtime_versions(torch.__version__, torch.version.cuda, transformers.__version__, sys.version_info)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one allocated CUDA GPU.")
    snapshot, provenance = require_snapshot(Path(hf_home), "target", TARGET_ID, TARGET_REVISION)
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=False)
    target = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=torch.bfloat16, local_files_only=True).to("cuda").eval()
    target.requires_grad_(False)
    cfg = target.config
    if (cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.num_key_value_heads) != (4096, 14336, 32, 8):
        raise ValueError("The target architecture differs from the team's Llama-3.1-8B target.")
    eos = target.generation_config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    pad = tok.convert_tokens_to_ids("<|finetune_right_pad_id|>")
    if pad is None or pad == tok.unk_token_id:
        pad = tok.eos_token_id
    namespace = {
        **PILOT, "torch": torch, "nn": torch.nn, "F": torch.nn.functional, "math": math,
        "random": random, "time": time, "Counter": Counter, "os": os, "json": json,
        "DEV": "cuda", "TARGET": TARGET_ID, "target": target, "tok": tok, "cfg": cfg,
        "D": 4096, "FFN": 14336, "HEADS": 32, "KV_HEADS": 8, "HEAD_DIM": 128,
        "ROPE_BASE": 500000.0, "EMBED": target.get_input_embeddings(), "EOS": stops, "PAD": pad,
    }
    exec(code, namespace)
    provenance = {
        "target": provenance, "target_revision": TARGET_REVISION,
        "notebook_commit": NOTEBOOK_COMMIT, "notebook_sha256": NOTEBOOK_SHA256,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda, "python_version": sys.version,
        "target_attention_implementation": cfg._attn_implementation,
        "gpu_name": torch.cuda.get_device_name(0),
    }
    return namespace, provenance


def attach_vocabulary(namespace, rows):
    torch, target = namespace["torch"], namespace["target"]
    keep = draft_vocabulary(rows, namespace["EOS"], namespace["DRAFT_VOCAB"])
    if any(t >= target.config.vocab_size for t in keep):
        raise ValueError("The saved training vocabulary exceeds the target vocabulary.")
    d2t = torch.tensor(keep, device="cuda")
    t2d = torch.full((target.config.vocab_size,), -1, dtype=torch.long, device="cuda")
    t2d[d2t] = torch.arange(len(keep), device="cuda")
    namespace.update(D2T=d2t, T2D=t2d, HEAD=target.lm_head.weight[d2t].detach().clone())
    return keep


class TeamEngine:
    def __init__(self, args):
        self.ns, self.provenance = load_runtime(args.notebook, args.hf_home)
        self.configuration = args.configuration
        self.width = CONFIGURATIONS[self.configuration]
        self.diagnostic_dir = Path(args.diagnostic_dir).resolve() if args.diagnostic_dir else None
        if self.diagnostic_dir:
            os.environ["HPML_DIAGNOSTIC_DIR"] = str(self.diagnostic_dir)
            os.environ["HPML_CONFIGURATION"] = self.configuration
            os.environ["HPML_GPU_UUID"] = args.gpu_uuid
        elif os.environ.get("HPML_DIAGNOSTIC_DIR"):
            raise ValueError("Clear HPML_DIAGNOSTIC_DIR for the primary timing run.")
        self.drafter = None
        if self.width is not None:
            directory = Path(args.checkpoint_dir)
            data_path = directory / "train_data.jsonl"
            rows = training_rows(data_path, PILOT["N_TRAIN"])
            keep = attach_vocabulary(self.ns, rows[:PILOT["N_TRAIN"]])
            vocabulary = directory / "draft_vocab.json"
            if vocabulary.exists() and json.loads(vocabulary.read_text())["target_token_ids"] != keep:
                raise ValueError("The saved draft vocabulary differs from reconstruction.")
            manifest_path = directory / "training-manifest.json"
            import_path = directory / "import-manifest.json"
            if manifest_path.exists() and import_path.exists():
                raise ValueError("Use one checkpoint provenance record per artifact directory.")
            original = import_path.exists()
            if original:
                manifest = json.loads(import_path.read_text())
                if (manifest["artifact_kind"] != "team_original_checkpoint_import"
                        or manifest["evaluation_notebook_sha256"] != NOTEBOOK_SHA256
                        or manifest["evaluation_target_revision"] != self.provenance["target_revision"]):
                    raise ValueError("The imported checkpoint evaluation provenance differs.")
            elif manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if manifest["notebook_sha256"] != NOTEBOOK_SHA256 or manifest["target_revision"] != self.provenance["target_revision"]:
                    raise ValueError("Training and evaluation provenance differ.")
                if manifest["settings"] != PILOT:
                    raise ValueError("The training manifest describes another training procedure.")
                for field in ("torch_version", "transformers_version", "cuda_version", "target_attention_implementation"):
                    if manifest[field] != self.provenance[field]:
                        raise ValueError("Training and evaluation runtime differ in " + field)
            else:
                raise ValueError("Import the supplied checkpoints or complete the recorded local pilot first.")
            if manifest["status"] != "complete" or manifest["training_data_sha256"] != sha256(data_path):
                raise ValueError("The pilot artifact must be complete and retain its verified training data.")
            if manifest["vocabulary_sha256"] != sha256(vocabulary):
                raise ValueError("The saved vocabulary differs from its artifact manifest.")
            if manifest["overlap_match_count"] != 0 or manifest["overlap_report_sha256"] != sha256(directory / "evaluation-overlap.json"):
                raise ValueError("Resolve the recorded evaluation-overlap check before measurement.")
            checkpoint = directory / f"drafter_{int(self.width * 100)}.pt"
            checkpoint_sha = sha256(checkpoint)
            if manifest and manifest["checkpoints"][str(int(self.width * 100))]["sha256"] != checkpoint_sha:
                raise ValueError("The drafter checkpoint differs from its training manifest.")
            torch = self.ns["torch"]
            state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
            if state["update"] != 50:
                raise ValueError("The saved pilot must contain exactly 50 completed updates.")
            self.drafter = self.ns["Drafter"](self.width)
            self.drafter.load_state_dict(state["drafter"], strict=True)
            self.drafter = self.drafter.to(torch.bfloat16).to("cuda").eval()
            del state
            self.provenance.update(
                checkpoint_path=str(checkpoint.resolve()), checkpoint_sha256=checkpoint_sha,
                training_data_sha256=sha256(data_path), vocabulary_reconstruction_rows=200, checkpoint_updates=50,
                draft_vocabulary_size=len(keep),
                draft_vocabulary_sha256=hashlib.sha256(json.dumps(keep, separators=(",", ":")).encode()).hexdigest(),
                training_target_revision=manifest["training_target_revision"] if original else manifest["target_revision"],
                training_provenance=manifest["training_provenance"] if original else "Recorded local reproduction.",
                artifact_origin="teammate-original" if original else "local-reproduction",
                training_manifest=None if original else manifest,
                import_manifest=manifest if original else None,
                checkpoint_manifest=manifest, feed_forward_width=self.width, draft_proposals_per_round=8,
            )
        self.ns["torch"].cuda.synchronize()
        self.rounds = []

    @contextmanager
    def observation(self, request_id, kind, token_count):
        if self.diagnostic_dir is None:
            yield
            return
        import trial_phase_hooks as hooks
        hooks.begin_round([request_id], kind, phase="worker_iteration", tokens={request_id: token_count})
        try:
            yield
            hooks.end_round()
        except BaseException as error:
            hooks._abort(error)
            raise

    def call(self, phase, function, *args):
        if self.diagnostic_dir is None:
            return function(*args)
        import trial_phase_hooks as hooks
        return hooks._phase_call(function, phase, "Team notebook function", *args)

    def generate(self, prompt, maximum, request_id):
        """Yield committed token groups at the notebook's natural decode boundaries."""
        ns, torch = self.ns, self.ns["torch"]
        self.rounds = []
        with torch.no_grad():
            ids = torch.tensor([prompt], device="cuda")
            with self.observation(request_id, "prefill", len(prompt)):
                features, logits, cache = self.call("target_execution", ns["target_step"], ids, None)
                first = logits[0, -1].argmax().item()
            output, reason, stop = retained_chunk([first], maximum, ns["EOS"])
            yield output[:], reason, stop
            if reason:
                return
            if self.drafter is None:
                while len(output) < maximum:
                    with self.observation(request_id, "autoregressive", 1):
                        _, logits, cache = self.call("target_execution", ns["target_step"],
                                                    torch.tensor([[output[-1]]], device="cuda"), cache)
                        token = logits[0, -1].argmax().item()
                    chunk, reason, stop = retained_chunk([token], maximum - len(output), ns["EOS"])
                    output.extend(chunk)
                    yield chunk, reason, stop
                    if reason:
                        return
                return
            toks = torch.cat([ids[0], logits[0, -1:].argmax(-1)])
            feats, nexts = features[0], toks[1:]
            dcache = ns["new_cache"]()
            while len(output) < maximum:
                with self.observation(request_id, "speculative", 9):
                    guesses = self.call("draft_proposal", ns["draft"], self.drafter, dcache, feats, nexts)
                    block = torch.cat([toks[-1:], guesses])[None]
                    features, logits, cache = self.call("target_execution", ns["target_step"], block, cache)
                    choice = logits[0].argmax(-1)
                    accepted = 0
                    while accepted < 8 and guesses[accepted] == choice[accepted]:
                        accepted += 1
                    new = torch.cat([guesses[:accepted], choice[accepted:accepted + 1]])
                    cache.crop(len(toks) + accepted)
                    feats, nexts = features[0, :accepted + 1], new
                    toks = torch.cat([toks, new])
                    chunk, reason, stop = retained_chunk(new.tolist(), maximum - len(output), ns["EOS"])
                row = {
                    "request_id": request_id, "round_index": len(self.rounds),
                    "proposed_draft_tokens": 8, "accepted_draft_tokens": accepted,
                    "retained_target_tokens": 1, "committed_tokens": accepted + 1,
                    "delivered_tokens": len(chunk) - int(stop is not None),
                    "proposed_depth": 8, "accepted_prefix_depth": accepted,
                }
                self.rounds.append(row)
                if self.diagnostic_dir:
                    self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
                    with (self.diagnostic_dir / f"team-rounds-{os.getpid()}.jsonl").open("a") as handle:
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                output.extend(chunk)
                yield chunk, reason, stop
                if reason:
                    return


def validate_generation_payload(payload, stops, vocab_size):
    from trial_recorder import check_ids
    ids = payload.get("input_ids")
    check_ids(ids)
    if not ids or any(t >= vocab_size for t in ids):
        raise ValueError("Provide nonempty input token IDs from the target vocabulary.")
    settings = payload.get("sampling_params", {})
    if (payload.get("stream") is not True or settings.get("temperature") != 0
            or settings.get("ignore_eos") is not False or settings.get("skip_special_tokens") is not True
            or settings.get("top_p") != 1 or settings.get("top_k") != -1):
        raise ValueError("Use the shared greedy streaming settings.")
    if set(settings.get("stop_token_ids", [])) != set(stops):
        raise ValueError("The requested stopping IDs differ from the target.")
    maximum = settings.get("max_new_tokens")
    if type(maximum) is not int or not 1 <= maximum <= 2048 or len(ids) + maximum > 131072:
        raise ValueError("The request exceeds the token limits.")
    rid = payload.get("rid")
    if not isinstance(rid, str) or not rid:
        raise ValueError("Provide a request ID.")
    return ids, maximum, rid


def serve(args):
    from trial_recorder import save_json
    engine = TeamEngine(args)
    settings = {
        "configuration": args.configuration, "backend": "transformers-team",
        "stream_protocol": "sglang-compatible cumulative output",
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
        "gpu_uuid": args.gpu_uuid, "phase": "diagnostic" if args.diagnostic_dir else "primary",
        "classification": "Team P-EAGLE-style FFN-width pilot",
        "adapter_sha256": sha256(__file__), **engine.provenance,
        "stopping": "The adapter stops at the first EOS or the shared output limit.",
        "residency": "The target and the selected drafter occupy the GPU.",
        "round_count_transport": "Compact accepted/delivered pairs; full records expand after measurement.",
    }

    class Handler(BaseHTTPRequestHandler):
        disable_nagle_algorithm = True

        def log_message(self, format, *values):
            pass

        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            body = json.dumps({"ready": True, "configuration": args.configuration}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/generate":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4 * 1024 * 1024:
                    raise ValueError("The request body has an invalid size.")
                prompt, maximum, rid = validate_generation_payload(
                    json.loads(self.rfile.read(size)), engine.ns["EOS"], engine.ns["cfg"].vocab_size)
            except (ValueError, TypeError, KeyError) as error:
                self.send_error(400, str(error))
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            output, previous_text = [], ""
            try:
                for chunk, reason, stop in engine.generate(prompt, maximum, rid):
                    output.extend(chunk)
                    text = engine.ns["tok"].decode(output, skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=False)
                    if reason is None:
                        text = text.rstrip("\ufffd")
                    if not text.startswith(previous_text):
                        raise ValueError("The tokenizer rewrote previously transmitted text.")
                    previous_text = text
                    meta = {"id": rid, "completion_tokens": len(output)}
                    if reason:
                        meta.update(
                            finish_reason={"type": reason, "matched": stop},
                            team_rounds_compact=[
                                [row["accepted_draft_tokens"], row["delivered_tokens"]] for row in engine.rounds],
                            team_rounds_encoding="accepted-delivered-pairs-v1",
                            team_rounds_schema_version=1,
                            team_initial_target_token_id=output[0],
                            team_initial_delivered_tokens=int(output[0] not in engine.ns["EOS"]),
                            team_delivered_output_tokens=len(output) - int(stop is not None),
                            spec_verify_ct=len(engine.rounds),
                            spec_num_correct_drafts=sum(r["accepted_draft_tokens"] for r in engine.rounds),
                            spec_num_proposed_drafts=8 * len(engine.rounds),
                        )
                    event = {"output_ids": output, "text": text, "meta_info": meta}
                    self.wfile.write(b"data: " + json.dumps(event, ensure_ascii=False, allow_nan=False).encode() + b"\n\n")
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception as error:
                self.wfile.write(b"data: " + json.dumps({"error": str(error)}).encode() + b"\n\n")
                self.wfile.flush()
                raise

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    save_json(args.server_record, settings)
    print(f"Serving {args.configuration} on http://127.0.0.1:{args.port}.", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def expand_team_rounds(metadata, request_id, speculative):
    """Expand exact integer counts after timing and validate their reconciliation."""
    if metadata.get("team_rounds_encoding") != "accepted-delivered-pairs-v1":
        raise ValueError("The team round encoding is unknown.")
    pairs = metadata.get("team_rounds_compact")
    if not isinstance(pairs, list):
        raise ValueError("The compact verification counts must be a list.")
    rows = []
    for index, pair in enumerate(pairs):
        if (not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair)
                or not 0 <= pair[0] <= 8 or not 0 <= pair[1] <= pair[0] + 1):
            raise ValueError("The compact verification counts are invalid.")
        accepted, delivered = pair
        rows.append({
            "request_id": request_id, "round_index": index, "proposed_draft_tokens": 8,
            "accepted_draft_tokens": accepted, "retained_target_tokens": 1,
            "committed_tokens": accepted + 1, "delivered_tokens": delivered,
            "proposed_depth": 8, "accepted_prefix_depth": accepted,
        })
    if (metadata["spec_verify_ct"] != len(rows)
            or metadata["spec_num_correct_drafts"] != sum(r["accepted_draft_tokens"] for r in rows)
            or metadata["spec_num_proposed_drafts"] != 8 * len(rows)):
        raise ValueError("The native team counters disagree with the compact round counts.")
    if not speculative and rows:
        raise ValueError("Ordinary decoding must contain zero speculative rounds.")
    if speculative and metadata["team_initial_delivered_tokens"] + sum(r["delivered_tokens"] for r in rows) != metadata["team_delivered_output_tokens"]:
        raise ValueError("Delivered round counts disagree with the complete output count.")
    return rows


def finalize_round_artifacts(output, configuration):
    """Keep raw stream events and add expanded count records after the timed suite."""
    paths = [Path(output) / "warmup.json", *sorted(Path(output).glob("[0-9][0-9]-*.json"))]
    for path in paths:
        data = json.loads(path.read_text())
        requests = data if isinstance(data, list) else data["requests"]
        for request in requests:
            metadata = request["server_metadata"]
            if metadata["id"] != request["request_id"]:
                raise ValueError("The stream metadata names another request.")
            if metadata["team_delivered_output_tokens"] != request["output_tokens"]:
                raise ValueError("The team's delivered count differs from the recorder.")
            if metadata["team_initial_target_token_id"] != request["native_output_token_ids"][0]:
                raise ValueError("The initial target token differs from the recorded token IDs.")
            metadata["team_rounds"] = expand_team_rounds(metadata, request["request_id"], configuration != "team-ar")
            metadata["team_rounds_materialization"] = "Expanded after measurement from the raw compact stream counts."
        temporary = path.with_name(path.name + ".expanding")
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temporary.replace(path)


def record(args):
    import trial_recorder as recorder
    from download_checkpoints import require_compute_allocation
    require_compute_allocation()
    if urlparse(args.url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Use the compute node's loopback endpoint.")
    raw = Path(args.prepared).read_bytes()
    bundle = json.loads(raw)
    if (bundle["input_sha256"] != recorder.INPUT_SHA256 or bundle["target_revision"] != recorder.TARGET_REVISION
            or len(bundle["warmups"]) != 3
            or Counter(r["benchmark"] for r in bundle["prompts"]) != Counter({b: 3 for b in recorder.BENCHMARKS})):
        raise ValueError("Use the shared prepared 30-prompt bundle.")
    for row in bundle["prompts"]:
        recorder.check_ids(row["input_token_ids"])
        if len(row["messages"]) != 1 or recorder.digest(row["messages"][0]["content"].encode()) != row["promptSha256"]:
            raise ValueError("Prepared prompt text differs from its checksum.")
    server_raw = Path(args.server_record).read_bytes()
    server = json.loads(server_raw)
    for field, expected in (("configuration", args.configuration), ("hostname", socket.gethostname()),
                            ("slurm_job_id", os.environ["SLURM_JOB_ID"]), ("gpu_uuid", args.gpu_uuid)):
        if server[field] != expected:
            raise ValueError("The server record differs in " + field)
    if server["target_revision"] != recorder.TARGET_REVISION or server["adapter_sha256"] != sha256(__file__):
        raise ValueError("Server provenance differs from the client.")
    if server.get("checkpoint_manifest") and server["checkpoint_manifest"]["evaluation_prepared_sha256"] != recorder.digest(raw):
        raise ValueError("The checkpoint overlap check used another prepared evaluation file.")
    device = subprocess.check_output(["nvidia-smi", "--id=" + args.gpu_uuid,
        "--query-gpu=uuid,name,driver_version", "--format=csv,noheader,nounits"], text=True).strip()
    if len(device.splitlines()) != 1 or "L40S" not in device:
        raise ValueError("The trial requires the assigned L40S GPU.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    recorder.save_json(output / "run.json", {
        "configuration": args.configuration, "reference": None if args.configuration == "team-ar" else "team-ar",
        "backend": "transformers-team", "stream_protocol": "sglang-compatible",
        "artifact_origin": server.get("artifact_origin", "paired-reference"),
        "server_url": args.url, "model": args.model, "server_settings": server,
        "server_record_sha256": recorder.digest(server_raw), "prepared_sha256": recorder.digest(raw),
        "input_sha256": recorder.INPUT_SHA256, "target_revision": recorder.TARGET_REVISION,
        "gpu_uuid": args.gpu_uuid, "gpu_record": device, "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "hostname": socket.gethostname(), "repetitions": 3, "temperature": 0, "max_new_tokens": 2048,
        "phase": server["phase"], "clock": "time.perf_counter_ns",
        "clock_info": vars(time.get_clock_info("perf_counter")),
        "clock_anchor": {"unix_ns": time.time_ns(), "monotonic_ns": time.perf_counter_ns()},
        "adapter_sha256": sha256(__file__), "recorder_sha256": sha256(recorder.__file__),
        "sglang_incremental": False,
    })
    with recorder.LiveWriter(output / "live.jsonl") as live:
        args.live_writer = live
        args.sglang_incremental = False
        recorder.measure(args, bundle, "sglang", output)
    finalize_round_artifacts(output, args.configuration)


def qualify(args):
    from trial_recorder import INPUT_SHA256, TARGET_REVISION, save_json
    bundle = json.loads(Path(args.prepared).read_text())
    if bundle["input_sha256"] != INPUT_SHA256 or bundle["target_revision"] != TARGET_REVISION:
        raise ValueError("Use the shared prepared prompt bundle.")
    if len(bundle["warmups"]) != 3:
        raise ValueError("The qualification requires the three shared warmup prompts.")
    engine = TeamEngine(args)
    checks = []
    prompts = bundle.get("prompts", [])
    if not prompts:
        raise ValueError("The qualification requires the prepared evaluation prompts.")
    longest = max(prompts, key=lambda row: len(row["input_token_ids"]))
    for row in [*bundle["warmups"], longest]:
        prompt = row["input_token_ids"]
        for maximum in (1, 9, 33):
            if engine.drafter is None:
                original, _ = engine.ns["standard"](prompt, maximum)
            else:
                original, _ = engine.ns["speculative"](prompt, engine.drafter, maximum)
            groups = list(engine.generate(prompt, maximum, f"qualify-{row['id']}-{maximum}"))
            observed = [token for group, _, _ in groups for token in group]
            expected_reason = "stop" if original and original[-1] in engine.ns["EOS"] else "length"
            checks.append({
                "prompt_id": row["id"], "max_new_tokens": maximum,
                "notebook_token_ids": original, "adapter_token_ids": observed,
                "token_ids_match": original == observed,
                "finish_reason_match": groups[-1][1] == expected_reason,
            })
    result = {"configuration": args.configuration, "provenance": engine.provenance, "checks": checks,
              "passed": all(r["token_ids_match"] and r["finish_reason_match"] for r in checks)}
    save_json(args.output, result)
    if not result["passed"]:
        raise ValueError("The streaming adapter differs from the pinned notebook. Inspect the saved qualification.")
    print("The adapter matches the pinned notebook on all twelve qualification cases.", flush=True)


def compare(args):
    import trial_recorder
    import trial_analysis
    method = json.loads((Path(args.method) / "run.json").read_text())
    reference = json.loads((Path(args.reference) / "run.json").read_text())
    for field in ("backend", "phase", "gpu_uuid"):
        if method[field] != reference[field]:
            raise ValueError("Paired team runs differ in " + field)
    if method["backend"] != "transformers-team" or method["phase"] not in {"primary", "diagnostic"}:
        raise ValueError("Use two team runs from the same measurement phase.")
    for field in ("torch_version", "transformers_version", "cuda_version", "target_attention_implementation"):
        if method["server_settings"][field] != reference["server_settings"][field]:
            raise ValueError("Paired team inference runtimes differ in " + field)
    trial_recorder.CONFIGURATIONS.update({
        name: ("transformers-team", None if name == "team-ar" else "team-ar")
        for name in CONFIGURATIONS
    })
    result = trial_analysis.compare(args.method, args.reference)
    trial_recorder.save_json(args.output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    source = commands.add_parser("export-source")
    source.add_argument("--output", type=Path, required=True)
    server = commands.add_parser("serve")
    server.add_argument("--notebook", type=Path, required=True)
    server.add_argument("--hf-home", type=Path, required=True)
    server.add_argument("--checkpoint-dir", type=Path, required=True)
    server.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    server.add_argument("--gpu-uuid", required=True)
    server.add_argument("--server-record", type=Path, required=True)
    server.add_argument("--port", type=int, default=31002)
    server.add_argument("--diagnostic-dir", type=Path)
    client = commands.add_parser("record")
    client.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    client.add_argument("--prepared", type=Path, required=True)
    client.add_argument("--server-record", type=Path, required=True)
    client.add_argument("--gpu-uuid", required=True)
    client.add_argument("--output", type=Path, required=True)
    client.add_argument("--url", default="http://127.0.0.1:31002")
    client.add_argument("--model", default="hpml-target")
    client.add_argument("--timeout", type=float, default=900)
    qualification = commands.add_parser("qualify")
    qualification.add_argument("--notebook", type=Path, required=True)
    qualification.add_argument("--hf-home", type=Path, required=True)
    qualification.add_argument("--checkpoint-dir", type=Path, required=True)
    qualification.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    qualification.add_argument("--gpu-uuid", required=True)
    qualification.add_argument("--prepared", type=Path, required=True)
    qualification.add_argument("--output", type=Path, required=True)
    qualification.set_defaults(diagnostic_dir=None)
    comparison = commands.add_parser("compare")
    comparison.add_argument("method", type=Path)
    comparison.add_argument("reference", type=Path)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.command == "export-source":
        export_notebook(args.output)
    elif args.command == "serve":
        serve(args)
    elif args.command == "qualify":
        qualify(args)
    elif args.command == "compare":
        compare(args)
    else:
        record(args)


if __name__ == "__main__":
    main()
