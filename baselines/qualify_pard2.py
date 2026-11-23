#!/usr/bin/env python3
"""Qualify canonical Meta target and released PARD-2 on four setup prompts.

Use the separate PARD-2 environment inside one GPU allocation. Target-only and
speculative stages run in separate processes. The unchanged released generator
collects internal timings; this wrapper discards those summaries and records
only token/correctness evidence. It runs no dataset performance benchmark.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True

from qualify_eagle import (PROMPTS, TARGET_REVISION, apply_output_policy,
                           compare_output_records, check_output_policy, write_record)

DRAFT_REVISION = 'a509e315bfe79a0be8b832004bb3ee14acfdf86a'
SOURCE_REVISION = '6f279bf3f1680e0b5d71c562ca5b91bdeef4c038'
CACHE_CAPACITY = 512


def execute_stage(stage, output, target, draft, source):
    if tuple(sys.version_info[:3]) != (3, 10, 22):
        raise RuntimeError('PARD-2 qualification requires the prepared Python 3.10.22 interpreter.')
    sys.path.insert(0, str(source))
    import torch
    import transformers
    if torch.__version__ != '2.6.0+cu124' or transformers.__version__ != '4.51.3':
        raise RuntimeError('Use the prepared PARD-2 environment and reviewed package pins.')
    from transformers import AutoTokenizer
    from pard.pard2_infer import PardInfer, logger

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Allocate exactly one visible GPU.')
    torch.manual_seed(0)
    # The released method prints timing summaries; the qualification keeps token evidence.
    logger.remove()
    runner = PardInfer(draft_k=15, tokens=32, draft=str(draft), target=str(target),
                       benchmark='humaneval', para=True, nc=True,
                       max_cache_len=CACHE_CAPACITY, model_serie='llama3')
    tokenizer = AutoTokenizer.from_pretrained(target, use_fast=False, local_files_only=True)
    generation = json.loads((target / 'generation_config.json').read_text())
    values = generation['eos_token_id']
    stop_ids = set(values if isinstance(values, list) else [values])

    def get_input(prompt, unused_tokenizer, prompt_type):
        messages = [{'role': 'user', 'content': prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids.to('cuda:0')
        return text, ids

    # Share the canonical user-only chat input with the other setup checks.
    runner.get_input = get_input
    print(f'Loading {stage} in a separate process.', flush=True)
    if stage == 'target':
        model, cache = runner.model_init(str(target), max_cache_len=CACHE_CAPACITY)
        model.eval()
    else:
        runner.get_model(str(draft), str(target), 'llama3', torch.bfloat16,
                         CACHE_CAPACITY, nc=True, maxtune=False)
        runner.model.eval()
        runner.model_draft.eval()

    class CaptureTokenizer:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            return getattr(tokenizer, name)

        def decode(self, token_ids, *args, **kwargs):
            self.calls.append(token_ids.detach().cpu().tolist())
            return ''

    records = []
    with torch.inference_mode():
        for name, prompt, cap in PROMPTS:
            _, input_ids = get_input(prompt, tokenizer, 'llama3')
            if input_ids.shape[-1] + cap + 64 >= CACHE_CAPACITY:
                raise RuntimeError('The setup input exceeds the fixed cache capacity.')
            if stage == 'target':
                cache.reset()
                state = model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                              cache_position=torch.arange(input_ids.shape[-1], device='cuda:0'),
                              return_dict=True, output_attentions=False)
                raw = []
                for position in range(cap):
                    token = int(state.logits[0, -1].argmax().item())
                    raw.append(token)
                    if token in stop_ids or position == cap - 1:
                        break
                    state = model(input_ids=torch.tensor([[token]], device='cuda:0'),
                                  past_key_values=cache, use_cache=True,
                                  cache_position=torch.tensor([input_ids.shape[-1] + position], device='cuda:0'),
                                  return_dict=True, output_attentions=False)
            else:
                runner.tokens = cap
                captured = CaptureTokenizer()
                with contextlib.redirect_stdout(io.StringIO()):
                    runner.generate([prompt], runner.model, runner.model_draft,
                                    runner.target_cache, runner.draft_cache, captured, runner.v2)
                if len(captured.calls) != 1:
                    raise RuntimeError('Expected one final token sequence from the released generator.')
                full = captured.calls[0]
                if full[:input_ids.shape[-1]] != input_ids[0].tolist():
                    raise RuntimeError('The released generator changed the prompt prefix.')
                raw = full[input_ids.shape[-1]:]
            record = {'name': name, 'prompt': prompt, 'max_new_tokens': cap,
                      'input_token_ids': input_ids[0].tolist(),
                      **apply_output_policy(raw, cap, stop_ids)}
            record['returned_text'] = tokenizer.decode(record['returned_token_ids'], skip_special_tokens=True)
            records.append(record)
            print(f"{stage}: {name}, {len(raw)} raw tokens, {record['stop_reason']}.", flush=True)
    torch.cuda.synchronize()
    write_record(output / f'{stage}.json', {
        'stage': stage, 'scope': 'Four pretrained setup prompts; output and stopping evidence only.',
        'target_revision': TARGET_REVISION, 'draft_revision': DRAFT_REVISION if stage == 'pard2' else None,
        'source_revision': SOURCE_REVISION, 'target_path': str(target),
        'draft_path': str(draft) if stage == 'pard2' else None,
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'python': sys.version, 'transformers': transformers.__version__,
        'cuda_runtime': torch.version.cuda, 'dtype': 'bfloat16', 'attention_backend': 'eager',
        'slurm_job_id': os.environ['SLURM_JOB_ID'], 'cache_capacity': CACHE_CAPACITY,
        'draft_length': 15 if stage == 'pard2' else None,
        'canonical_stop_token_ids': sorted(stop_ids),
        'released_producer_stop_token_id': tokenizer.eos_token_id if stage == 'pard2' else None,
        'prompt_format': 'Canonical Meta tokenizer user-only chat template with generation prompt; same input IDs for both stages.',
        'input_adaptation': 'The released PARD helper inserts a system message. This wrapper supplies canonical user-only input to match the setup prompt protocol.',
        'timing_scope': 'Released internal timing summaries are discarded; no performance result is recorded.',
        'output_policy': 'Retain the first canonical stop token or strict output cap; exclude a terminal stop token from returned IDs. Preserve every raw producer token separately.',
        'prompts': records,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-policy', action='store_true', help='Run pure boundary checks without model loading.')
    parser.add_argument('--stage', choices=['target', 'pard2'])
    args = parser.parse_args()
    if args.check_policy:
        check_output_policy()
        return 0
    if args.output_dir is None:
        parser.error('--output-dir is required for model qualification.')
    from download_checkpoints import require_compute_allocation, require_scratch_cache
    require_compute_allocation()
    home = require_scratch_cache()
    root = Path(os.environ['HPML_ROOT']).resolve()
    output = args.output_dir.resolve()
    if root / 'runs' not in output.parents:
        raise RuntimeError('Use a fresh directory under HPML_ROOT/runs.')
    source = root / 'source/hpml-project/external/pard'
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain=v1'], text=True).strip()
    if actual != SOURCE_REVISION or dirty:
        raise RuntimeError('Use the clean reviewed PARD source revision.')
    target = home / 'hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots' / TARGET_REVISION
    draft = home / 'hub/models--amd--PARD2-Llama-3.1-8B/snapshots' / DRAFT_REVISION
    if not (home / 'hpml-provenance/target' / f'{TARGET_REVISION}.json').is_file():
        raise RuntimeError('Complete and verify the direct Meta target acquisition first.')
    for snapshot, names in [(target, ['config.json', 'generation_config.json', 'tokenizer.json', 'model.safetensors.index.json']),
                            (draft, ['config.json', 'model.safetensors', 'warp_model.bin'])]:
        for name in names:
            if not (snapshot / name).is_file():
                raise RuntimeError(f'The verified snapshot requires {name}.')
    os.umask(0o077)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    if args.stage:
        execute_stage(args.stage, output, target, draft, source)
        return 0
    output.mkdir(parents=True, exist_ok=False)
    for stage in ('target', 'pard2'):
        subprocess.run([sys.executable, __file__, '--output-dir', str(output), '--stage', stage], check=True)
    ordinary = json.loads((output / 'target.json').read_text())
    speculative = json.loads((output / 'pard2.json').read_text())
    result = compare_output_records(ordinary, speculative)
    result.update({
        'scope': 'Four setup prompts qualify these observed outputs. The common baseline comparison requires broader greedy correctness and performance measurements.',
        'stopping_note': 'The released PARD-2 producer checks a scalar tokenizer EOS and its length condition after whole blocks. Native readiness requires raw boundary compliance. Adapted readiness identifies the explicit output-clipping adapter. Future timing must include all executed work.',
    })
    write_record(output / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
