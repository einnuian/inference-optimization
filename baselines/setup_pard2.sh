#!/usr/bin/env bash
# Prepare the pinned PARD-2 inference imports in a separate Slurm environment.
# Setup: HPML_ROOT=/your/private/scratch/hpml-baselines bash baselines/setup_pard2.sh
# Resume: bash baselines/setup_pard2.sh --resume
# Brief runtime check in a GPU allocation: bash baselines/setup_pard2.sh --check-gpu
# Source: AMD-AGI/PARD at 6f279bf3f1680e0b5d71c562ca5b91bdeef4c038.
# This script installs packages, checks imports/kernels, and can load the released
# drafter. Canonical Meta target loading and exact-output checks are separate steps.
set -Eeuo pipefail
umask 077
fail() { printf 'PARD-2 setup: %s\n' "$*" >&2; exit 1; }
case "${1:-}" in
  '') mode=setup ;;
  --resume) mode=resume ;;
  --check-gpu) mode=gpu ;;
  --check-draft) mode=draft ;;
  -h|--help)
    printf '%s\n' 'Inside a Slurm allocation, set HPML_ROOT to your private scratch root.' \
      'Setup creates envs/pard2-cu124-py310 using the existing uv bootstrap and Python 3.10 runtime.' \
      'Use --resume for a partial environment, --check-gpu for small kernels, or --check-draft to load the cached released drafter and adapter.'
    exit 0 ;;
  *) fail 'Use no argument, --resume, --check-gpu, or --check-draft.' ;;
esac
[[ $# -le 1 ]] || fail 'Provide at most one argument.'
[[ $(uname -s) == Linux ]] || fail 'Run on a Linux Slurm compute node.'
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_JOB_NODELIST:-} ]] || fail 'Enter a Slurm compute allocation.'
[[ ${HPML_ROOT:-} == /* && ${HPML_ROOT:-} != / ]] || fail 'Set HPML_ROOT to an absolute private scratch directory.'
current_host=$(hostname -s)
found=false
while IFS= read -r host; do
  [[ ${host%%.*} != "$current_host" ]] || found=true
done < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
$found || fail 'This host must belong to the assigned compute-node list.'

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$HPML_ROOT/cache/huggingface"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export UV_CACHE_DIR="$HPML_ROOT/cache/uv" UV_LINK_MODE=copy
export XDG_CACHE_HOME="$HPML_ROOT/cache" TMPDIR="$HPML_ROOT/tmp"
export TORCHINDUCTOR_CACHE_DIR="$HPML_ROOT/cache/torchinductor-pard2"
export TRITON_CACHE_DIR="$HPML_ROOT/cache/triton-pard2"
export HPML_PARD_SOURCE="$HPML_ROOT/source/hpml-project/external/pard"
export HPML_PARD_VENV="$HPML_ROOT/envs/pard2-cu124-py310"
export HPML_PARD_REVISION=6f279bf3f1680e0b5d71c562ca5b91bdeef4c038
uv_bin="$HPML_ROOT/envs/uv-bootstrap/bin/uv"
python_bin="$HPML_PARD_VENV/bin/python"
[[ -x "$uv_bin" ]] || fail 'Prepare the existing uv-bootstrap environment first.'
[[ -x "$HPML_ROOT/envs/eagle-cu124-py310/bin/python" ]] || fail 'Prepare the shared Python 3.10 runtime first.'
[[ $(git -C "$HPML_PARD_SOURCE" rev-parse HEAD) == "$HPML_PARD_REVISION" ]] || fail 'PARD source revision differs from the reviewed pin.'
[[ -z $(git -C "$HPML_PARD_SOURCE" status --porcelain=v1 --untracked-files=all) ]] || fail 'Preserve and review local PARD source changes.'

python3 - <<'PY'
import os, stat
from pathlib import Path
root = Path(os.environ['HPML_ROOT'])
for path in (root, *root.parents):
    if path.is_symlink():
        raise SystemExit('Use ordinary directory components for HPML_ROOT.')
if not root.is_dir() or root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode) & 0o077:
    raise SystemExit('HPML_ROOT must be an existing private directory owned by your account.')
venv = Path(os.environ['HPML_PARD_VENV'])
if venv.is_symlink() or (venv.exists() and (not venv.is_dir() or venv.stat().st_uid != os.getuid())):
    raise SystemExit('The PARD-2 environment path must be an owned ordinary directory.')
PY

if [[ "$mode" == setup || "$mode" == resume ]]; then
  setup_lock="$HPML_ROOT/.pard2-setup.lock"
  mkdir "$setup_lock" || fail 'Another PARD-2 setup or an unfinished setup lock exists.'
  trap 'rmdir "$setup_lock"' EXIT
  if [[ "$mode" == setup ]]; then
    [[ ! -e "$HPML_PARD_VENV" ]] || fail 'Preserving the existing environment. Inspect it and use --resume.'
    "$uv_bin" --no-config venv --python "$HPML_ROOT/envs/eagle-cu124-py310/bin/python" "$HPML_PARD_VENV"
  else
    [[ -x "$python_bin" ]] || fail 'Resume requires an existing PARD-2 environment.'
  fi
  "$python_bin" - <<'PY'
import importlib.metadata as md
import os, sys
from pathlib import Path
if Path(sys.prefix).resolve() != Path(os.environ['HPML_PARD_VENV']).resolve() or tuple(sys.version_info[:3]) != (3, 10, 22):
    raise SystemExit('Expected the separate Python 3.10.22 PARD-2 environment.')
for name, expected in {'torch':'2.6.0+cu124','transformers':'4.51.3','numpy':'1.26.4',
                       'click':'8.1.8','loguru':'0.7.3','PyYAML':'6.0.2','setuptools':'80.9.0'}.items():
    try: actual = md.version(name)
    except md.PackageNotFoundError: continue
    if actual != expected: raise SystemExit(f'Preserving {name} {actual}; expected {expected}.')
PY
  unset UV_INDEX UV_INDEX_URL UV_EXTRA_INDEX_URL UV_DEFAULT_INDEX UV_FIND_LINKS
  "$uv_bin" --no-config pip install --python "$python_bin" --index-url https://pypi.org/simple \
    --find-links https://download.pytorch.org/whl/cu124/torch/ 'torch==2.6.0+cu124'
  "$uv_bin" --no-config pip install --python "$python_bin" --index-url https://pypi.org/simple \
    'torch==2.6.0+cu124' 'transformers==4.51.3' 'numpy==1.26.4' \
    'click==8.1.8' 'loguru==0.7.3' 'PyYAML==6.0.2' 'setuptools==80.9.0'
  "$uv_bin" --no-config pip check --python "$python_bin"
  "$uv_bin" --no-config pip freeze --python "$python_bin" > "$HPML_PARD_VENV/hpml-installed.txt"
fi

[[ -x "$python_bin" ]] || fail 'Complete PARD-2 setup first.'
export HPML_PARD_CHECK_MODE="$mode"
cd "$HPML_PARD_SOURCE"
"$python_bin" - <<'PY'
import importlib.metadata as md
import json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import torch
import triton
import transformers
from transformers import LlamaConfig, StaticCache
from pard.pard2_infer import PardInfer, TargetFeatEmbedWarp
import pard.infer
expected = {'torch':'2.6.0+cu124','transformers':'4.51.3','numpy':'1.26.4',
            'click':'8.1.8','loguru':'0.7.3','PyYAML':'6.0.2','setuptools':'80.9.0'}
for name, version in expected.items():
    if md.version(name) != version: raise SystemExit(f'{name} differs from {version}.')
if torch.version.cuda != '12.4': raise SystemExit('Expected the CUDA 12.4 PyTorch wheel.')
record = {'checked_at_utc':datetime.now(timezone.utc).isoformat(), 'python':sys.version,
          'packages':expected, 'cuda_runtime':torch.version.cuda,
          'source_revision':os.environ['HPML_PARD_REVISION'],
          'source_path':os.environ['HPML_PARD_SOURCE'], 'slurm_job_id':os.environ['SLURM_JOB_ID'],
          'imports':['PardInfer','TargetFeatEmbedWarp','pard.infer','StaticCache','triton'],
          'scope':'Inference imports only; model loading and output qualification remain pending.',
          'runtime_choice':'Torch 2.6.0+cu124 is a driver-compatible preparation choice; the authors document NGC 25.02.'}
filename = 'hpml-setup.json'
if os.environ['HPML_PARD_CHECK_MODE'] == 'gpu':
    if not torch.cuda.is_available(): raise SystemExit('CUDA is unavailable in this allocation.')
    with torch.inference_mode():
        operand = torch.ones((32,32),device='cuda',dtype=torch.bfloat16)
        eager = operand @ operand
        compiled = torch.compile(lambda x: x @ x, fullgraph=True, mode='reduce-overhead')(operand)
        torch.cuda.synchronize()
        expected_result = torch.full_like(eager,32)
        if not torch.equal(eager,expected_result) or not torch.equal(compiled,expected_result):
            raise SystemExit('The eager or compiled BF16 kernel check failed.')
        config = LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=1,
                             num_attention_heads=4, num_key_value_heads=2)
        cache = StaticCache(config=config,max_batch_size=1,max_cache_len=16,
                            device='cuda',dtype=torch.bfloat16)
        cache.reset()
    record.update({'scope':'Brief eager/compiled BF16 and StaticCache checks; no model execution or benchmark.',
                   'gpu':torch.cuda.get_device_name(0),'capability':list(torch.cuda.get_device_capability(0)),
                   'bf16_matmul_32x32':'passed','compiled_matmul_32x32':'passed','static_cache':'passed'})
    filename = 'hpml-gpu-check.json'
if os.environ['HPML_PARD_CHECK_MODE'] == 'draft':
    if not torch.cuda.is_available(): raise SystemExit('CUDA is unavailable in this allocation.')
    revision = 'a509e315bfe79a0be8b832004bb3ee14acfdf86a'
    snapshot = Path(os.environ['HF_HOME']) / 'hub/models--amd--PARD2-Llama-3.1-8B/snapshots' / revision
    for name in ('config.json','model.safetensors','warp_model.bin'):
        if not (snapshot/name).is_file(): raise SystemExit(f'Acquire the pinned PARD-2 snapshot first; missing {name}.')
    runner = PardInfer(draft_k=15,tokens=1,draft=str(snapshot),
                       target='meta-llama/Llama-3.1-8B-Instruct',benchmark='humaneval',
                       para=True,nc=True,max_cache_len=32,model_serie='llama3')
    model, cache = runner.model_init_draft2(str(snapshot),max_cache_len=32,
                                           scale=runner.scale,proj_bias=runner.proj_bias)
    from safetensors import safe_open
    with safe_open(snapshot/'model.safetensors',framework='pt',device='cpu') as handle:
        saved_keys = set(handle.keys())
    model_keys = set(model.base_model.state_dict())
    if saved_keys != model_keys: raise SystemExit('Backbone tensor keys differ from the released safetensors file.')
    warp = torch.load(snapshot/'warp_model.bin',map_location='cpu',weights_only=True)
    if not torch.equal(model.target_proj.weight.detach().cpu(),warp['target_proj.weight'].to(torch.bfloat16)):
        raise SystemExit('The loaded adapter differs from the saved projection after BF16 conversion.')
    torch.cuda.synchronize()
    record.update({'scope':'Released drafter and adapter load only; no target loading or generated tokens.',
                   'draft_revision':revision,'snapshot_path':str(snapshot),
                   'gpu':torch.cuda.get_device_name(0),'backbone_tensor_keys':len(saved_keys),
                   'adapter_shape':list(model.target_proj.weight.shape),'adapter_equal_after_bf16_cast':True,
                   'target_feature_layers':runner.target_layers,'draft_cache_capacity':32})
    filename = 'hpml-draft-load.json'
path = Path(os.environ['HPML_PARD_VENV']) / filename
path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
PY
printf 'PARD-2 environment: %s\n' "$HPML_PARD_VENV"
