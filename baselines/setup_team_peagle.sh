#!/usr/bin/env bash
# Prepare the isolated runtime for the team's pinned pilot.
set -Eeuo pipefail
umask 077
fail() { printf 'Team P-EAGLE setup: %s\n' "$*" >&2; exit 1; }
mode=gpu
if [[ $# == 1 && $1 == --prepare-only ]]; then mode=cpu
elif [[ $# != 0 ]]; then fail 'Use no argument for GPU setup or --prepare-only inside a CPU allocation.'
fi
export HPML_TEAM_SETUP_MODE="$mode"
[[ $(uname -s) == Linux && ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || fail 'Use a Linux Slurm compute allocation.'
[[ ${HPML_ROOT:-} == /* && ${HPML_ROOT:-} != / ]] || fail 'Set HPML_ROOT to the private project root.'
unset PYTHONPATH PYTHONHOME UV_INDEX UV_INDEX_URL UV_EXTRA_INDEX_URL UV_DEFAULT_INDEX UV_FIND_LINKS
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="$HPML_ROOT/cache/uv" UV_LINK_MODE=hardlink TMPDIR="$HPML_ROOT/tmp"
export HPML_TEAM_VENV="$HPML_ROOT/envs/team-peagle-cu124-py310"
export HPML_TEAM_SETUP_SOURCE="${BASH_SOURCE[0]}"
python3 - <<'PY'
import os, socket, stat, subprocess
from pathlib import Path
root = Path(os.environ['HPML_ROOT'])
environment = Path(os.environ['HPML_TEAM_VENV'])
if any(p.is_symlink() for p in (root, *root.parents, environment)):
    raise SystemExit('Use ordinary owned directories.')
if not root.is_dir() or root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode) & 0o077:
    raise SystemExit('Use the existing private project root.')
if environment.exists() and (not environment.is_dir() or environment.stat().st_uid != os.getuid()):
    raise SystemExit('Inspect the existing environment path and ownership.')
job = subprocess.check_output(['scontrol', '--oneliner', 'show', 'job', os.environ['SLURM_JOB_ID']], text=True)
fields = dict(item.split('=', 1) for item in job.split() if '=' in item)
if fields.get('JobState') != 'RUNNING' or not fields.get('UserId', '').endswith(f'({os.getuid()})'):
    raise SystemExit('Use your running Slurm allocation.')
nodes = subprocess.check_output(['scontrol', 'show', 'hostnames', fields['NodeList']], text=True).splitlines()
if socket.gethostname().split('.')[0] not in {host.split('.')[0] for host in nodes}:
    raise SystemExit('Run on the assigned compute node.')
PY
uv_bin="$HPML_ROOT/envs/uv-bootstrap/bin/uv"
base_python="$HPML_ROOT/envs/eagle-cu124-py310/bin/python"
python_bin="$HPML_TEAM_VENV/bin/python"
[[ -x "$uv_bin" && -x "$base_python" ]] || fail 'Prepare the existing uv bootstrap and Python 3.10 runtime first.'
mkdir -p "$UV_CACHE_DIR" "$TMPDIR"
setup_lock="$HPML_ROOT/.team-peagle-setup.lock"
mkdir "$setup_lock" || fail 'An active or unfinished team setup lock exists.'
trap 'rmdir "$setup_lock"' EXIT
if [[ ! -e "$HPML_TEAM_VENV" ]]; then
  "$uv_bin" --no-config venv --python "$base_python" "$HPML_TEAM_VENV"
fi
"$python_bin" - <<'PY'
import importlib.metadata as md, os, sys
from pathlib import Path
if sys.version_info[:2] != (3, 10) or Path(sys.prefix).resolve() != Path(os.environ['HPML_TEAM_VENV']).resolve():
    raise SystemExit('Use the isolated team Python 3.10 environment.')
expected = {'torch': '2.6.0+cu124', 'transformers': '4.57.6', 'datasets': '3.6.0',
            'huggingface_hub': '0.36.2', 'wandb': '0.30.0'}
for name, version in expected.items():
    try:
        installed = md.version(name)
    except md.PackageNotFoundError:
        continue
    if installed != version:
        raise SystemExit(f'Preserving the existing {name} {installed}; inspect this environment before changing it.')
PY
"$uv_bin" --no-config pip install --python "$python_bin" \
  --index-url https://pypi.org/simple \
  --find-links https://download.pytorch.org/whl/cu124/torch/ \
  'torch==2.6.0+cu124' 'transformers==4.57.6' 'datasets==3.6.0' \
  'huggingface_hub==0.36.2' 'wandb==0.30.0'
"$uv_bin" --no-config pip check --python "$python_bin"
"$uv_bin" --no-config pip freeze --python "$python_bin" > "$HPML_TEAM_VENV/hpml-installed.txt"
"$python_bin" - <<'PY'
import hashlib, importlib.metadata as md, json, os, socket, subprocess, sys, time
from pathlib import Path
import torch, datasets, huggingface_hub, wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
assert torch.__version__ == '2.6.0+cu124'
assert torch.version.cuda == '12.4'
gpu = os.environ['HPML_TEAM_SETUP_MODE'] == 'gpu'
if gpu:
    if not torch.cuda.is_available():
        raise SystemExit('The assigned CUDA runtime must be available.')
    check = torch.tensor([2.0], device='cuda') * 3
    torch.cuda.synchronize()
    assert check.item() == 6.0
environment = Path(os.environ['HPML_TEAM_VENV'])
freeze = environment / 'hpml-installed.txt'
source = Path(os.environ['HPML_TEAM_SETUP_SOURCE']).resolve()
record = {
    'python': sys.version, 'python_executable': sys.executable,
    'packages': {name: md.version(name) for name in ('torch', 'transformers', 'datasets', 'huggingface_hub', 'wandb')},
    'cuda_version': torch.version.cuda, 'gpu_name': torch.cuda.get_device_name(0) if gpu else None,
    'gpu_inventory': subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader'], text=True).strip() if gpu else None,
    'hostname': socket.gethostname(), 'slurm_job_id': os.environ['SLURM_JOB_ID'],
    'created_unix_ns': time.time_ns(),
    'freeze_sha256': hashlib.sha256(freeze.read_bytes()).hexdigest(),
    'setup_script': str(source), 'setup_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'package_index': 'https://pypi.org/simple',
    'package_link_mode': os.environ['UV_LINK_MODE'],
    'setup_mode': os.environ['HPML_TEAM_SETUP_MODE'], 'gpu_check_passed': gpu,
    'torch_find_links': 'https://download.pytorch.org/whl/cu124/torch/',
}
for name in ('hpml-setup.json', f"hpml-setup-{record['slurm_job_id']}-{record['setup_mode']}.json"):
    path = environment / name
    temporary = path.with_suffix('.partial')
    with temporary.open('w') as handle:
        json.dump(record, handle, indent=2)
        handle.write('\n')
    temporary.replace(path)
print('The pinned packages import successfully.' + (' The CUDA check passes.' if gpu else ' GPU validation runs in the evaluation allocation.'))
PY
printf 'Team runtime: %s\n' "$HPML_TEAM_VENV"
