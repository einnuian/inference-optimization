#!/usr/bin/env bash
# Install the independent offline tracking environment inside a CPU allocation.
set -Eeuo pipefail
umask 077
fail() { printf 'Tracking setup: %s\n' "$*" >&2; exit 1; }
[[ $# == 0 ]] || fail 'Run with no arguments.'
[[ $(uname -s) == Linux && -n ${SLURM_JOB_ID:-} && -n ${SLURM_JOB_NODELIST:-} ]] || fail 'Use a Linux Slurm compute allocation.'
[[ ${HPML_ROOT:-} == /* && ${HPML_ROOT:-} != / ]] || fail 'Set HPML_ROOT to the private project root.'
current_host=$(hostname -s)
found=false
while IFS= read -r host; do
  [[ ${host%%.*} != "$current_host" ]] || found=true
done < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
$found || fail 'Run on the assigned compute node.'
unset PYTHONPATH PYTHONHOME UV_INDEX UV_INDEX_URL UV_EXTRA_INDEX_URL UV_DEFAULT_INDEX UV_FIND_LINKS
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="$HPML_ROOT/cache/uv" UV_LINK_MODE=copy TMPDIR="$HPML_ROOT/tmp"
export HPML_TRACK_VENV="$HPML_ROOT/envs/tracking-py310"
python3 - <<'PY'
import os,stat
from pathlib import Path
root=Path(os.environ['HPML_ROOT']); env=Path(os.environ['HPML_TRACK_VENV'])
if any(p.is_symlink() for p in (root,*root.parents,env)):
    raise SystemExit('Use ordinary owned directories.')
if not root.is_dir() or root.stat().st_uid!=os.getuid() or stat.S_IMODE(root.stat().st_mode)&0o077:
    raise SystemExit('Use the existing private project root.')
if env.exists() and (not env.is_dir() or env.stat().st_uid!=os.getuid()):
    raise SystemExit('Preserve the existing environment path and inspect its ownership.')
PY
uv_bin="$HPML_ROOT/envs/uv-bootstrap/bin/uv"
python_bin="$HPML_TRACK_VENV/bin/python"
[[ -x "$uv_bin" && -x "$HPML_ROOT/envs/eagle-cu124-py310/bin/python" ]] || fail 'Prepare the existing uv bootstrap and Python 3.10 runtime first.'
setup_lock="$HPML_ROOT/.tracking-setup.lock"
mkdir "$setup_lock" || fail 'An active or unfinished tracking setup lock exists.'
trap 'rmdir "$setup_lock"' EXIT
if [[ ! -e "$HPML_TRACK_VENV" ]]; then
  "$uv_bin" --no-config venv --python "$HPML_ROOT/envs/eagle-cu124-py310/bin/python" "$HPML_TRACK_VENV"
fi
"$python_bin" - <<'PY'
import importlib.metadata as md,sys
if sys.version_info[:2]!=(3,10):raise SystemExit('Expected Python 3.10.')
try: version=md.version('wandb')
except md.PackageNotFoundError: version=None
if version and version!='0.30.0':raise SystemExit('Preserving a different installed W&B version.')
PY
"$uv_bin" --no-config pip install --python "$python_bin" --index-url https://pypi.org/simple 'wandb==0.30.0'
"$uv_bin" --no-config pip check --python "$python_bin"
"$uv_bin" --no-config pip freeze --python "$python_bin" > "$HPML_TRACK_VENV/hpml-installed.txt"
"$python_bin" - <<'PY'
import wandb
settings=wandb.Settings(disable_code=True,disable_git=True,save_code=False,console='off',capture_loggers={},x_disable_meta=True,x_disable_stats=True,x_disable_machine_info=True)
assert wandb.__version__=='0.30.0'
print('W&B 0.30.0 imports and restricted capture settings pass. Authentication and sync remain separate.')
PY
printf 'Tracking environment: %s\n' "$HPML_TRACK_VENV"
