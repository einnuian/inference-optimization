#!/usr/bin/env bash
# Prepare the reviewed EAGLE model-inference environment inside a Slurm allocation.
# Usage: HPML_ROOT=/your/private/scratch/hpml-baselines bash baselines/setup_eagle.sh
# Resume a verified partial environment: bash baselines/setup_eagle.sh --resume
# Later, inside a GPU allocation: bash baselines/setup_eagle.sh --check-gpu
# The script retrieves public source and packages. Model access and runs are separate.
# Reviewed 2026-11-16:
# https://github.com/SafeAILab/EAGLE/tree/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b
# https://pytorch.org/get-started/previous-versions/#v260
# https://pypi.org/project/uv/0.13.8/
# https://docs.astral.sh/uv/getting-started/installation/#pypi
set -Eeuo pipefail
umask 077

fail() { printf 'EAGLE setup: %s\n' "$*" >&2; exit 1; }
case "${1:-}" in
  '') mode=setup ;;
  --resume) mode=resume ;;
  --check-gpu) mode=gpu ;;
  -h|--help)
    printf '%s\n' \
      'Inside a Slurm compute allocation, set HPML_ROOT to your private scratch directory.' \
      'Run: bash baselines/setup_eagle.sh' \
      'Resume a verified partial environment: bash baselines/setup_eagle.sh --resume' \
      'Inside a later GPU allocation, run: bash baselines/setup_eagle.sh --check-gpu' \
      'Setup creates envs/uv-bootstrap and envs/eagle-cu124-py310 exclusively.'
    exit 0 ;;
  *) fail 'Use no argument for setup, --resume, or --check-gpu.' ;;
esac
[[ $# -le 1 ]] || fail 'Provide at most one argument.'
[[ $(uname -s) == Linux ]] || fail 'Run this script on a Linux Slurm compute node.'
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_JOB_NODELIST:-} ]] || \
  fail 'Start a Slurm compute allocation before running this script.'
[[ ${HPML_ROOT:-} == /* && ${HPML_ROOT:-} != / ]] || \
  fail 'Set HPML_ROOT to an absolute private scratch directory.'
for command_name in python3 git scontrol; do
  command -v "$command_name" >/dev/null || fail "Load a module providing $command_name."
done
# A CPU allocation supports installation. Check that this host belongs to the job.
current_host=$(hostname -s)
job_hosts=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
found_host=false
while IFS= read -r host; do
  if [[ ${host%%.*} == "$current_host" ]]; then found_host=true; fi
done <<< "$job_hosts"
$found_host || fail 'Run inside a compute-node shell belonging to this Slurm job.'

# Keep Python imports, caches and downloaded runtimes inside the project scratch tree.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$HPML_ROOT/cache/huggingface"
export UV_CACHE_DIR="$HPML_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$HPML_ROOT/python"
export UV_PYTHON_INSTALL_BIN=0
export UV_LINK_MODE=copy
export XDG_CACHE_HOME="$HPML_ROOT/cache"
export PIP_CACHE_DIR="$HPML_ROOT/cache/pip"
export TMPDIR="$HPML_ROOT/tmp"
export HPML_EAGLE_SOURCE="$HPML_ROOT/source/EAGLE"
export HPML_EAGLE_VENV="$HPML_ROOT/envs/eagle-cu124-py310"
export HPML_UV_BOOTSTRAP="$HPML_ROOT/envs/uv-bootstrap"
export HPML_EAGLE_REVISION=cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b
export HPML_EAGLE_PYTHON=3.10.22
export HPML_SETUP_MODE="$mode"

python3 - <<'PY'
import os
from pathlib import Path
import stat
import sys
if sys.version_info < (3, 8):
    raise SystemExit('The bootstrap Python must be version 3.8 or newer.')
root = Path(os.environ['HPML_ROOT'])
for path in (root, *root.parents):
    if path.is_symlink():
        raise SystemExit(f'Choose a scratch path with ordinary directory components: {path}')
if os.environ['HPML_SETUP_MODE'] == 'setup':
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
if not root.is_dir() or root.stat().st_uid != os.getuid():
    raise SystemExit('HPML_ROOT must be an existing directory owned by your account.')
if stat.S_IMODE(root.stat().st_mode) & 0o077:
    raise SystemExit('Set HPML_ROOT permissions to 0700 to keep this scratch tree private.')
for relative in ('envs', 'source', 'cache', 'cache/huggingface', 'cache/uv',
                 'cache/pip', 'python', 'tmp'):
    path = root / relative
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise SystemExit(f'Expected an ordinary directory: {path}')
    if os.environ['HPML_SETUP_MODE'] == 'setup':
        path.mkdir(mode=0o700, exist_ok=True)
for key in ('HPML_EAGLE_VENV', 'HPML_UV_BOOTSTRAP', 'HPML_EAGLE_SOURCE'):
    path = Path(os.environ[key])
    if path.is_symlink():
        raise SystemExit(f'Expected an ordinary path: {path}')
    if path.exists() and (not path.is_dir() or path.stat().st_uid != os.getuid()):
        raise SystemExit(f'Expected a directory owned by your account: {path}')
    if os.environ['HPML_SETUP_MODE'] == 'resume' and not path.is_dir():
        raise SystemExit(f'Resume requires the existing setup directory: {path}')
PY

if [[ "$mode" == gpu ]]; then
  [[ -x "$HPML_EAGLE_VENV/bin/python" ]] || fail 'Complete environment setup first.'
  [[ -n ${CUDA_VISIBLE_DEVICES:-} && ${CUDA_VISIBLE_DEVICES:-} != NoDevFiles ]] || \
    fail 'Request a GPU allocation with a visible CUDA device.'
  command -v nvidia-smi >/dev/null || fail 'The GPU driver utility is unavailable.'
  nvidia-smi
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$HPML_EAGLE_VENV/bin/python" - <<'PY'
import json
import torch
from eagle.model.ea_model import EaModel
if torch.__version__ != '2.6.0+cu124' or torch.version.cuda != '12.4':
    raise SystemExit('The environment differs from the reviewed CUDA wheel.')
if not torch.cuda.is_available():
    raise SystemExit('PyTorch cannot access CUDA. Check the allocation and driver.')
# Verify the allocated GPU can execute the BF16 kernel used by the environment.
with torch.inference_mode():
    operand = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
    product = operand @ operand
    torch.cuda.synchronize()
    if not torch.equal(product, torch.full_like(product, 32)):
        raise SystemExit('The BF16 matrix multiplication returned an unexpected result.')
print(json.dumps({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                  'bf16_matmul_32x32': 'passed',
                  'visible_gpus': [torch.cuda.get_device_name(i)
                                   for i in range(torch.cuda.device_count())]}, indent=2))
PY
  exit 0
fi

# Reserve setup ownership. Partial environments remain available for inspection.
setup_lock="$HPML_ROOT/.eagle-setup.lock"
mkdir "$setup_lock" || fail 'Another setup or an unfinished setup lock exists.'
trap 'rmdir "$setup_lock"' EXIT
trap 'printf "Setup stopped at line %s. Preserve and inspect partial paths before retrying.\n" "$LINENO" >&2' ERR
if [[ "$mode" == setup ]]; then
  for environment_path in "$HPML_UV_BOOTSTRAP" "$HPML_EAGLE_VENV"; do
    [[ ! -e "$environment_path" && ! -L "$environment_path" ]] || \
      fail "Preserving the existing environment: $environment_path. Use --resume after inspection."
  done
fi

# Retrieve only the reviewed public EAGLE revision. Preserve existing checkouts.
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0
export GIT_LFS_SKIP_SMUDGE=1
public_git() {
  git -c core.hooksPath=/dev/null -c core.fsmonitor=false \
    -c submodule.recurse=false -c protocol.file.allow=never -c credential.helper= "$@"
}
if [[ ! -e "$HPML_EAGLE_SOURCE" ]]; then
  mkdir "$HPML_EAGLE_SOURCE"
  public_git init --quiet --template= "$HPML_EAGLE_SOURCE"
  public_git -C "$HPML_EAGLE_SOURCE" remote add origin https://github.com/SafeAILab/EAGLE.git
  public_git -C "$HPML_EAGLE_SOURCE" fetch --depth=1 --no-tags origin "$HPML_EAGLE_REVISION"
  public_git -C "$HPML_EAGLE_SOURCE" checkout --quiet --detach "$HPML_EAGLE_REVISION"
fi
[[ -d "$HPML_EAGLE_SOURCE/.git" && ! -L "$HPML_EAGLE_SOURCE/.git" ]] || \
  fail 'EAGLE source must be an independent Git checkout.'
[[ $(public_git -C "$HPML_EAGLE_SOURCE" config --get remote.origin.url) == \
   https://github.com/SafeAILab/EAGLE.git ]] || fail 'EAGLE source has a different origin.'
[[ $(public_git -C "$HPML_EAGLE_SOURCE" rev-parse HEAD) == "$HPML_EAGLE_REVISION" ]] || \
  fail 'EAGLE source has a different revision.'
[[ -z $(public_git -C "$HPML_EAGLE_SOURCE" status --porcelain=v1 --untracked-files=all) ]] || \
  fail 'Preserving local changes in the EAGLE checkout.'

# Bootstrap uv through an isolated standard-library venv and official PyPI.
uv_bin="$HPML_UV_BOOTSTRAP/bin/uv"
unset UV_INDEX UV_INDEX_URL UV_EXTRA_INDEX_URL UV_DEFAULT_INDEX UV_FIND_LINKS
if [[ "$mode" == setup ]]; then
  python3 -m venv "$HPML_UV_BOOTSTRAP"
  "$HPML_UV_BOOTSTRAP/bin/python" -m pip --isolated install --disable-pip-version-check \
    --index-url https://pypi.org/simple --only-binary=:all: 'uv==0.13.8'
  "$uv_bin" --no-config python install "$HPML_EAGLE_PYTHON" --no-bin
  "$uv_bin" --no-config venv --managed-python --python "$HPML_EAGLE_PYTHON" "$HPML_EAGLE_VENV"
else
  # The original failed setup predates a completion record. Verify its concrete
  # paths, isolated venvs, bootstrap version and installed pins before resuming.
  "$HPML_UV_BOOTSTRAP/bin/python" - <<'PYBOOT'
import importlib.metadata as metadata
import os
from pathlib import Path
import sys
expected = Path(os.environ['HPML_UV_BOOTSTRAP']).resolve()
if Path(sys.prefix).resolve() != expected or sys.prefix == sys.base_prefix:
    raise SystemExit('The bootstrap interpreter is outside its isolated environment.')
if metadata.version('uv') != '0.13.8':
    raise SystemExit('Resume requires uv 0.13.8 in the bootstrap environment.')
PYBOOT
  [[ $("$uv_bin" --version) == 'uv 0.13.8'* ]] || fail 'The uv executable differs from its package pin.'
  "$HPML_EAGLE_VENV/bin/python" - <<'PYRESUME'
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse
venv = Path(os.environ['HPML_EAGLE_VENV']).resolve()
if Path(sys.prefix).resolve() != venv or sys.prefix == sys.base_prefix:
    raise SystemExit('The target interpreter is outside its isolated environment.')
if tuple(sys.version_info[:3]) != tuple(map(int, os.environ['HPML_EAGLE_PYTHON'].split('.'))):
    raise SystemExit('Resume requires the existing Python 3.10.22 interpreter.')
runtime = Path(sys.base_prefix).resolve()
if Path(os.environ['UV_PYTHON_INSTALL_DIR']).resolve() not in runtime.parents:
    raise SystemExit('The Python runtime must belong to this scratch installation.')
for key in ('HPML_UV_BOOTSTRAP', 'HPML_EAGLE_VENV'):
    cfg = Path(os.environ[key]) / 'pyvenv.cfg'
    if cfg.is_symlink() or cfg.stat().st_uid != os.getuid():
        raise SystemExit(f'Expected an owned environment configuration: {cfg}')
    values = dict(line.split('=', 1) for line in cfg.read_text().splitlines() if '=' in line)
    values = {key.strip(): value.strip() for key, value in values.items()}
    if values.get('include-system-site-packages', '').lower() != 'false':
        raise SystemExit('Resume requires isolated environment site-packages.')
expected = {'torch': '2.6.0+cu124', 'transformers': '4.53.1',
            'accelerate': '0.26.0', 'numpy': '1.26.4',
            'setuptools': '80.9.0', 'wheel': '0.45.1', 'eagle-llm': '3.0.0'}
for package, version in expected.items():
    try:
        installed = metadata.version(package)
    except metadata.PackageNotFoundError:
        continue
    if installed != version:
        raise SystemExit(f'Preserving {package} {installed}; resume expects {version}.')
constraints = venv / 'hpml-constraints.txt'
if constraints.exists() or constraints.is_symlink():
    if constraints.is_symlink() or constraints.stat().st_uid != os.getuid():
        raise SystemExit('The existing constraints file must belong to this environment owner.')
    expected_lines = [f'{name}=={version}' for name, version in expected.items()
                      if name != 'eagle-llm']
    if constraints.read_text().splitlines() != expected_lines:
        raise SystemExit('Preserving the existing constraints file; its pins differ from this setup.')
try:
    distribution = metadata.distribution('eagle-llm')
except metadata.PackageNotFoundError:
    pass
else:
    origin = json.loads(distribution.read_text('direct_url.json') or '{}')
    location = urlparse(origin.get('url', ''))
    expected_source = Path(os.environ['HPML_EAGLE_SOURCE']).resolve()
    if (location.scheme != 'file' or Path(unquote(location.path)).resolve() != expected_source
            or not origin.get('dir_info', {}).get('editable')):
        raise SystemExit('The installed EAGLE package has a different source path.')
completion = venv / 'hpml-setup.json'
if completion.exists() or completion.is_symlink():
    if completion.is_symlink() or completion.stat().st_uid != os.getuid():
        raise SystemExit('The existing setup record must belong to this environment owner.')
    record = json.loads(completion.read_text())
    if record.get('source_revision') != os.environ['HPML_EAGLE_REVISION']:
        raise SystemExit('The existing setup record identifies a different source revision.')
print('Verified the existing Python 3.10.22 environment and installed core package pins.')
PYRESUME
fi

# The PyTorch page supplies the CUDA wheel. PyPI supplies its NVIDIA dependencies,
# avoiding the pypi.nvidia.com redirect observed from the full cu124 index.
# https://download.pytorch.org/whl/cu124/torch/
# https://pypi.org/project/nvidia-curand-cu12/10.3.5.147/
"$uv_bin" --no-config pip install --python "$HPML_EAGLE_VENV/bin/python" \
  --index-url https://pypi.org/simple \
  --find-links https://download.pytorch.org/whl/cu124/torch/ 'torch==2.6.0+cu124'
cat > "$HPML_EAGLE_VENV/hpml-constraints.txt" <<'PINS'
torch==2.6.0+cu124
transformers==4.53.1
accelerate==0.26.0
numpy==1.26.4
setuptools==80.9.0
wheel==0.45.1
PINS
"$uv_bin" --no-config pip install --python "$HPML_EAGLE_VENV/bin/python" \
  --index-url https://pypi.org/simple --constraint "$HPML_EAGLE_VENV/hpml-constraints.txt" \
  'transformers==4.53.1' 'accelerate==0.26.0' 'numpy==1.26.4' \
  'setuptools==80.9.0' 'wheel==0.45.1'
# Verify dependency consistency while accounting for EAGLE's documented legacy
# metadata if a previous attempt already reached the editable installation.
"$HPML_EAGLE_VENV/bin/python" - <<'PYCHECK'
from importlib import metadata
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
errors = []
for distribution in metadata.distributions():
    if canonicalize_name(distribution.metadata['Name']) == 'eagle-llm':
        continue
    for raw in distribution.requires or []:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate({'extra': ''}):
            continue
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            errors.append(f'{distribution.metadata["Name"]} requires {requirement}.')
            continue
        if requirement.specifier and not requirement.specifier.contains(version, prereleases=True):
            errors.append(f'{distribution.metadata["Name"]} requires {requirement}; found {version}.')
if errors:
    raise SystemExit('\n'.join(errors))
print('Runtime dependencies satisfy their declared requirements.')
PYCHECK

# setup.py retains older Torch/Transformers/Accelerate declarations. The current
# requirements.txt supplies the reviewed core versions. Model inference uses the
# dependencies above; the upstream web, API and evaluation extras are separate.
"$uv_bin" --no-config pip install --python "$HPML_EAGLE_VENV/bin/python" \
  --no-deps --no-build-isolation --editable "$HPML_EAGLE_SOURCE"
"$uv_bin" --no-config pip freeze --python "$HPML_EAGLE_VENV/bin/python" \
  > "$HPML_EAGLE_VENV/hpml-installed.txt"

# Import checks use offline mode and instantiate no model or GPU workload.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$HPML_EAGLE_VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import torch
import transformers
import accelerate
from eagle.model.ea_model import EaModel
expected = {'torch': '2.6.0+cu124', 'transformers': '4.53.1', 'accelerate': '0.26.0'}
for package, version in expected.items():
    if metadata.version(package) != version:
        raise SystemExit(f'{package} differs from the reviewed version {version}.')
if tuple(sys.version_info[:3]) != tuple(map(int, os.environ['HPML_EAGLE_PYTHON'].split('.'))) or torch.version.cuda != '12.4':
    raise SystemExit('Python or the bundled CUDA runtime differs from the reviewed environment.')
record = {'python': sys.version, 'packages': expected, 'cuda_runtime': torch.version.cuda,
          'source_revision': os.environ['HPML_EAGLE_REVISION'],
          'source_path': os.environ['HPML_EAGLE_SOURCE'], 'slurm_job_id': os.environ['SLURM_JOB_ID'],
          'scope': 'Package and EaModel imports; model loading and GPU qualification remain pending.',
          'metadata_override': 'EAGLE editable installation uses --no-deps because setup.py retains older requirements.'}
path = Path(os.environ['HPML_EAGLE_VENV']) / 'hpml-setup.json'
path.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
PY
printf '\nEnvironment prepared: %s\n' "$HPML_EAGLE_VENV"
printf 'Activate with: source %q/bin/activate\n' "$HPML_EAGLE_VENV"
printf '%s\n' \
  'Keep HF_HOME and UV_CACHE_DIR under HPML_ROOT in subsequent shells.' \
  'Inside a GPU allocation, run this script with --check-gpu before loading models.' \
  'The installed package list records resolved transitive versions for this environment.'
