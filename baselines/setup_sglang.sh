#!/usr/bin/env bash
# Prepare and qualify the official SGLang CUDA 12.9 container on Insomnia.
# Source: https://github.com/sgl-project/sglang/tree/0bcd822377da7b5718e674eaf9c870d349424dd1
# Image: https://hub.docker.com/r/lmsysorg/sglang/tags?name=v0.5.19-cu129
set -Eeuo pipefail
umask 077
fail() { printf 'SGLang setup: %s\n' "$*" >&2; exit 1; }
mode=${1:---prepare}
case "$mode" in
  --prepare|--inspect|--check-gpu) [[ $# -le 1 ]] || fail 'Use one setup mode.' ;;
  --exec) shift; [[ $# -gt 0 ]] || fail 'Provide an inference command after --exec.' ;;
  -h|--help)
    printf '%s\n' 'Inside a Slurm CPU allocation: bash baselines/setup_sglang.sh --prepare' \
      'Inside a Slurm GPU allocation: bash baselines/setup_sglang.sh --check-gpu' \
      'Run an inference command in that container: bash baselines/setup_sglang.sh --exec COMMAND [ARG ...]' \
      'Set HPML_ROOT to your private scratch hpml-baselines directory.'
    exit 0 ;;
  *) fail 'Use --prepare, --inspect, --check-gpu or --exec.' ;;
esac
[[ $(uname -s) == Linux ]] || fail 'Run on a Linux Slurm compute node.'
[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || fail 'Start a Slurm compute allocation.'
[[ ${HPML_ROOT:-} == /* && ${HPML_ROOT:-} != / ]] || fail 'Set an absolute private HPML_ROOT.'
export HPML_ROOT
python3 - <<'PY'
import os
from pathlib import Path
import socket
import stat
import subprocess
job = subprocess.check_output(['scontrol', '--oneliner', 'show', 'job', os.environ['SLURM_JOB_ID']], text=True)
fields = dict(item.split('=', 1) for item in job.split() if '=' in item)
if fields.get('JobState') != 'RUNNING' or not fields.get('UserId', '').endswith(f'({os.getuid()})'):
    raise SystemExit('Slurm must confirm your running allocation.')
nodes = subprocess.check_output(['scontrol', 'show', 'hostnames', fields['NodeList']], text=True).splitlines()
if socket.gethostname().split('.')[0] not in {n.split('.')[0] for n in nodes}:
    raise SystemExit('Run on a compute node in your allocation.')
root = Path(os.environ['HPML_ROOT'])
if any(p.is_symlink() for p in (root, *root.parents)):
    raise SystemExit('Use ordinary directory components for HPML_ROOT.')
if not root.is_dir() or root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode) & 0o077:
    raise SystemExit('HPML_ROOT must be an owned private directory with mode 0700.')
for relative in ('containers', 'cache/apptainer-sglang', 'cache/sglang', 'cache/sglang/triton',
                 'cache/sglang/flashinfer', 'tmp/sglang', 'runs/sglang-setup'):
    path = root / relative
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise SystemExit(f'Expected ordinary directories: {path}')
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.stat().st_uid != os.getuid():
        raise SystemExit(f'Expected an owned directory: {path}')
PY
module load apptainer/1.4.5
[[ $(apptainer --version) == 'apptainer version 1.4.5' ]] || fail 'Expected Apptainer 1.4.5.'
export APPTAINER_CACHEDIR="$HPML_ROOT/cache/apptainer-sglang"
export APPTAINER_TMPDIR="$HPML_ROOT/tmp/sglang"
image="$HPML_ROOT/containers/sglang-0.5.19-cu129-amd64.sif"
digest=sha256:1173f6a57ac63209b1bfb054e6a7366896fd3df5539c809bbd6d332694e91d15
source_revision=0bcd822377da7b5718e674eaf9c870d349424dd1
evidence="$HPML_ROOT/runs/sglang-setup"
if [[ "$mode" == --prepare && ! -e "$image" ]]; then
  lock="$HPML_ROOT/containers/.sglang-pull.lock"
  mkdir "$lock" || fail 'A SGLang image pull is already active or needs inspection.'
  trap 'rmdir "$lock"' EXIT
  partial="$HPML_ROOT/containers/sglang-0.5.19-cu129-amd64.partial.sif"
  [[ ! -e "$partial" ]] || fail 'Preserving an existing partial image for inspection.'
  script_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
  python3 "$script_directory/prepare_sglang_oci.py"
  [[ ${TMPDIR:-} == /local/slurm-job-* && -d "$TMPDIR" && -O "$TMPDIR" ]] || \
    fail 'Use the owned node-local TMPDIR supplied by the Insomnia allocation.'
  python3 - <<'PYSPACE'
import os
import shutil
if shutil.disk_usage(os.environ['TMPDIR']).free < 100 * 1024 ** 3:
    raise SystemExit('Image conversion requires 100 GiB of free node-local working space.')
PYSPACE
  conversion_work=$(mktemp -d "$TMPDIR/hpml-sglang.XXXXXX")
  mkdir "$conversion_work/cache" "$conversion_work/tmp"
  export APPTAINER_CACHEDIR="$conversion_work/cache"
  export APPTAINER_TMPDIR="$conversion_work/tmp"
  printf 'Converting the verified image in private node-local storage: %s\n' "$conversion_work"
  apptainer build --mksquashfs-args "-processors ${SLURM_CPUS_PER_TASK:-1} -mem 8G" \
    "$partial" "oci:$HPML_ROOT/cache/sglang-oci"
  mv "$partial" "$image"
  rmdir "$lock"
  trap - EXIT
fi
[[ -f "$image" && ! -L "$image" && -O "$image" ]] || fail 'Prepare the owned SGLang image first.'
apptainer inspect --json "$image" > "$evidence/container-inspect.json"
export HPML_SGLANG_IMAGE="$image" HPML_SGLANG_DIGEST="$digest" HPML_SGLANG_SOURCE="$source_revision"
python3 - <<'PY'
import json
import os
from pathlib import Path
labels = json.loads((Path(os.environ['HPML_ROOT']) / 'runs/sglang-setup/container-inspect.json').read_text())['data']['attributes']['labels']
if labels.get('org.opencontainers.image.revision') != os.environ['HPML_SGLANG_SOURCE']:
    raise SystemExit('The container source revision differs from the reviewed release.')
if labels.get('ai.sglang.image.tag') != 'lmsysorg/sglang:v0.5.19-cu129':
    raise SystemExit('The container label identifies a different release.')
PY
if [[ "$mode" == --prepare ]]; then
  sha256sum "$image" > "$evidence/image-sha256.txt"
  printf '%s\n' "docker://lmsysorg/sglang@$digest" > "$evidence/image-origin.txt"
fi

# This cluster module has no squashfuse. Reuse one private node-local directory
# for the image's runtime filesystem throughout this allocation.
[[ ${TMPDIR:-} == /local/slurm-job-* && -d "$TMPDIR" && -O "$TMPDIR" ]] || \
  fail 'Use the owned node-local TMPDIR supplied by the Insomnia allocation.'
runtime_base="$TMPDIR/hpml-sglang-runtime-0.5.19-cu129"
runtime_image="$runtime_base/rootfs"
export HPML_SGLANG_RUNTIME_BASE="$runtime_base"
python3 - <<'PYRUNTIME'
import json
import os
from pathlib import Path
import shutil
base = Path(os.environ['HPML_SGLANG_RUNTIME_BASE'])
if any(path.is_symlink() for path in (base, *base.parents)):
    raise SystemExit('The runtime directory requires ordinary path components.')
base.mkdir(mode=0o700, exist_ok=True)
if base.stat().st_uid != os.getuid():
    raise SystemExit('The runtime directory must belong to your account.')
image = Path(os.environ['HPML_SGLANG_IMAGE'])
expected = {'source_digest': os.environ['HPML_SGLANG_DIGEST'],
            'source_revision': os.environ['HPML_SGLANG_SOURCE'],
            'sif_path': str(image), 'sif_bytes': image.stat().st_size,
            'sif_mtime_ns': image.stat().st_mtime_ns}
marker = base / 'ready.json'
if marker.exists():
    if json.loads(marker.read_text()) != expected or not (base / 'rootfs').is_dir():
        raise SystemExit('The cached runtime directory differs from the prepared image.')
elif (base / 'rootfs').exists():
    raise SystemExit('Preserving an unfinished runtime extraction for inspection.')
elif shutil.disk_usage(base).free < 100 * 1024 ** 3:
    raise SystemExit('Runtime extraction requires 100 GiB of free node-local working space.')
(base / 'expected.json').write_text(json.dumps(expected, indent=2) + '\n')
PYRUNTIME
export APPTAINER_CACHEDIR="$runtime_base/cache"
export APPTAINER_TMPDIR="$runtime_base/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
if [[ ! -f "$runtime_base/ready.json" ]]; then
  mkdir "$runtime_base/.extract.lock" || fail 'Another runtime extraction is active.'
  trap 'rmdir "$runtime_base/.extract.lock"' EXIT
  apptainer build --sandbox "$runtime_image" "$image"
  cp "$runtime_base/expected.json" "$runtime_base/ready.json"
  rmdir "$runtime_base/.extract.lock"
  trap - EXIT
  printf 'Prepared the node-local runtime directory: %s\n' "$runtime_image"
fi
runtime=(apptainer exec --cleanenv --no-home --bind "$HPML_ROOT:/hpml" --pwd /hpml
  --env HPML_ROOT=/hpml --env SLURM_JOB_ID="$SLURM_JOB_ID"
  --env HF_HOME=/hpml/cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1
  --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1
  --env XDG_CACHE_HOME=/hpml/cache/sglang --env TRITON_CACHE_DIR=/hpml/cache/sglang/triton
  --env TORCHINDUCTOR_CACHE_DIR=/hpml/cache/sglang/torchinductor --env TORCH_HOME=/hpml/cache/sglang/torch
  --env SGLANG_JIT_CACHE_DIR=/hpml/cache/sglang/jit --env TORCH_EXTENSIONS_DIR=/hpml/cache/sglang/torch-extensions
  --env CUDA_CACHE_PATH=/hpml/cache/sglang/cuda
  --env FLASHINFER_WORKSPACE_BASE=/hpml/cache/sglang/flashinfer)
if [[ ${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-0} == 1 ]]; then
  runtime+=(--env SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1)
fi
if [[ "$mode" == --check-gpu || "$mode" == --exec ]]; then
  [[ -n ${CUDA_VISIBLE_DEVICES:-} && ${CUDA_VISIBLE_DEVICES:-} != NoDevFiles ]] || fail 'Request a visible GPU allocation.'
  runtime+=(--nv --env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi
if [[ "$mode" == --exec ]]; then
  exec "${runtime[@]}" "$runtime_image" "$@"
fi
"${runtime[@]}" "$runtime_image" python3 - <<'PY' | tee "$evidence/package-inspection-${SLURM_JOB_ID}.json"
from importlib import metadata
import json
import os
import platform
import torch
import sglang
names = ('sglang', 'torch', 'transformers', 'sglang-kernel', 'flashinfer-python', 'triton', 'cuda-python')
packages = {name: metadata.version(name) for name in names}
if packages['sglang'] != '0.5.19' or torch.version.cuda != '12.9':
    raise SystemExit('The container package versions differ from the reviewed CUDA 12.9 release.')
if os.environ.get('SGLANG_BUILD_COMMIT') != '0bcd822377da7b5718e674eaf9c870d349424dd1':
    raise SystemExit('The container build commit differs from the reviewed source.')
print(json.dumps({'python': platform.python_version(), 'packages': packages,
                  'cuda_runtime': torch.version.cuda, 'build_commit': os.environ['SGLANG_BUILD_COMMIT'],
                  'scope': 'Package imports and container identity'}, indent=2))
PY
if [[ "$mode" == --check-gpu ]]; then
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$evidence/gpu-${SLURM_JOB_ID}.csv"
  "${runtime[@]}" "$runtime_image" python3 - <<'PY' | tee "$evidence/gpu-check-${SLURM_JOB_ID}.json"
import json
import torch
import flashinfer
import sgl_kernel
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in the SGLang container.')
with torch.inference_mode():
    x = torch.ones((32, 32), dtype=torch.bfloat16, device='cuda')
    y = x @ x
    torch.cuda.synchronize()
    if not torch.equal(y, torch.full_like(y, 32)):
        raise SystemExit('The BF16 matrix multiplication result differs from 32.')
    values = torch.ones((2, 128), dtype=torch.bfloat16, device='cuda')
    weight = torch.ones((128,), dtype=torch.bfloat16, device='cuda')
    normalized = flashinfer.norm.rmsnorm(values, weight, eps=1e-6)
    torch.cuda.synchronize()
    if not torch.allclose(normalized, torch.ones_like(normalized), atol=0.01, rtol=0.01):
        raise SystemExit('The FlashInfer RMSNorm result differs from the expected value.')
print(json.dumps({'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                  'gpu': torch.cuda.get_device_name(0), 'compute_capability': torch.cuda.get_device_capability(0),
                  'bf16_matmul_32x32': 'passed', 'flashinfer_bf16_rmsnorm': 'passed',
                  'sgl_kernel_import': 'passed', 'scope': 'Kernel compatibility; model qualification remains separate'}, indent=2))
PY
  "${runtime[@]}" "$runtime_image" python3 -m sglang.launch_server --help > "$evidence/server-help-${SLURM_JOB_ID}.txt"
fi
if [[ "$mode" == --prepare ]]; then
  if [[ -n ${conversion_work:-} ]]; then rm -rf -- "$conversion_work"; fi
fi
printf 'SGLang %s completed in Slurm job %s. Evidence: %s\n' "$mode" "$SLURM_JOB_ID" "$evidence"
