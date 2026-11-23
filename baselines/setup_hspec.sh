#!/usr/bin/env bash
# Prepare the released P-EAGLE/H-Spec inference stack in private scratch.
# Usage: HPML_ROOT=/your/scratch/hpml-baselines bash baselines/setup_hspec.sh
# Resume: bash baselines/setup_hspec.sh --resume
# GPU runtime check: bash baselines/setup_hspec.sh --check-gpu
# Pretrained model checks: bash baselines/setup_hspec.sh --check-models
set -Eeuo pipefail
umask 077
fail() { printf 'H-Spec setup: %s\n' "$*" >&2; exit 1; }
mode=setup
case "${1:-}" in
  '') ;;
  --resume) mode=resume ;;
  --finish-install) mode=finish ;;
  --check-gpu) mode=gpu ;;
  --check-models) mode=models ;;
  -h|--help)
    printf '%s\n' 'Run inside an Insomnia Slurm compute allocation with HPML_ROOT set.' \
      'Setup uses envs/hspec-cu129-py313 and source/hspec-cu129-py313.' \
      'Use --resume for a verified partial setup; --check-gpu for a GPU allocation.' \
      'Use --finish-install to complete dependencies/import checks after the patched packages are installed.' \
      'Use --check-models after the GPU check and verified direct-Meta target import.' \
      'The script prepares inference packages and qualification evidence.'
    exit 0 ;;
  *) fail 'Use no argument, --resume, --finish-install, --check-gpu, or --check-models.' ;;
esac
[[ $# -le 1 ]] || fail 'Provide at most one argument.'
[[ $(uname -s) == Linux ]] || fail 'Run on a Linux compute node.'
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_JOB_NODELIST:-} ]] || fail 'A Slurm compute allocation is required.'
[[ ${HPML_ROOT:-} == /* && ${HPML_ROOT:-} != / ]] || fail 'Set HPML_ROOT to an absolute private scratch directory.'
for command_name in python3 git scontrol; do command -v "$command_name" >/dev/null || fail "Missing $command_name."; done
current_host=$(hostname -s)
found_host=false
while IFS= read -r host; do
  [[ ${host%%.*} != "$current_host" ]] || found_host=true
done < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
$found_host || fail 'Run on a compute node belonging to the active job.'

unset PYTHONPATH PYTHONHOME UV_INDEX UV_INDEX_URL UV_EXTRA_INDEX_URL UV_DEFAULT_INDEX UV_FIND_LINKS
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export UV_CACHE_DIR="$HPML_ROOT/cache/uv" UV_PYTHON_INSTALL_DIR="$HPML_ROOT/python"
export UV_PYTHON_INSTALL_BIN=0 UV_LINK_MODE=copy
export XDG_CACHE_HOME="$HPML_ROOT/cache" PIP_CACHE_DIR="$HPML_ROOT/cache/pip"
export HF_HOME="$HPML_ROOT/cache/huggingface" TMPDIR="$HPML_ROOT/tmp"
export TRITON_CACHE_DIR="$HPML_ROOT/cache/hspec/triton"
export TORCHINDUCTOR_CACHE_DIR="$HPML_ROOT/cache/hspec/torchinductor"
export CUDA_CACHE_PATH="$HPML_ROOT/cache/hspec/cuda"
export FLASHINFER_WORKSPACE_BASE="$HPML_ROOT/cache/hspec"
export HPML_HSPEC_ENV="$HPML_ROOT/envs/hspec-cu129-py313"
export PATH="$HPML_HSPEC_ENV/bin:$PATH"
export HPML_HSPEC_BUILD="$HPML_ROOT/source/hspec-cu129-py313"
export HPML_PUBLIC_SOURCE="$HPML_ROOT/source/hpml-project/external"
export HPML_HSPEC_MODE="$mode"
export HPML_HSPEC_PYTHON=3.13.18
export HPML_HSPEC_REV=de911876905440bfd8edfa78b482050f7ac0279c
export HPML_VLLM_REV=752a3a504485790a2e8491cacbb35c137339ad34
export HPML_SPECULATORS_REV=3251ed020cfc954717c1d075d83dca8721713f0c
export HPML_VLLM_PATCH_SHA=2c59f719bcd1d5747971575f1e65f2770c05891487b353f38d72225f0ec52bdd
export HPML_SPECULATORS_PATCH_SHA=8fd05dd3fbed949b47adc81aef9aeb1726011347015ffaaeecedc839fe472860
export HPML_VLLM_WHEEL_SHA=9e206f370c934a2d4b6b1f05d3d09708d344e05d80260189ef19f60755709431
export HPML_VLLM_WHEEL_URL='https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl'
export HPML_FLASHINFER_WHEEL_URL='https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.13/flashinfer_jit_cache-0.6.13%2Bcu129-cp39-abi3-manylinux_2_28_x86_64.whl'
export HPML_FLASHINFER_WHEEL_SHA=0af546a6a0db3470b0a52b43c8158fc85358852ab1b0a8a42d83a411f0aaa369
export HPML_HSPEC_UV="$HPML_ROOT/envs/uv-bootstrap/bin/uv"
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0 GIT_LFS_SKIP_SMUDGE=1

python3 - <<'PY'
import os, stat
from pathlib import Path
root = Path(os.environ["HPML_ROOT"])
for path in (root, *root.parents):
    if path.is_symlink():
        raise SystemExit(f"Choose an ordinary scratch directory: {path}")
if not root.is_dir() or root.stat().st_uid != os.getuid():
    raise SystemExit("HPML_ROOT must exist and belong to this account.")
if stat.S_IMODE(root.stat().st_mode) & 0o077:
    raise SystemExit("Keep HPML_ROOT private with mode 0700.")
for key in ("HPML_HSPEC_ENV", "HPML_HSPEC_BUILD", "HPML_PUBLIC_SOURCE"):
    path = Path(os.environ[key])
    if path.is_symlink() or (path.exists() and (not path.is_dir() or path.stat().st_uid != os.getuid())):
        raise SystemExit(f"Preserving unexpected path: {path}")
for name in ("cache/uv", "cache/pip", "cache/hspec/triton", "cache/hspec/torchinductor", "cache/hspec/cuda", "tmp", "python", "runs"):
    (root / name).mkdir(mode=0o700, parents=True, exist_ok=True)
PY
[[ -x "$HPML_HSPEC_UV" ]] || fail 'Prepare the reviewed uv-bootstrap environment first.'
[[ $("$HPML_HSPEC_UV" --version) == 'uv 0.13.8'* ]] || fail 'Expected uv 0.13.8.'
python_bin="$HPML_HSPEC_ENV/bin/python"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ "$mode" == models ]]; then
  [[ -x "$python_bin" ]] || fail 'Complete environment installation first.'
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  "$python_bin" "$script_dir/qualify_hspec.py" \
    --output-dir "$HPML_ROOT/runs/hspec-models-$SLURM_JOB_ID"
  exit 0
fi

if [[ "$mode" == gpu ]]; then
  [[ -x "$python_bin" ]] || fail 'Complete CPU environment installation first.'
  [[ -n ${CUDA_VISIBLE_DEVICES:-} && ${CUDA_VISIBLE_DEVICES:-} != NoDevFiles ]] || fail 'Request a GPU allocation.'
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  "$python_bin" - <<'PY'
import importlib.metadata as md
import json, os, socket, subprocess
from pathlib import Path
import torch
from vllm.model_executor.models import qwen3_dflash_mamba_attn_hybrid
import speculators.models.mamba_attn_hybrid
from speculators.models.peagle import PEagleDraftModel
import vllm._C_stable_libtorch
from vllm import _custom_ops as ops
from vllm.model_executor.models.qwen3_dflash_hybrid_blocks import _dflash_hybrid_ssd_scan
if torch.__version__ != "2.11.0+cu129" or torch.version.cuda != "12.9":
    raise SystemExit("The Torch CUDA runtime differs from the recorded selection.")
if not torch.cuda.is_available():
    raise SystemExit("CUDA initialization failed on the allocated GPU.")
with torch.inference_mode():
    x = torch.ones((32, 128), dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x)
    weight = torch.ones(128, dtype=torch.bfloat16, device="cuda")
    ops.rms_norm(out, x, weight, 1e-6)
    torch.cuda.synchronize()
    if not torch.allclose(out.float(), torch.ones_like(out).float(), atol=0.01, rtol=0):
        raise SystemExit("vLLM RMS normalization returned an unexpected result.")
    product = x[:, :32] @ x[:, :32]
    torch.cuda.synchronize()
    if not torch.equal(product, torch.full_like(product, 32)):
        raise SystemExit("BF16 matrix multiplication failed.")
    # Exercise the runtime's compiler path, which can expose driver/PTX incompatibility.
    compiled = torch.compile(lambda value: value + 1, fullgraph=True)
    compiled_out = compiled(x)
    torch.cuda.synchronize()
    if not torch.equal(compiled_out, torch.full_like(x, 2)):
        raise SystemExit("Compiled elementwise CUDA execution failed.")
    # Exercise the released hybrid scan with an analytically known result.
    scan_x = torch.ones((32, 4, 16), dtype=torch.bfloat16, device="cuda")
    scan_dt = torch.zeros((32, 4), dtype=torch.bfloat16, device="cuda")
    scan_bc = torch.zeros((32, 1, 16), dtype=torch.bfloat16, device="cuda")
    scan_a = -torch.ones(4, dtype=torch.float32, device="cuda")
    scan_d = torch.ones(4, dtype=torch.float32, device="cuda")
    scan_bias = torch.zeros(4, dtype=torch.float32, device="cuda")
    scan_out = _dflash_hybrid_ssd_scan(
        scan_x, scan_dt, scan_a, scan_bc, scan_bc, scan_d, scan_bias, 1
    )
    torch.cuda.synchronize()
    if not torch.allclose(scan_out.float(), scan_x.float(), atol=0.01, rtol=0):
        raise SystemExit("H-Spec's hybrid scan returned an unexpected result.")
record = {
    "job_id": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(),
    "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
    "vllm": md.version("vllm"), "speculators": md.version("speculators"),
    "gpu": torch.cuda.get_device_name(0),
    "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
    "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
    "nvidia_driver_versions": sorted(set(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True).splitlines())),
    "checks": {"patched_imports": "passed", "bf16_matmul": "passed",
               "vllm_rms_norm": "passed", "torch_compile": "passed",
               "hybrid_ssd_scan": "passed"},
    "scope": "Basic inference runtime checks. Target loading and speculative outputs require separate checks.",
}
destination = Path(os.environ["HPML_HSPEC_ENV"]) / "hpml-gpu-check.json"
destination.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
  exit 0
fi

lock="$HPML_ROOT/.hspec-setup.lock"
mkdir "$lock" || fail 'Another H-Spec setup or an unfinished lock exists.'
trap 'rmdir "$lock"' EXIT
trap 'printf "H-Spec setup stopped at line %s. Preserve the partial environment and inspect the job log before --resume.\n" "$LINENO" >&2' ERR

public_git() {
  git -c core.hooksPath=/dev/null -c core.fsmonitor=false -c submodule.recurse=false \
    -c credential.helper= "$@"
}
for name in hspec vllm speculators; do
  case "$name" in
    hspec) expected="$HPML_HSPEC_REV" ;;
    vllm) expected="$HPML_VLLM_REV" ;;
    speculators) expected="$HPML_SPECULATORS_REV" ;;
  esac
  source_path="$HPML_PUBLIC_SOURCE/$name"
  [[ -d "$source_path/.git" && ! -L "$source_path/.git" ]] || fail "Missing independent source: $name."
  [[ $(public_git -C "$source_path" rev-parse HEAD) == "$expected" ]] || fail "Different source revision: $name."
  [[ -z $(public_git -C "$source_path" status --porcelain=v1 --untracked-files=all) ]] || fail "Preserving source edits: $name."
done

if [[ "$mode" == setup ]]; then
  [[ ! -e "$HPML_HSPEC_ENV" && ! -e "$HPML_HSPEC_BUILD" ]] || fail 'Preserving existing setup paths; inspect and use --resume.'
  mkdir "$HPML_HSPEC_BUILD"
  for name in vllm speculators; do
    public_git clone --quiet --no-hardlinks --no-checkout "$HPML_PUBLIC_SOURCE/$name" "$HPML_HSPEC_BUILD/$name"
    case "$name" in
      vllm) expected="$HPML_VLLM_REV"; origin=https://github.com/vllm-project/vllm.git ;;
      speculators) expected="$HPML_SPECULATORS_REV"; origin=https://github.com/vllm-project/speculators.git ;;
    esac
    public_git -C "$HPML_HSPEC_BUILD/$name" remote set-url origin "$origin"
    public_git -C "$HPML_HSPEC_BUILD/$name" config core.hooksPath /dev/null
    public_git -C "$HPML_HSPEC_BUILD/$name" checkout --quiet --detach "$expected"
  done
  "$HPML_HSPEC_UV" --no-config python install "$HPML_HSPEC_PYTHON" --no-bin
  "$HPML_HSPEC_UV" --no-config venv --managed-python --python "$HPML_HSPEC_PYTHON" "$HPML_HSPEC_ENV"
else
  [[ -x "$python_bin" && -d "$HPML_HSPEC_BUILD/vllm/.git" && -d "$HPML_HSPEC_BUILD/speculators/.git" ]] || fail 'Resume needs the existing isolated environment and build sources.'
  "$python_bin" - <<'PY'
import os, sys
from pathlib import Path
if Path(sys.prefix).resolve() != Path(os.environ["HPML_HSPEC_ENV"]).resolve() or sys.prefix == sys.base_prefix:
    raise SystemExit("The existing interpreter belongs to another environment.")
if tuple(sys.version_info[:3]) != tuple(map(int, os.environ["HPML_HSPEC_PYTHON"].split("."))):
    raise SystemExit("The existing interpreter differs from the pinned Python version.")
PY
fi

python3 - <<'PY'
import hashlib, os, subprocess
from pathlib import Path
base=Path(os.environ["HPML_HSPEC_BUILD"])
source=Path(os.environ["HPML_PUBLIC_SOURCE"])/"hspec"
for name in ("vllm", "speculators"):
    revision=os.environ["HPML_"+name.upper()+"_REV"]
    directory=base/name
    command=["git","-c","core.hooksPath=/dev/null","-c","core.fsmonitor=false","-C",str(directory)]
    if subprocess.check_output(command+["rev-parse","HEAD"],text=True).strip()!=revision:
        raise SystemExit(f"Preserving different build revision: {name}")
    patch=source/(name+".patch")
    digest=hashlib.sha256(patch.read_bytes()).hexdigest()
    if digest!=os.environ["HPML_"+name.upper()+"_PATCH_SHA"]:
        raise SystemExit(f"The released patch changed: {name}")
    reverse=subprocess.run(command+["apply","--reverse","--check",str(patch)],capture_output=True)
    if reverse.returncode:
        subprocess.run(command+["apply","--check",str(patch)],check=True)
        subprocess.run(command+["apply",str(patch)],check=True)
    subprocess.run(command+["diff","--check"],check=True)
    print(f"Verified released {name} patch {digest}",flush=True)
PY

constraints="$HPML_HSPEC_ENV/hpml-constraints.txt"
export HPML_HSPEC_CONSTRAINTS="$constraints"
dependency_constraints=()
if [[ -f "$script_dir/hspec-requirements.txt" ]]; then
  dependency_constraints=(--constraint "$script_dir/hspec-requirements.txt")
fi
"$python_bin" - <<'PY'
import os
from pathlib import Path
previous="torch==2.11.0+cu129\ntorchaudio==2.11.0+cu129\ntorchvision==0.26.0+cu129\ntransformers==5.13.0\n"
text=previous+"torchcodec==0.17.0+cpu\n"
p=Path(os.environ["HPML_HSPEC_CONSTRAINTS"])
if p.exists() and p.read_text() not in (previous,text):
    raise SystemExit("Preserving a modified constraints file.")
p.write_text(text)
PY
uv_install() {
  "$HPML_HSPEC_UV" --no-config pip install --python "$python_bin" \
    --index-url https://pypi.org/simple --constraint "$constraints" \
    "${dependency_constraints[@]}" "$@"
}
if [[ "$mode" != finish ]]; then
uv_install --find-links https://download.pytorch.org/whl/cu129/torch/ \
  --find-links https://download.pytorch.org/whl/cu129/torchaudio/ \
  --find-links https://download.pytorch.org/whl/cu129/torchvision/ \
  'torch==2.11.0+cu129' 'torchaudio==2.11.0+cu129' 'torchvision==0.26.0+cu129'
fi
# Use the official CPU media codec with the selected CUDA 12.9 Torch runtime.
uv_install --find-links https://download.pytorch.org/whl/cpu/torchcodec/ 'torchcodec==0.17.0+cpu'
export HPML_FLASHINFER_WHEEL="$HPML_HSPEC_BUILD/flashinfer_jit_cache-0.6.13+cu129-cp39-abi3-manylinux_2_28_x86_64.whl"
"$python_bin" - <<'PY'
import hashlib, os, urllib.request
from pathlib import Path
destination=Path(os.environ["HPML_FLASHINFER_WHEEL"])
if not destination.exists():
    temporary=destination.with_suffix(".whl.partial")
    if temporary.exists():
        raise SystemExit(f"Inspect and preserve or remove the interrupted wheel download before resuming: {temporary}")
    request=urllib.request.Request(os.environ["HPML_FLASHINFER_WHEEL_URL"],headers={"User-Agent":"HPML-pretrained-inference-setup"})
    with urllib.request.urlopen(request,timeout=120) as response,temporary.open("xb") as output:
        count=0
        while block:=response.read(8*1024*1024):
            output.write(block)
            count+=len(block)
            if count % (256*1024*1024)==0:
                print(f"FlashInfer cache wheel downloaded: {count} bytes",flush=True)
    temporary.rename(destination)
digest=hashlib.file_digest(destination.open("rb"),"sha256").hexdigest()
if digest!=os.environ["HPML_FLASHINFER_WHEEL_SHA"]:
    raise SystemExit("The FlashInfer wheel differs from the official release digest.")
print(f"Verified official FlashInfer CUDA12.9 cache wheel: {digest}",flush=True)
PY
uv_install "$HPML_FLASHINFER_WHEEL"
if [[ "$mode" != finish ]]; then
uv_install 'transformers==5.13.0' 'setuptools==80.9.0' 'setuptools-scm>=8.0' \
  'setuptools-rust>=1.9.0' 'cmake>=3.26.1' wheel jinja2 ninja packaging

export HPML_VLLM_WHEEL="$HPML_HSPEC_BUILD/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
"$python_bin" - <<'PY'
import hashlib, os, urllib.request
from pathlib import Path
destination=Path(os.environ["HPML_VLLM_WHEEL"])
if not destination.exists():
    temporary=destination.with_suffix(".whl.partial")
    request=urllib.request.Request(os.environ["HPML_VLLM_WHEEL_URL"],headers={"User-Agent":"HPML-pretrained-inference-setup"})
    with urllib.request.urlopen(request,timeout=120) as response,temporary.open("xb") as output:
        while block:=response.read(8*1024*1024):
            output.write(block)
    temporary.rename(destination)
digest=hashlib.file_digest(destination.open("rb"),"sha256").hexdigest()
if digest!=os.environ["HPML_VLLM_WHEEL_SHA"]:
    raise SystemExit("The downloaded vLLM wheel differs from the reviewed release digest.")
print(f"Verified official CUDA12.9 vLLM wheel: {digest}",flush=True)
PY
uv_install "$HPML_VLLM_WHEEL"
# All patch changes are Python; reuse the matching official compiled libraries.
# The local version identifies the patched release even in a tag-free source clone.
VLLM_TARGET_DEVICE=cuda VLLM_USE_PRECOMPILED=1 \
  VLLM_VERSION_OVERRIDE=0.25.1+cu129.hspec \
  VLLM_PRECOMPILED_WHEEL_LOCATION="$HPML_VLLM_WHEEL" \
  VLLM_PRECOMPILED_WHEEL_COMMIT="$HPML_VLLM_REV" \
  VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 \
  uv_install --no-build-isolation --no-deps "$HPML_HSPEC_BUILD/vllm"
# Speculators builds in its own isolated build environment.
uv_install --editable "$HPML_HSPEC_BUILD/speculators"
fi
"$HPML_HSPEC_UV" --no-config pip check --python "$python_bin"
"$HPML_HSPEC_UV" --no-config pip freeze --python "$python_bin" > "$HPML_HSPEC_ENV/hpml-installed.txt"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$python_bin" - <<'PY'
import importlib.metadata as md
import json, os, socket, sys
from pathlib import Path
import torch
import transformers
from vllm.model_executor.models import qwen3_dflash_mamba_attn_hybrid
import speculators.models.mamba_attn_hybrid
if torch.__version__ != "2.11.0+cu129" or torch.version.cuda != "12.9":
    raise SystemExit("Torch version/runtime differs from the selected CUDA12.9 wheel.")
if transformers.__version__ != "5.13.0":
    raise SystemExit("Transformers differs from the selected version.")
record={
    "python":sys.version,"torch":torch.__version__,"cuda_runtime":torch.version.cuda,
    "transformers":transformers.__version__,"vllm":md.version("vllm"),
    "speculators":md.version("speculators"),"job_id":os.environ["SLURM_JOB_ID"],
    "node":socket.gethostname(),"source_revisions":{
        name:os.environ["HPML_"+name+"_REV"] for name in ("HSPEC","VLLM","SPECULATORS")},
    "patch_sha256":{name:os.environ["HPML_"+name+"_PATCH_SHA"] for name in ("VLLM","SPECULATORS")},
    "vllm_wheel_sha256":os.environ["HPML_VLLM_WHEEL_SHA"],
    "flashinfer_cache_wheel_sha256":os.environ["HPML_FLASHINFER_WHEEL_SHA"],
    "patched_module_imports":"passed",
    "scope":"Package and patched model-class import checks. GPU kernels and target-model qualification have separate records."
}
path=Path(os.environ["HPML_HSPEC_ENV"])/"hpml-setup.json"
path.write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))
PY
printf 'H-Spec/P-EAGLE environment prepared at %s\n' "$HPML_HSPEC_ENV"
printf '%s\n' 'Next run --check-gpu in a one-GPU allocation before target loading.'
