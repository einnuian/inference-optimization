#!/usr/bin/env python3
"""List or download the reviewed pretrained model snapshots.

The default action lists artifacts using the Python standard library.
Download selected artifacts inside a running Slurm compute allocation:

    python baselines/download_checkpoints.py --download --only target eagle3

Set HF_HOME to your personal Insomnia scratch cache. Public drafters download
anonymously. The gated target uses an existing Hugging Face CLI sign-in
and requires approved Meta access.
The target requests the complete Transformers inference artifact set, model card,
license and use policy at its fixed revision. Public drafters request complete
repository snapshots. Existing cached files remain available.
The script records provenance in HF_HOME/hpml-provenance and performs no model
execution, package installation, training, or target substitution.

Artifact identities: baselines/core-methods.md (checked November 16, 2026).
API reference: https://huggingface.co/docs/huggingface_hub/package_reference/file_download
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import socket
import subprocess
import sys
import tempfile


@dataclass(frozen=True)
class Artifact:
    name: str
    repo_id: str
    revision: str
    role: str
    required_files: tuple[str, ...]
    download_files: tuple[str, ...] | None = None


# Public repository metadata checked November 16, 2026, without accessing weights:
# https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct/revision/0e9e39f249a16976918f6564b8830bc894c89659?blobs=true
# The four safetensors shards supply the Transformers weights. The original/
# directory contains Meta-format weights and tokenizer files for another loader.
TARGET_FILE_BYTES = {
    "LICENSE": 7627,
    "README.md": 44044,
    "USE_POLICY.md": 4691,
    "config.json": 855,
    "generation_config.json": 184,
    "model-00001-of-00004.safetensors": 4976698672,
    "model-00002-of-00004.safetensors": 4999802720,
    "model-00003-of-00004.safetensors": 4915916176,
    "model-00004-of-00004.safetensors": 1168138808,
    "model.safetensors.index.json": 23950,
    "special_tokens_map.json": 296,
    "tokenizer.json": 9085657,
    "tokenizer_config.json": 55351,
}
TARGET_FILES = tuple(TARGET_FILE_BYTES)

# Official pinned model metadata, checked November 16, 2026:
# https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct/revision/0e9e39f249a16976918f6564b8830bc894c89659?blobs=true
SMALL_FILE_BLOBS = {
    "LICENSE": "a7c3ca16cee30425ed6ad841a809590f2bcbf290",
    "README.md": "bbd5630a05b65c1a8b25141bd11ec44844107d58",
    "USE_POLICY.md": "81ebb55902285e8dd5804ccf423d17ffb2a622ee",
    "config.json": "0bb6fd75b3ad2fe988565929f329945262c2814e",
    "generation_config.json": "cc7276afd599de091142c6ed3005faf8a74aa257",
    "model.safetensors.index.json": "0fd8120f1c6acddc268ebc2583058efaf699a771",
    "special_tokens_map.json": "02ee80b6196926a5ad790a004d9efd6ab1ba6542",
    "tokenizer.json": "5cc5f00a5b203e90a27a3bd60d1ec393b07971e8",
    "tokenizer_config.json": "db88166e2bc4c799fd5d1ae643b75e84d03ee70e",
}
WEIGHT_SHA256 = {
    "model-00001-of-00004.safetensors": "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
    "model-00002-of-00004.safetensors": "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15",
    "model-00003-of-00004.safetensors": "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa",
    "model-00004-of-00004.safetensors": "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b",
}



ARTIFACTS = (
    Artifact("target", "meta-llama/Llama-3.1-8B-Instruct",
             "0e9e39f249a16976918f6564b8830bc894c89659",
             "Canonical target and tokenizer; approved Meta access required.",
             TARGET_FILES, TARGET_FILES),
    Artifact("eagle3", "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
             "ada412b672e293d682423de84a095447bf38a637",
             "Original EAGLE-3 research implementation.",
             ("config.json", "pytorch_model.bin")),
    Artifact("eagle3-sglang", "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B",
             "e5ed08d66f528a95ce89f5d4fd136a28f6def714",
             "Architecture conversion for SGLang EAGLE-3.",
             ("config.json", "pytorch_model.bin")),
    Artifact("dflash", "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat",
             "d3af30def9601abdd10810aba220d692f0e803f0",
             "Official UltraChat Llama DFlash drafter.",
             ("config.json", "model.safetensors")),
    Artifact("peagle", "weifanjiang/llama3.1-8b-it.speculators.peagle",
             "0db2ae706f7a314286516835006183efdcb5bb39",
             "H-Spec authors' P-EAGLE reproduction; patched Model Runner V1.",
             ("config.json", "model.safetensors")),
    Artifact("hspec", "weifanjiang/llama3.1-8b-it.speculators.hspec",
             "a43cf8a437816ddf894270ef03b23520dedcde88",
             "H-Spec Llama drafter; patched Model Runner V2.",
             ("config.json", "model.safetensors")),
    Artifact("pard2", "amd/PARD2-Llama-3.1-8B",
             "a509e315bfe79a0be8b832004bb3ee14acfdf86a",
             "PARD-2 drafter and feature adapter; target-copy identity needs qualification.",
             ("config.json", "model.safetensors", "warp_model.bin")),
    Artifact("dspark", "weifanjiang/llama3.1-8b-it.speculators.dspark",
             "110d99b89501e3007a893da937d9560e087243c0",
             "Optional H-Spec authors' DSpark reproduction for Llama.",
             ("config.json", "model.safetensors")),
)


class DownloadError(RuntimeError):
    """A prerequisite or snapshot verification failed."""


class PrivateArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # argparse's original message can echo an accidentally supplied credential.
        self.print_usage(sys.stderr)
        self.exit(2, "Invalid arguments. Use --help for supported options. "
                  "The target uses a cached Hugging Face CLI sign-in.\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = PrivateArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--list", dest="action", action="store_const", const="list",
                         help="List fixed artifact IDs and revisions (default).")
    actions.add_argument("--download", dest="action", action="store_const", const="download",
                         help="Download only artifacts explicitly selected with --only.")
    parser.set_defaults(action="list")
    parser.add_argument("--only", nargs="+", choices=[a.name for a in ARTIFACTS],
                        help="Select exact artifact names; required for --download.")
    args = parser.parse_args(argv)
    if args.action == "download" and not args.only:
        parser.error("--download requires --only")
    return args


def require_compute_allocation() -> tuple[str, str]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job_id):
        raise DownloadError("Enter a Slurm compute allocation before downloading.")
    try:
        job = subprocess.run(
            ["scontrol", "--oneliner", "show", "job", job_id],
            check=False, capture_output=True, text=True, timeout=20,
        )
        fields = dict(item.split("=", 1) for item in job.stdout.split() if "=" in item)
        owner = fields.get("UserId", "")
        if (job.returncode or fields.get("JobState") != "RUNNING"
                or not owner.endswith(f"({os.getuid()})")):
            raise DownloadError("Slurm must confirm a running allocation owned by your account.")
        node_list = fields.get("NodeList", "")
        if not node_list or node_list == "(null)":
            raise DownloadError("Slurm has assigned no compute node to this allocation.")
        nodes = subprocess.run(
            ["scontrol", "show", "hostnames", node_list],
            check=False, capture_output=True, text=True, timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise DownloadError("Slurm allocation verification failed. Check scontrol in a login-initialized shell.") from error
    host = socket.gethostname().split(".")[0]
    allocated = {node.split(".")[0] for node in nodes.stdout.splitlines()}
    if nodes.returncode or host not in allocated:
        raise DownloadError("Run the download on a compute node assigned to this allocation.")
    return job_id, host


def require_scratch_cache() -> Path:
    raw = os.environ.get("HF_HOME", "")
    if not raw or not Path(raw).is_absolute():
        raise DownloadError("Set HF_HOME to an absolute directory in your personal Insomnia scratch storage.")
    user = pwd.getpwuid(os.getuid()).pw_name
    scratch = (Path("/insomnia001/depts/edu/users") / user).resolve()
    home = Path(raw).resolve()
    if home == scratch or scratch not in home.parents:
        raise DownloadError("HF_HOME must be a cache directory below your personal /insomnia001/depts/edu/users/ scratch directory.")
    if home.exists() and not home.is_dir():
        raise DownloadError("HF_HOME must identify a directory.")
    # Keep credentials in the library's cache and heavy downloads in scratch.
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        raise DownloadError("Clear token environment overrides. The target uses a cached CLI sign-in; public drafters use anonymous access.")
    os.environ["HF_HOME"] = str(home)
    os.environ["HF_HUB_CACHE"] = str(home / "hub")
    os.environ["HF_XET_CACHE"] = str(home / "xet")
    os.environ["HF_HUB_VERBOSITY"] = "error"
    os.environ["HF_DEBUG"] = "0"
    return home


def verify_target_file(path: Path, name: str) -> None:
    if name not in TARGET_FILE_BYTES or name not in (SMALL_FILE_BLOBS.keys() | WEIGHT_SHA256.keys()):
        raise DownloadError(f"{name}: the canonical content-hash metadata is unavailable.")
    if not path.is_file() or path.stat().st_size != TARGET_FILE_BYTES[name]:
        raise DownloadError(f"{name}: file size differs from the canonical revision.")
    if name in SMALL_FILE_BLOBS:
        data = path.read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        expected = SMALL_FILE_BLOBS[name]
    else:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        actual, expected = digest.hexdigest(), WEIGHT_SHA256[name]
    if actual != expected:
        raise DownloadError(f"{name}: content hash differs from the canonical revision.")


def verify_snapshot(artifact: Artifact, snapshot: Path, cache: Path) -> dict[str, int]:
    expected = cache / ("models--" + artifact.repo_id.replace("/", "--")) / "snapshots" / artifact.revision
    if snapshot.resolve() != expected.resolve() or snapshot.name != artifact.revision:
        raise DownloadError(f"{artifact.name}: the returned snapshot path differs from the pinned revision.")
    required = set(artifact.required_files)
    if artifact.name == "target" and required != set(TARGET_FILE_BYTES):
        raise DownloadError("target: verify the complete canonical thirteen-file artifact set.")
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text())["weight_map"]
            if (not isinstance(weight_map, dict) or not weight_map
                    or any(not isinstance(key, str) or not key for key in weight_map)):
                raise ValueError("Invalid weight map")
            shards = set(weight_map.values())
            if not shards or any(not isinstance(s, str) or Path(s).name != s for s in shards):
                raise ValueError("Invalid weight shard names")
            if artifact.name == "target":
                expected_shards = {name for name in TARGET_FILES if name.endswith(".safetensors")}
                if shards != expected_shards:
                    raise ValueError("Target index differs from the reviewed four-shard artifact set")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            raise DownloadError(f"{artifact.name}: the weight index needs inspection.") from error
        required.update(shards)
    sizes = {}
    for name in sorted(required):
        path = snapshot / name
        if not path.is_file() or path.stat().st_size == 0:
            raise DownloadError(f"{artifact.name}: required file {name} is missing or empty.")
        sizes[name] = path.stat().st_size
        if artifact.name == "target":
            verify_target_file(path, name)
            if name.endswith(".json"):
                try:
                    content = json.loads(path.read_text())
                    if not isinstance(content, dict) or not content:
                        raise ValueError("Expected a nonempty JSON object")
                except (OSError, ValueError) as error:
                    raise DownloadError(f"target: {name} needs inspection.") from error
    return sizes


def target_acquisition(artifact: Artifact, source: str) -> dict:
    """Describe the canonical source route after every target file is verified."""
    return {
        "source": source,
        "weight_source_repo": artifact.repo_id,
        "weight_source_revision": artifact.revision,
        "weight_sha256": dict(WEIGHT_SHA256),
        "canonical_small_file_git_blobs": dict(SMALL_FILE_BLOBS),
        "weight_identity": "Every weight file matches the canonical Meta LFS SHA256 and byte size.",
        "scope": "Canonical target files obtained directly from Meta; model parameters, tokenizer and generation configuration remain unchanged.",
    }


def read_provenance(destination: Path) -> dict:
    if not destination.exists():
        return {}
    try:
        record = json.loads(destination.read_text())
        if not isinstance(record, dict):
            raise ValueError("Expected an object")
        return record
    except (ValueError, OSError) as error:
        raise DownloadError("Preserving an unreadable provenance record for inspection.") from error


def canonical_target_origin(record: dict, artifact: Artifact, snapshot: Path) -> bool:
    """Recognize an existing fully attributed canonical acquisition record."""
    acquisition = record.get("acquisition")
    if not isinstance(acquisition, dict) or not isinstance(record.get("snapshot_path"), str):
        return False
    return (
        record.get("artifact") == artifact.name
        and record.get("repo_id") == artifact.repo_id
        and record.get("revision") == artifact.revision
        and Path(record["snapshot_path"]).resolve() == snapshot.resolve()
        and record.get("required_file_bytes") == TARGET_FILE_BYTES
        and acquisition.get("weight_source_repo") == artifact.repo_id
        and acquisition.get("weight_source_revision") == artifact.revision
        and acquisition.get("weight_sha256") == WEIGHT_SHA256
        and acquisition.get("canonical_small_file_git_blobs") == SMALL_FILE_BLOBS
        and bool(acquisition.get("source"))
    )


def save_provenance(home: Path, artifact: Artifact, snapshot: Path,
                    sizes: dict[str, int] | None, job_id: str, host: str,
                    hub_version: str | None, *, acquisition: dict | None = None,
                    import_tool: str | None = None) -> Path:
    directory = home / "hpml-provenance" / artifact.name
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{artifact.revision}.json"
    # Keep the complete-record replacement and preservation decision together.
    # This advisory lock also coordinates simultaneous imports and CLI retries.
    with (directory / ".provenance.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if artifact.name == "target":
            verified_sizes = verify_snapshot(artifact, snapshot, home / "hub")
            if sizes is not None and sizes != verified_sizes:
                raise DownloadError("target: supplied file sizes differ from the verified snapshot.")
            sizes = verified_sizes
        elif sizes is None:
            raise DownloadError(f"{artifact.name}: verify required files before saving provenance.")
        existing = read_provenance(destination)
        if existing and (existing.get("repo_id") != artifact.repo_id
                         or existing.get("revision") != artifact.revision
                         or not isinstance(existing.get("snapshot_path"), str)
                         or Path(existing["snapshot_path"]).resolve() != snapshot.resolve()):
            raise DownloadError("Preserving provenance that identifies a different snapshot.")
        return _write_provenance(artifact, snapshot, sizes, job_id, host, hub_version,
                                 destination, existing, acquisition, import_tool)


def _write_provenance(artifact: Artifact, snapshot: Path,
                      sizes: dict[str, int], job_id: str, host: str,
                      hub_version: str | None, destination: Path, existing: dict,
                      acquisition: dict | None, import_tool: str | None) -> Path:
    record = {
        **existing,
        "artifact": artifact.name,
        "repo_id": artifact.repo_id,
        "revision": artifact.revision,
        "source_url": f"https://huggingface.co/{artifact.repo_id}/tree/{artifact.revision}",
        "snapshot_path": str(snapshot.resolve()),
        "snapshot_scope": ("requested Transformers inference artifact set, model card, license and use policy at pinned revision"
                           if artifact.download_files else "complete repository at pinned revision"),
        "requested_files": list(artifact.download_files) if artifact.download_files else None,
        "required_file_bytes": sizes,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": job_id,
        "compute_host": host,
        "verification_tool": import_tool or "baselines/download_checkpoints.py",
        "verification": ("snapshot commit path, all thirteen file sizes against pinned public repository metadata, "
                         "all four weight SHA256 hashes and all nine canonical small-file Git blob hashes, "
                         "valid JSON objects and exact four-shard index coverage; model loading recorded separately"
                         if artifact.name == "target" else
                         "snapshot commit path and required-file presence/size; model loading pending"),
    }
    if hub_version is not None:
        record["huggingface_hub_version"] = hub_version
    if artifact.name == "target":
        if canonical_target_origin(existing, artifact, snapshot):
            # Preserve the original route and any richer acquisition evidence.
            record["acquisition"] = existing["acquisition"]
        else:
            record["acquisition"] = acquisition
            if not canonical_target_origin(record, artifact, snapshot):
                raise DownloadError("target: a verified canonical Meta acquisition route is required.")
            record.pop("import_tool", None)
            if hub_version is None:
                record.pop("huggingface_hub_version", None)
            if import_tool is not None:
                record["import_tool"] = import_tool
    if artifact.download_files:
        record["retained_extra_cached_files"] = sorted(
            str(path.relative_to(snapshot)) for path in snapshot.rglob("*")
            if path.is_file() and str(path.relative_to(snapshot)) not in artifact.download_files
        )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected = [a for a in ARTIFACTS if args.only is None or a.name in args.only]
    if args.action == "list":
        for artifact in selected:
            print(f"{artifact.name}\n  {artifact.repo_id}\n  {artifact.revision}\n  {artifact.role}")
            if artifact.download_files:
                print(f"  Download scope: {len(artifact.download_files)} Transformers and documentation files.")
            else:
                print("  Download scope: complete repository snapshot.")
        print("Use --download --only NAME [NAME ...] inside a Slurm compute allocation.")
        return 0
    try:
        job_id, host = require_compute_allocation()
        home = require_scratch_cache()
        # Import after cache configuration; the default list action needs no Hub package.
        try:
            import huggingface_hub
            from huggingface_hub import get_token, snapshot_download
        except ImportError as error:
            raise DownloadError("Activate a prepared environment containing huggingface_hub.") from error
        if any(artifact.name == "target" for artifact in selected) and not get_token():
            raise DownloadError("The target requires a cached Hugging Face CLI sign-in using this HF_HOME and approved Meta access.")
        os.umask(0o077)
        cache = home / "hub"
        for artifact in selected:
            print(f"Acquiring {artifact.name}: {artifact.repo_id}@{artifact.revision}", flush=True)
            target_refresh = False
            acquisition = None
            if artifact.name == "target":
                expected = cache / ("models--" + artifact.repo_id.replace("/", "--")) / "snapshots" / artifact.revision
                previous = read_provenance(home / "hpml-provenance" / artifact.name / f"{artifact.revision}.json")
                target_refresh = not canonical_target_origin(previous, artifact, expected)
                if target_refresh:
                    print("Refreshing all thirteen files directly from the canonical Meta repository to establish acquisition provenance.", flush=True)
                acquisition = target_acquisition(
                    artifact,
                    "All thirteen files retrieved directly from the canonical Meta Hugging Face repository "
                    "through approved access using snapshot_download at https://huggingface.co.",
                )
            try:
                path = snapshot_download(
                    repo_id=artifact.repo_id, repo_type="model", revision=artifact.revision,
                    cache_dir=cache, token=(artifact.name == "target"),
                    endpoint="https://huggingface.co", max_workers=4,
                    allow_patterns=list(artifact.download_files) if artifact.download_files else None,
                    force_download=target_refresh,
                )
            except Exception as error:
                # Hub errors can contain request details; keep credentials out of output.
                raise DownloadError(
                    f"{artifact.name}: snapshot acquisition failed ({type(error).__name__}). "
                    "Check approved Meta access and cached sign-in for the target, "
                    "or anonymous access for a public drafter. Check connectivity and scratch storage."
                ) from None
            snapshot = Path(path)
            # Target verification runs inside the complete-record publication path.
            sizes = None if artifact.name == "target" else verify_snapshot(artifact, snapshot, cache)
            manifest = save_provenance(home, artifact, snapshot, sizes, job_id, host,
                                       huggingface_hub.__version__, acquisition=acquisition)
            print(f"Verified revision: {artifact.revision}\nSnapshot: {snapshot.resolve()}\nProvenance: {manifest}", flush=True)
    except DownloadError as error:
        print(f"Checkpoint preparation stopped: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"Checkpoint preparation stopped: local storage or process failure ({type(error).__name__}).", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Checkpoint preparation interrupted. The Hub cache retains resumable files.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
