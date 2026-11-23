#!/usr/bin/env python3
"""Import files downloaded directly from the approved Meta repository.

Obtain all thirteen files through your approved Meta access and transfer them
to a private HF_HOME import directory. Inside a Slurm allocation, run:

    python baselines/import_meta_target.py --meta-files PATH

Use --only-weights NAME to verify and import one newly transferred weight file
at a time. The nine small files must always be present. Every file must match
Meta's exact content hash and size. For the first complete acquisition record,
retain all four downloaded weight files in the source directory. An existing
verified Meta acquisition record establishes the origin of previously imported
shards. Existing files are preserved. No model executes, no alternate
repository is accessed, and no credential is read or transferred.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import tempfile

from download_checkpoints import (
    ARTIFACTS, DownloadError, SMALL_FILE_BLOBS, WEIGHT_SHA256, canonical_target_origin,
    read_provenance, require_compute_allocation, require_scratch_cache, save_provenance,
    target_acquisition, verify_target_file,
)

TARGET = next(artifact for artifact in ARTIFACTS if artifact.name == "target")


def ensure_directory(path: Path) -> None:
    if path.is_symlink():
        raise DownloadError(f"Preserving an existing directory symlink: {path}")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.stat().st_uid != os.getuid():
        raise DownloadError(f"Expected a cache directory owned by your account: {path}")


def publish_file(source: Path, name: str, blobs: Path, snapshot: Path) -> None:
    oid = SMALL_FILE_BLOBS.get(name) or WEIGHT_SHA256[name]
    blob = blobs / oid
    if blob.is_symlink():
        raise DownloadError(f"Preserving an unexpected blob symlink for {name}.")
    if blob.exists():
        verify_target_file(blob, name)
    elif name in WEIGHT_SHA256:
        # The import directory and cache share the private scratch filesystem.
        # A hard link retains one physical copy during the import.
        os.link(source.resolve(), blob)
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=blobs, delete=False) as handle:
                temporary = Path(handle.name)
                with source.open("rb") as reader:
                    shutil.copyfileobj(reader, handle)
            os.link(temporary, blob)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    destination = snapshot / name
    if destination.exists() or destination.is_symlink():
        verify_target_file(destination, name)
    else:
        destination.symlink_to(Path("../../blobs") / oid)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-files", required=True, type=Path)
    parser.add_argument("--only-weights", nargs="+", choices=list(WEIGHT_SHA256))
    args = parser.parse_args()
    try:
        job_id, host = require_compute_allocation()
        home = require_scratch_cache()
        canonical = args.meta_files.resolve()
        if home not in canonical.parents or not canonical.is_dir():
            raise DownloadError("Canonical files must be in a directory under the private HF_HOME cache.")
        for name in SMALL_FILE_BLOBS:
            verify_target_file(canonical / name, name)
        print("All nine canonical configuration, tokenizer and documentation hashes pass.", flush=True)
        os.umask(0o077)
        cache = home / "hub"
        selected_weights = args.only_weights or list(WEIGHT_SHA256)
        for name in selected_weights:
            verify_target_file(canonical / name, name)
            print(f"Canonical SHA256 verified: {name}", flush=True)
        target_cache = cache / "models--meta-llama--Llama-3.1-8B-Instruct"
        blobs = target_cache / "blobs"
        snapshot = target_cache / "snapshots" / TARGET.revision
        for directory in (target_cache, blobs, target_cache / "snapshots", snapshot):
            ensure_directory(directory)
        for name in SMALL_FILE_BLOBS:
            publish_file(canonical / name, name, blobs, snapshot)
        for name in selected_weights:
            publish_file(canonical / name, name, blobs, snapshot)
        if not all((snapshot / name).is_file() for name in WEIGHT_SHA256):
            print("Selected Meta files verified and imported. Remaining weight files are required before model loading.", flush=True)
            return 0
        previous = read_provenance(home / "hpml-provenance/target" / f"{TARGET.revision}.json")
        if not canonical_target_origin(previous, TARGET, snapshot):
            missing_sources = [name for name in WEIGHT_SHA256 if not (canonical / name).is_file()]
            if missing_sources:
                raise DownloadError(
                    "Selected files are imported. Establishing the first complete Meta acquisition record "
                    "requires all four directly downloaded weight files in --meta-files. "
                    "Retain the imported files and supply the remaining source files: " + ", ".join(missing_sources)
                )
            for name in WEIGHT_SHA256:
                if name not in selected_weights:
                    verify_target_file(canonical / name, name)
        provenance = save_provenance(
            home, TARGET, snapshot, None, job_id, host, None,
            acquisition=target_acquisition(
                TARGET,
                "All thirteen files downloaded directly through the user's approved Meta Hugging Face "
                "browser session and transferred to private Insomnia storage.",
            ),
            import_tool="baselines/import_meta_target.py",
        )
        print(f"Canonical target staged: {snapshot}\nProvenance: {provenance}", flush=True)
    except (DownloadError, OSError) as error:
        print(f"Target staging stopped: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
