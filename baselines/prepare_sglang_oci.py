#!/usr/bin/env python3
"""Retrieve the pinned public SGLang OCI image with resumable verified layers.

Run through setup_sglang.sh inside a Slurm compute allocation. This helper uses
the cluster's aria2c executable and Python standard library. All data remains in
HPML_ROOT/cache/sglang-oci. It retrieves container packages and source only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request


DIGEST = "sha256:1173f6a57ac63209b1bfb054e6a7366896fd3df5539c809bbd6d332694e91d15"
REVISION = "0bcd822377da7b5718e674eaf9c870d349424dd1"
REGISTRY = "https://registry-1.docker.io/v2/lmsysorg/sglang"
REFERENCE = "sglang-cu129"


def get_bytes(url: str, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def sha256_file(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_verified_blob(blobs: Path, raw: bytes, digest: str) -> None:
    checksum = digest.removeprefix("sha256:")
    if hashlib.sha256(raw).hexdigest() != checksum:
        raise RuntimeError("Registry metadata checksum differs from its pinned identity.")
    path = blobs / checksum
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError("Preserving a conflicting metadata blob for inspection.")
    path.write_bytes(raw)


def main() -> None:
    os.umask(0o077)
    root = Path(os.environ["HPML_ROOT"])
    layout = root / "cache/sglang-oci"
    blobs = layout / "blobs/sha256"
    if any(path.is_symlink() for path in (blobs, *blobs.parents)):
        raise RuntimeError("The OCI cache requires ordinary directory components.")
    blobs.mkdir(parents=True, exist_ok=True)
    if any(path.stat().st_uid != os.getuid() for path in (layout, blobs.parent, blobs)):
        raise RuntimeError("The OCI cache directories must belong to your account.")
    if shutil.which("aria2c") is None:
        raise RuntimeError("The verified cluster aria2c executable is required.")
    token = json.loads(get_bytes(
        "https://auth.docker.io/token?service=registry.docker.io&scope=repository:lmsysorg/sglang:pull"
    ))["token"]
    headers = {"Authorization": "Bearer " + token,
               "Accept": "application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json"}
    raw_manifest = get_bytes(REGISTRY + "/manifests/" + DIGEST, headers)
    write_verified_blob(blobs, raw_manifest, DIGEST)
    manifest = json.loads(raw_manifest)
    raw_config = get_bytes(REGISTRY + "/blobs/" + manifest["config"]["digest"], headers)
    write_verified_blob(blobs, raw_config, manifest["config"]["digest"])
    config = json.loads(raw_config)
    labels = config["config"]["Labels"]
    if (config["architecture"] != "amd64" or config["os"] != "linux"
            or labels.get("org.opencontainers.image.revision") != REVISION
            or labels.get("ai.sglang.image.tag") != "lmsysorg/sglang:v0.5.19-cu129"):
        raise RuntimeError("The registry image configuration differs from the reviewed release.")
    evidence = root / "runs/sglang-setup"
    evidence.mkdir(parents=True, exist_ok=True)
    for name, raw in (("upstream-manifest.json", raw_manifest), ("upstream-config.json", raw_config)):
        destination = evidence / name
        if destination.exists() and destination.read_bytes() != raw:
            raise RuntimeError("Preserving conflicting upstream provenance for inspection.")
        destination.write_bytes(raw)
    input_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", prefix="aria2-input-", suffix=".txt",
                                         dir=layout, delete=False) as handle:
            input_path = Path(handle.name)
            seen = set()
            for layer in manifest["layers"]:
                checksum = layer["digest"].removeprefix("sha256:")
                if checksum in seen:
                    continue
                seen.add(checksum)
                path = blobs / checksum
                if not path.exists():
                    # Reuse completed, verified blobs from the initial Apptainer pull.
                    for previous in (root / "cache/apptainer-sglang").rglob(checksum):
                        if (previous.is_file() and previous.stat().st_size == layer["size"]
                                and sha256_file(previous) == checksum):
                            os.link(previous, path)
                            break
                if path.exists() and path.stat().st_size == layer["size"] and sha256_file(path) == checksum:
                    print(f"Verified cached layer {checksum[:12]} ({layer['size']} bytes).", flush=True)
                    continue
                request = urllib.request.Request(REGISTRY + "/blobs/" + layer["digest"],
                                                 headers={**headers, "Range": "bytes=0-0"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    download_url = response.url
                    response.read(1)
                if not download_url.startswith("https://"):
                    raise RuntimeError("The registry supplied a non-HTTPS layer URL.")
                handle.write(download_url + "\n")
                handle.write(f"  dir={blobs}\n  out={checksum}\n  checksum=sha-256={checksum}\n")
        subprocess.run([
            "aria2c", "--input-file=" + str(input_path), "--continue=true",
            "--check-integrity=true", "--allow-overwrite=false", "--auto-file-renaming=false",
            "--max-concurrent-downloads=4", "--split=4", "--max-connection-per-server=4",
            "--min-split-size=16M", "--file-allocation=none", "--max-tries=5",
            "--retry-wait=5", "--timeout=90", "--connect-timeout=20", "--summary-interval=60",
            "--console-log-level=warn", "--download-result=hide", "--enable-color=false",
        ], check=True)
    finally:
        if input_path is not None:
            input_path.unlink(missing_ok=True)
    for layer in manifest["layers"]:
        checksum = layer["digest"].removeprefix("sha256:")
        path = blobs / checksum
        if not path.is_file() or path.stat().st_size != layer["size"] or sha256_file(path) != checksum:
            raise RuntimeError(f"Layer {checksum[:12]} requires another verified retrieval.")
    (layout / "oci-layout").write_text(json.dumps({"imageLayoutVersion": "1.0.0"}) + "\n")
    index = {"schemaVersion": 2, "manifests": [{
        "mediaType": manifest["mediaType"], "digest": DIGEST, "size": len(raw_manifest),
        "platform": {"architecture": "amd64", "os": "linux"},
        "annotations": {"org.opencontainers.image.ref.name": REFERENCE},
    }]}
    (layout / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Verified {len(manifest['layers'])} layers for {DIGEST}. OCI layout: {layout}", flush=True)


if __name__ == "__main__":
    main()
