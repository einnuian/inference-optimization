#!/usr/bin/env python3
"""Fetch the reviewed pretrained-baseline source revisions.

The default action lists the sources. --fetch downloads Git source only.
Dependency installation, model downloads, submodules and Git LFS downloads
remain separate actions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    revision: str


SOURCES = (
    Source("eagle", "https://github.com/SafeAILab/EAGLE.git",
           "cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b"),
    Source("dflash", "https://github.com/z-lab/dflash.git",
           "07ebd93db9f472af339b644bb70221ad8428328a"),
    Source("hspec", "https://github.com/weifanjiang/H-Spec.git",
           "de911876905440bfd8edfa78b482050f7ac0279c"),
    Source("pard", "https://github.com/AMD-AGI/PARD.git",
           "6f279bf3f1680e0b5d71c562ca5b91bdeef4c038"),
    Source("sglang", "https://github.com/sgl-project/sglang.git",
           "94602c9c2b7cbdb8efd5c52802dac6a1c180089e"),
    Source("vllm", "https://github.com/vllm-project/vllm.git",
           "752a3a504485790a2e8491cacbb35c137339ad34"),
    Source("speculators", "https://github.com/vllm-project/speculators.git",
           "3251ed020cfc954717c1d075d83dca8721713f0c"),
)
DEFAULT_DESTINATION = Path(__file__).resolve().parents[1] / "external"


class SourceError(RuntimeError):
    """A source could not be acquired or verified safely."""


def git_environment() -> dict[str, str]:
    # Keep public-source acquisition independent of user Git configuration.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GIT_")}
    env.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_LFS_SKIP_SMUDGE": "1",
    })
    return env


def git(directory: Path | None, *args: str) -> str:
    command = [
        "git",
        "-c", f"core.hooksPath={os.devnull}",
        "-c", "core.fsmonitor=false",
        "-c", "submodule.recurse=false",
        "-c", "protocol.file.allow=never",
        "-c", "credential.helper=",
    ]
    if directory is not None:
        command.extend(["-C", str(directory)])
    command.extend(args)
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=git_environment(),
        )
    except OSError as error:
        raise SourceError(f"Cannot run Git: {error}") from error
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SourceError(
            f"Git {args[0]} failed in {directory or Path.cwd()}: {detail}"
        )
    return result.stdout.strip()


def canonical_url(url: str) -> str:
    return url.rstrip("/").removesuffix(".git")


def check_destination_path(destination: Path) -> None:
    for path in (destination, *destination.parents):
        if path.is_symlink():
            raise SourceError(f"Choose a destination without symlink components: {path}")
    if destination.exists() and not destination.is_dir():
        raise SourceError(f"The destination is not a directory: {destination}")


def verify(source: Source, path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise SourceError(f"Expected a regular checkout directory: {path}")
    git_dir = path / ".git"
    if git_dir.is_symlink() or not git_dir.is_dir():
        raise SourceError(f"Expected an independent Git checkout at {path}")
    top = Path(git(path, "rev-parse", "--show-toplevel")).resolve()
    if top != path.resolve():
        raise SourceError(f"Git identifies a different checkout root for {path}: {top}")
    remote = git(path, "config", "--get", "remote.origin.url")
    if canonical_url(remote) != canonical_url(source.url):
        raise SourceError(
            f"Preserving {path}: origin differs from the audited repository URL."
        )
    actual = git(path, "rev-parse", "HEAD")
    if actual != source.revision:
        raise SourceError(
            f"Preserving {path}: HEAD is {actual}; expected {source.revision}."
        )
    changes = git(path, "status", "--porcelain=v1", "--untracked-files=all")
    if changes:
        raise SourceError(f"Preserving {path}: the checkout has local changes.\n{changes}")


def fetch(source: Source, path: Path) -> None:
    # Exclusive creation preserves every pre-existing destination.
    try:
        path.mkdir()
    except FileExistsError as error:
        raise SourceError(f"Preserving a destination created during this run: {path}") from error
    try:
        git(None, "init", "--quiet", "--template=", str(path))
        # Persist these settings so subsequent manual checks retain the safeguards.
        git(path, "config", "core.hooksPath", os.devnull)
        git(path, "config", "core.fsmonitor", "false")
        git(path, "config", "submodule.recurse", "false")
        git(path, "remote", "add", "origin", source.url)
        git(path, "fetch", "--no-tags", "--depth=1", "origin", source.revision)
        git(path, "checkout", "--quiet", "--detach", source.revision)
        verify(source, path)
    except SourceError as error:
        raise SourceError(
            f"{error}\nThe incomplete checkout remains at {path}. "
            "Inspect that directory before removing it or retrying."
        ) from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--list", dest="action", action="store_const", const="list",
                         help="List audited source URLs and revisions (default).")
    actions.add_argument("--fetch", dest="action", action="store_const", const="fetch",
                         help="Fetch missing source checkouts and verify existing ones.")
    actions.add_argument("--verify", dest="action", action="store_const", const="verify",
                         help="Verify existing source checkouts without fetching.")
    parser.set_defaults(action="list")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION,
                        help="Parent directory for checkouts (default: repository external/).")
    parser.add_argument("--only", nargs="+", choices=[source.name for source in SOURCES],
                        help="Limit the action to the named sources.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    destination = Path(os.path.abspath(args.destination.expanduser()))
    selected = [source for source in SOURCES
                if args.only is None or source.name in args.only]
    if args.action == "list":
        print(f"Destination: {destination}")
        for source in selected:
            print(f"{source.name}\n  {source.url}\n  {source.revision}")
        print("Use --fetch to acquire source or --verify to check existing checkouts.")
        return 0

    try:
        check_destination_path(destination)
        missing: list[Source] = []
        # Validate every existing checkout before starting any network or write action.
        for source in selected:
            path = destination / source.name
            if path.exists() or path.is_symlink():
                verify(source, path)
                print(f"Verified {source.name}: {source.revision}", flush=True)
            elif args.action == "verify":
                raise SourceError(f"Missing source checkout: {path}")
            else:
                missing.append(source)
        if args.action == "fetch":
            destination.mkdir(parents=True, exist_ok=True)
            for source in missing:
                print(f"Fetching {source.name}: {source.revision}", flush=True)
                fetch(source, destination / source.name)
                print(f"Verified {source.name}: {source.revision}", flush=True)
    except (SourceError, OSError) as error:
        print(f"Source acquisition stopped: {error}", file=sys.stderr)
        return 1
    print(f"All {len(selected)} selected source checkouts match their reviewed revisions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
