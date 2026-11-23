#!/usr/bin/env python3
"""Refresh local trial records through an existing SSH connection.

Run this script in a visible terminal on the Mac. Keep the browser viewer in
a separate visible terminal. Each refresh starts one short remote rsync
process and waits for that transfer to finish.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEASUREMENTS_ROOT = PROJECT_ROOT / "measurements"
REMOTE_PATTERN = re.compile(
    r"aa5886@insomnia\.rcs\.columbia\.edu:"
    r"/insomnia001/depts/edu/users/aa5886/hpml-baselines/results/"
    r"trial-first30-([0-9]{8})/?"
)
TRANSFER_TIMEOUT_SECONDS = 60
HEARTBEAT_SECONDS = 30


class SyncError(Exception):
    """Describe a condition that requires the foreground sync to stop."""


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def announce(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def remote_source(value: str) -> str:
    match = REMOTE_PATTERN.fullmatch(value)
    if match is None:
        raise argparse.ArgumentTypeError(
            "Use the Insomnia source "
            "aa5886@insomnia.rcs.columbia.edu:"
            "/insomnia001/depts/edu/users/aa5886/hpml-baselines/results/"
            "trial-first30-YYYYMMDD/."
        )
    try:
        datetime.strptime(match.group(1), "%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "The trial folder must contain a valid date in YYYYMMDD format."
        ) from exc
    return value.rstrip("/") + "/"


def local_destination(value: str) -> Path:
    root = MEASUREMENTS_ROOT.resolve()
    destination = Path(value).expanduser().resolve()
    if destination == root or not destination.is_relative_to(root):
        raise argparse.ArgumentTypeError(
            f"Choose a trial folder inside {root}."
        )
    if destination.exists() and not destination.is_dir():
        raise argparse.ArgumentTypeError("The destination must be a directory.")
    return destination


def control_socket(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if any(character in str(path) for character in ("\n", "\r", "\0", "%")):
        raise argparse.ArgumentTypeError(
            "Use an SSH socket path without line breaks or percent signs."
        )
    try:
        is_socket = stat.S_ISSOCK(path.stat().st_mode)
    except OSError as exc:
        raise argparse.ArgumentTypeError(
            "Open the authenticated SSH connection before starting the sync. "
            f"The control socket is unavailable: {path}."
        ) from exc
    if not is_socket:
        raise argparse.ArgumentTypeError(
            f"The control path must identify an SSH socket: {path}."
        )
    return path


def refresh_interval(value: str) -> float:
    try:
        interval = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "The refresh interval must be a positive number of seconds."
        ) from exc
    if not math.isfinite(interval) or interval <= 0:
        raise argparse.ArgumentTypeError(
            "The refresh interval must be a positive number of seconds."
        )
    return interval


def transfer_command(
    executable: str, remote: str, destination: Path, socket: Path
) -> list[str]:
    ssh = shlex.join(
        [
            "ssh",
            "-o", f"ControlPath={socket}",
            "-o", "ControlMaster=no",
            "-o", "BatchMode=yes",
            "-o", "PreferredAuthentications=none",
            "-o", "PasswordAuthentication=no",
            "-o", "KbdInteractiveAuthentication=no",
            "-o", "PubkeyAuthentication=no",
            "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=10",
            "-o", "ServerAliveCountMax=2",
        ]
    )
    return [
        executable,
        "-rlt",
        "--no-perms",
        "--itemize-changes",
        "--include=*/",
        "--include=*.json",
        "--include=*.jsonl",
        "--exclude=*",
        "-e", ssh,
        remote,
        str(destination) + "/",
    ]


def stop_transfer(process: subprocess.Popen[str]) -> None:
    """Stop the transfer and its SSH child, then collect their remaining output."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()


def transfer(command: list[str]) -> tuple[str, str]:
    # A separate process group lets the waiting foreground controller stop both
    # rsync and its short SSH child after an interruption or a transfer timeout.
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=TRANSFER_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        stop_transfer(process)
        raise SyncError(
            "The transfer exceeded 60 seconds. The transfer and its SSH "
            "process have stopped. Check the connection before restarting."
        ) from exc
    except BaseException:
        stop_transfer(process)
        raise
    if process.returncode:
        details = stderr.strip() or stdout.strip()
        message = (
            f"The transfer stopped with exit code {process.returncode}. "
            "Check the authenticated SSH connection and the source folder "
            "before restarting. The existing local records remain available."
        )
        if details:
            message += "\n" + details
        raise SyncError(message)
    return stdout.strip(), stderr.strip()


def handle_stop(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh local trial JSON records through an existing authenticated "
            "SSH connection. Keep this terminal visible while the trial runs."
        )
    )
    parser.add_argument("--remote", required=True, type=remote_source)
    parser.add_argument(
        "--destination", required=True, type=local_destination,
        help=f"Use a trial folder inside {MEASUREMENTS_ROOT}.",
    )
    parser.add_argument("--control-path", required=True, type=control_socket)
    parser.add_argument(
        "--interval", type=refresh_interval, default=3.0,
        help="Set the pause between completed refreshes in seconds. The default is 3.",
    )
    args = parser.parse_args()
    rsync = shutil.which("rsync")
    if rsync is None or shutil.which("ssh") is None:
        parser.error("Install rsync and SSH before starting the sync.")
    signal.signal(signal.SIGTERM, handle_stop)
    last_report = float("-inf")
    last_warning = ""
    try:
        args.destination.mkdir(parents=True, exist_ok=True)
        command = transfer_command(
            rsync, args.remote, args.destination, args.control_path
        )
        announce(f"The sync reads records from {args.remote}.")
        announce(f"The browser viewer reads the local records in {args.destination}.")
        announce(
            f"Each refresh waits {args.interval:g} seconds before the next transfer. "
            "Press Control-C to stop the sync."
        )
        while True:
            # Check again before each transfer because the user may close SSH.
            try:
                socket_available = stat.S_ISSOCK(args.control_path.stat().st_mode)
            except OSError:
                socket_available = False
            if not socket_available:
                raise SyncError(
                    "The authenticated SSH socket has closed. The sync has "
                    "stopped. Reopen the SSH connection before restarting."
                )
            changes, warning = transfer(command)
            now = time.monotonic()
            if changes:
                announce("The refresh updated these paths:")
                print(changes, flush=True)
            if warning and warning != last_warning:
                announce("The transfer reported this message:")
                print(warning, flush=True)
            last_warning = warning
            if changes or now - last_report >= HEARTBEAT_SECONDS:
                announce("The refresh completed.")
                last_report = now
            time.sleep(args.interval)
    except KeyboardInterrupt:
        announce("The sync has stopped. The saved local records remain available.")
        return 0
    except (OSError, SyncError) as exc:
        announce(f"The sync has stopped. {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
