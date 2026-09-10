#!/usr/bin/env python3
"""Atomically checkpoint valid Inspect .eval logs from node-local storage."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import tempfile
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CheckpointStatus:
    """Latest pass and cumulative publisher health, including failures."""

    expected: int = 0
    published: int = 0
    incomplete: int = 0
    errors: list[str] = field(default_factory=list)
    checked_at: float | None = None
    total_published: int = 0
    last_published_at: float | None = None
    last_error: str | None = None

    @property
    def complete(self) -> bool:
        return self.expected > 0 and self.published == self.expected and not self.errors

    def fail(self, message: str) -> None:
        self.errors.append(message)
        self.last_error = message
        print(f"checkpoint failed: {message}", flush=True)


def _copy_and_fsync(source: Path, destination: Path) -> None:
    with source.open("rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())


def _valid_eval(path: Path) -> bool:
    if path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            return bool(names) and archive.testzip() is None
    except (zipfile.BadZipFile, EOFError):
        # A writer may still be appending the ZIP directory. I/O errors, however,
        # must reach the caller so a broken filesystem is not called incomplete.
        return False


def _cleanup(path: Path, status: CheckpointStatus) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        status.fail(f"cleanup {path}: {error}")


def checkpoint(
    source_dir: Path,
    durable_dir: Path,
    status: CheckpointStatus | None = None,
) -> int:
    """Publish complete snapshots; never replace a good copy with invalid bytes."""
    if status is None:
        status = CheckpointStatus()
    status.expected = status.published = status.incomplete = 0
    status.errors = []
    try:
        durable_dir.mkdir(parents=True, exist_ok=True)
        # iterdir surfaces unreadable/missing source directories instead of
        # silently treating a glob traversal failure as an empty directory.
        sources = sorted(p for p in source_dir.iterdir() if p.suffix == ".eval")
        status.expected = len(sources)
        for source in sources:
            snapshot = None
            durable_tmp = durable_dir / f".{source.name}.{os.getpid()}.tmp"
            try:
                snapshot_fd, snapshot_name = tempfile.mkstemp(
                    prefix=f".{source.name}.snapshot-", dir=source_dir
                )
                snapshot = Path(snapshot_name)
                os.close(snapshot_fd)
                _copy_and_fsync(source, snapshot)
                if not _valid_eval(snapshot):
                    status.incomplete += 1
                    print(f"checkpoint skipped incomplete source: {source}", flush=True)
                    continue
                _copy_and_fsync(snapshot, durable_tmp)
                if not _valid_eval(durable_tmp):
                    status.fail(f"invalid durable copy: {source}")
                    continue
                os.replace(durable_tmp, durable_dir / source.name)
                directory_fd = os.open(durable_dir, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                status.published += 1
                status.total_published += 1
                status.last_published_at = time.time()
                print(f"checkpoint published: {source.name}", flush=True)
            except (OSError, RuntimeError) as error:
                status.fail(f"{source}: {error}")
            finally:
                if snapshot is not None:
                    _cleanup(snapshot, status)
                _cleanup(durable_tmp, status)
    except OSError as error:
        status.fail(str(error))
    status.checked_at = time.time()
    return status.published


def _write_status(path: Path, status: CheckpointStatus) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as output:
            json.dump({**asdict(status), "pid": os.getpid()}, output)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        _cleanup(temporary, status)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--durable-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--status-file", type=Path, help="Prefer a node-local path")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval must be positive")
    os.umask(0o077)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    status = CheckpointStatus()

    def publish() -> None:
        checkpoint(args.source_dir, args.durable_dir, status)
        print(f"checkpoint status: {json.dumps(asdict(status))}", flush=True)
        if args.status_file:
            _write_status(args.status_file, status)

    if args.once:
        publish()
        return 0 if status.complete else 1

    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    while not stop.is_set():
        publish()
        stop.wait(args.interval)
    publish()
    return 0 if status.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
