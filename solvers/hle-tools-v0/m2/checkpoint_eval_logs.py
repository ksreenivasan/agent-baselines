#!/usr/bin/env python3
"""Atomically checkpoint valid Inspect .eval logs from node-local storage."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import tempfile
import threading
import zipfile
from pathlib import Path


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
    except (OSError, zipfile.BadZipFile):
        return False


def checkpoint(source_dir: Path, durable_dir: Path) -> int:
    """Publish complete snapshots; never replace a good checkpoint with a bad one."""
    durable_dir.mkdir(parents=True, exist_ok=True)
    published = 0
    for source in sorted(source_dir.glob("*.eval")):
        snapshot_fd, snapshot_name = tempfile.mkstemp(
            prefix=f".{source.name}.snapshot-", dir=source_dir
        )
        os.close(snapshot_fd)
        snapshot = Path(snapshot_name)
        destination = durable_dir / source.name
        durable_tmp = durable_dir / f".{source.name}.{os.getpid()}.tmp"
        try:
            _copy_and_fsync(source, snapshot)
            if not _valid_eval(snapshot):
                print(f"checkpoint skipped incomplete source: {source}", flush=True)
                continue
            _copy_and_fsync(snapshot, durable_tmp)
            if not _valid_eval(durable_tmp):
                print(f"checkpoint skipped invalid durable copy: {source}", flush=True)
                continue
            os.replace(durable_tmp, destination)
            directory_fd = os.open(durable_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            published += 1
            print(
                f"checkpoint published: {source.name} ({destination.stat().st_size} bytes)",
                flush=True,
            )
        except OSError as error:
            print(f"checkpoint failed without replacing prior copy: {error}", flush=True)
        finally:
            snapshot.unlink(missing_ok=True)
            durable_tmp.unlink(missing_ok=True)
    return published


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--durable-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.source_dir.mkdir(parents=True, exist_ok=True)
    os.umask(0o077)

    if args.once:
        checkpoint(args.source_dir, args.durable_dir)
        return 0

    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    while not stop.is_set():
        checkpoint(args.source_dir, args.durable_dir)
        stop.wait(args.interval)
    checkpoint(args.source_dir, args.durable_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
