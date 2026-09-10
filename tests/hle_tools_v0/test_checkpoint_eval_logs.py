import errno
import importlib.util
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "solvers/hle-tools-v0/m2/checkpoint_eval_logs.py"
)
spec = importlib.util.spec_from_file_location("checkpoint_eval_logs", SCRIPT)
checkpoint_logs = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checkpoint_logs
spec.loader.exec_module(checkpoint_logs)


def write_eval(path, content):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("header.json", content)


@pytest.fixture
def directories(tmp_path):
    source = tmp_path / "source"
    durable = tmp_path / "durable"
    source.mkdir()
    durable.mkdir()
    return source, durable


def test_failed_write_preserves_prior_checkpoint_and_later_recovers(
    directories, monkeypatch
):
    source, durable = directories
    write_eval(source / "run.eval", "old")
    status = checkpoint_logs.CheckpointStatus()
    assert checkpoint_logs.checkpoint(source, durable, status) == 1
    previous = (durable / "run.eval").read_bytes()
    previous_time = status.last_published_at
    write_eval(source / "run.eval", "new")
    original_copy = checkpoint_logs._copy_and_fsync

    def fail_durable_copy(src, dst):
        if dst.parent == durable:
            dst.write_bytes(b"partial")
            raise OSError(errno.ENOSPC, "injected full filesystem")
        return original_copy(src, dst)

    monkeypatch.setattr(checkpoint_logs, "_copy_and_fsync", fail_durable_copy)
    assert checkpoint_logs.checkpoint(source, durable, status) == 0
    assert (durable / "run.eval").read_bytes() == previous
    assert status.errors and status.incomplete == 0
    assert status.total_published == 1
    assert status.last_published_at == previous_time
    assert not list(durable.glob("*.tmp"))
    monkeypatch.setattr(checkpoint_logs, "_copy_and_fsync", original_copy)
    assert checkpoint_logs.checkpoint(source, durable, status) == 1
    assert status.complete and status.total_published == 2
    assert status.last_error is not None  # Preserve the last historical failure.
    assert (durable / "run.eval").read_bytes() == (source / "run.eval").read_bytes()


def test_incomplete_running_archive_is_distinct_from_io_failure(directories):
    source, durable = directories
    write_eval(source / "run.eval", "old")
    status = checkpoint_logs.CheckpointStatus()
    checkpoint_logs.checkpoint(source, durable, status)
    previous = (durable / "run.eval").read_bytes()
    (source / "run.eval").write_bytes(b"PK unfinished")
    assert checkpoint_logs.checkpoint(source, durable, status) == 0
    assert status.incomplete == 1 and status.errors == []
    assert (durable / "run.eval").read_bytes() == previous
    write_eval(source / "run.eval", "finished")
    assert checkpoint_logs.checkpoint(source, durable, status) == 1
    assert status.complete


def test_cleanup_failure_does_not_kill_publisher(directories, monkeypatch):
    source, durable = directories
    write_eval(source / "run.eval", "good")
    original_unlink = Path.unlink

    def fail_snapshot_cleanup(path, *args, **kwargs):
        if ".snapshot-" in path.name:
            raise PermissionError("injected cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_snapshot_cleanup)
    status = checkpoint_logs.CheckpointStatus()
    assert checkpoint_logs.checkpoint(source, durable, status) == 1
    assert "cleanup" in status.last_error
    assert not status.complete
    assert checkpoint_logs._valid_eval(durable / "run.eval")
    monkeypatch.setattr(Path, "unlink", original_unlink)
    assert checkpoint_logs.checkpoint(source, durable, status) == 1
    assert status.complete


def test_source_io_error_is_not_reported_as_incomplete(directories, monkeypatch):
    source, durable = directories
    write_eval(source / "run.eval", "good")

    def fail_read(_path):
        raise OSError(errno.EIO, "injected read error")

    monkeypatch.setattr(checkpoint_logs, "_valid_eval", fail_read)
    status = checkpoint_logs.CheckpointStatus()
    assert checkpoint_logs.checkpoint(source, durable, status) == 0
    assert status.errors and status.incomplete == 0


def test_invalid_durable_copy_preserves_previous_checkpoint(directories, monkeypatch):
    source, durable = directories
    write_eval(source / "run.eval", "old")
    checkpoint_logs.checkpoint(source, durable)
    previous = (durable / "run.eval").read_bytes()
    write_eval(source / "run.eval", "new")
    original_copy = checkpoint_logs._copy_and_fsync

    def corrupt_copy(src, dst):
        if dst.parent == durable:
            dst.write_bytes(b"corrupt durable transfer")
        else:
            original_copy(src, dst)

    monkeypatch.setattr(checkpoint_logs, "_copy_and_fsync", corrupt_copy)
    status = checkpoint_logs.CheckpointStatus()
    assert checkpoint_logs.checkpoint(source, durable, status) == 0
    assert "invalid durable copy" in status.last_error
    assert (durable / "run.eval").read_bytes() == previous


@pytest.mark.parametrize(
    "contents,expected_code", [([], 1), ([False], 1), ([True, False], 1), ([True], 0)]
)
def test_once_requires_every_expected_log_and_writes_status(
    directories, tmp_path, contents, expected_code
):
    source, durable = directories
    for index, valid in enumerate(contents):
        path = source / f"run-{index}.eval"
        if valid:
            write_eval(path, "good")
        else:
            path.write_bytes(b"incomplete")
    status_file = tmp_path / "status.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-dir",
            str(source),
            "--durable-dir",
            str(durable),
            "--status-file",
            str(status_file),
            "--once",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected_code, result.stderr
    status = json.loads(status_file.read_text())
    assert status["expected"] == len(contents)
    assert status["published"] == sum(contents)
    assert status["checked_at"] > 0
    assert status["total_published"] == sum(contents)


def test_continuous_heartbeat_and_successful_final_publication(directories, tmp_path):
    source, durable = directories
    write_eval(source / "run.eval", "good")
    status_file = tmp_path / "status.json"
    with subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--source-dir",
            str(source),
            "--durable-dir",
            str(durable),
            "--status-file",
            str(status_file),
            "--interval",
            "0.05",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    ) as publisher:
        try:
            deadline = time.monotonic() + 5
            observed = None
            while time.monotonic() < deadline:
                if status_file.exists():
                    observed = json.loads(status_file.read_text())
                    if observed["total_published"] >= 2:
                        break
                time.sleep(0.01)
            assert observed and observed["total_published"] >= 2
            assert observed["pid"] == publisher.pid
            publisher.terminate()
            assert publisher.wait(timeout=5) == 0
            final = json.loads(status_file.read_text())
            assert final["total_published"] > observed["total_published"]
            assert checkpoint_logs._valid_eval(durable / "run.eval")
        finally:
            if publisher.poll() is None:
                publisher.kill()
                publisher.wait(timeout=5)


def test_directory_failure_reports_error_without_losing_previous_health(
    directories, monkeypatch
):
    source, durable = directories
    write_eval(source / "run.eval", "good")
    status = checkpoint_logs.CheckpointStatus()
    checkpoint_logs.checkpoint(source, durable, status)
    previous_time = status.last_published_at
    original_iterdir = Path.iterdir

    def fail_listing(path):
        if path == source:
            raise OSError(errno.EIO, "injected directory failure")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_listing)
    assert checkpoint_logs.checkpoint(source, durable, status) == 0
    assert status.expected == 0 and status.errors
    assert status.last_published_at == previous_time
    assert status.checked_at >= previous_time
    monkeypatch.setattr(Path, "iterdir", original_iterdir)
    assert checkpoint_logs.checkpoint(source, durable, status) == 1
    assert status.complete
