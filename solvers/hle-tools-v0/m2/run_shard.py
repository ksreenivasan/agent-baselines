#!/usr/bin/env python3
"""Supervise one bounded HLE shard and its node-local checkpoint publisher."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import zipfile
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def make_command(config: dict, manifest: Path, local: Path, repo: Path) -> list[str]:
    executable = str(Path(sys.executable).parent / "inspect")
    task = config.get("task", "hle_tools")
    targets = {
        "hle_tools": "agent_baselines/evals/hle_tools_v0/task.py@hle_tools_v0",
        "hle_direct": "agent_baselines/evals/hle_direct/task.py@hle_direct",
        "hle_tools_canary": "solvers/hle-tools-v0/m2/canary.py@hle_tools_canary",
    }
    if task != "hle_direct" and config.get("search_backend") != "keenable":
        raise ValueError("this live campaign requires the frozen Keenable backend")
    if task not in targets:
        raise ValueError("unsupported HLE task")
    command = [
        executable,
        "eval",
        str(repo / targets[task]),
        "--model",
        config["model"],
        "--reasoning-effort",
        config["reasoning_effort"],
        "--max-connections",
        str(config["concurrency"]),
        "--max-samples",
        str(config["concurrency"]),
        "--max-sandboxes",
        str(config["concurrency"]),
        "--max-dataset-memory",
        "2048",
        "--max-retries",
        str(config["max_retries"]),
        "--timeout",
        str(config["timeout"]),
        "--attempt-timeout",
        str(config["attempt_timeout"]),
        "--no-fail-on-error",
        "--display",
        "plain",
        "--no-score-display",
        "--log-format",
        "eval",
        "--log-buffer",
        "1",
        "--log-dir",
        str(local),
    ]
    if task != "hle_tools_canary":
        command += [
            "-T",
            f"data_path={config['dataset_path']}",
            "-T",
            f"dataset_revision={config['dataset_revision']}",
            "-T",
            f"manifest_path={manifest}",
            "-T",
            f"judge_model={config['judge_model']}",
            "-T",
            "judge_reasoning_effort=medium",
        ]
    if task == "hle_tools":
        command += ["-T", "sandbox_backend=m2-enroot"]
    if config.get("model_base_url"):
        command += [
            "--model-base-url",
            config["model_base_url"],
            "--reasoning-history",
            config["reasoning_history"],
        ]
    for field, flag in (
        ("temperature", "--temperature"),
        ("top_p", "--top-p"),
        ("max_tokens", "--max-tokens"),
    ):
        if config.get(field) is not None:
            command += [flag, str(config[field])]
    providers = ["openai"]
    if task != "hle_direct":
        providers += [config["search_backend"]]
    if config["model"].startswith("google/"):
        providers += ["google"]
    runner = [sys.executable, str(repo / "solvers/hle-tools-v0/run_with_secrets.py")]
    for provider in providers:
        runner += ["--provider", provider]
    return runner + command


def stop_process(process: subprocess.Popen | None, grace: float = 60) -> None:
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def publication_problem(
    status: dict, started: float, now: float, stale: float
) -> str | None:
    # The archive may not exist until the first model sample completes.
    # Pending completed work, rather than a long in-flight generation, starts the lag clock.
    last = status.get("last_published_at")
    if isinstance(last, str):
        from datetime import datetime

        last = datetime.fromisoformat(last.replace("Z", "+00:00")).timestamp()
    if now - (status.get("checked_at") or started) > stale:
        return "checkpoint publisher heartbeat stalled"
    if (
        status.get("errors") or (status.get("expected", 0) and status.get("incomplete"))
    ) and now - (last or started) > stale:
        return "checkpoint publication repeatedly failed"
    return None


def validate_archive(path: Path, expected: list[str]) -> dict:
    from inspect_ai.log import (
        read_eval_log,
        read_eval_log_sample,
        read_eval_log_sample_summaries,
    )
    from agent_baselines.evals.hle_tools_v0.recovery import (
        validate_samples,
        disposition,
    )

    header = read_eval_log(str(path), header_only=True)
    if header.status != "success" or header.invalidated or header.error:
        return {"valid": False, "reason": "archive header is not successful and valid"}
    summaries = read_eval_log_sample_summaries(str(path))
    samples = (
        read_eval_log_sample(str(path), sample.id, sample.epoch).model_dump(mode="json")
        for sample in summaries
    )
    result = validate_samples(samples, expected)
    if not result["valid"] and expected != ["hle-tools-canary"]:
        rows = (
            read_eval_log_sample(str(path), item.id, item.epoch).model_dump(mode="json")
            for item in summaries
        )
        structural = validate_samples(rows, expected, allow_errors=True)
        residual = []
        if structural["valid"] and structural.get("counts", {}).get("retain_score", 0):
            for item in summaries:
                row = read_eval_log_sample(str(path), item.id, item.epoch).model_dump(
                    mode="json"
                )
                kind, why = disposition(row)
                if kind != "retain_score":
                    residual.append(
                        {"id": str(item.id), "disposition": kind, "reason": why}
                    )
            result["residual"] = residual
            result["continuable"] = bool(residual) and all(
                row["reason"]
                in {"provider_infrastructure_error", "malformed_judge_result"}
                for row in residual
            )
    if expected == ["hle-tools-canary"]:
        sample = read_eval_log_sample(str(path), expected[0], 1)
        trajectory = (sample.scores or {}).get("trajectory")
        if (
            trajectory is None
            or trajectory.value != "C"
            or any(score.value != "C" for score in (sample.scores or {}).values())
        ):
            result["valid"] = False
            result["reason"] = "model did not complete the multistep canary"
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-interval", type=float, default=60)
    parser.add_argument("--max-durable-lag", type=float, default=300)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run inside a Slurm allocation")
    repo = Path(__file__).resolve().parents[3]
    config = json.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    expected = [str(value) for value in manifest["ids"]]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("manifest must contain nonempty unique IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "launch.json").exists():
        raise ValueError(
            "attempt directory already launched; select a new attempt directory"
        )
    scratch = Path(os.environ.get("SLURM_TMPDIR", "/var/tmp"))
    local = (
        scratch
        / f"hle-resume-{os.getuid()}-{os.environ['SLURM_JOB_ID']}"
        / f"{args.output.parent.name}-{args.output.name}"
    )
    local.mkdir(parents=True, exist_ok=False)
    live = local / "logs"
    live.mkdir()
    durable = args.output / "logs"
    durable.mkdir()
    health = local / "checkpoint-status.json"
    guard = local / "tool-guard"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    env["HLE_SEARCH_BACKEND"] = config.get("search_backend", "")
    env["HLE_TOOL_GUARD_DIR"] = str(guard)
    if config.get("vllm_key_file"):
        env["VLLM_API_KEY"] = Path(config["vllm_key_file"]).read_text().strip()
    command = make_command(config, args.manifest.resolve(), live, repo)
    launch = {
        "config": config,
        "manifest": manifest,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "job_id": os.environ["SLURM_JOB_ID"],
        "local_root": str(local),
        "started_at": time.time(),
        "command": command,
    }
    atomic_json(args.output / "launch.json", launch)
    publisher_command = [
        sys.executable,
        str(repo / "solvers/hle-tools-v0/m2/checkpoint_eval_logs.py"),
        "--source-dir",
        str(live),
        "--durable-dir",
        str(durable),
        "--status-file",
        str(health),
        "--interval",
        str(args.checkpoint_interval),
    ]
    stopped = False
    reason = None

    def on_signal(_signum: int, _frame: Any) -> None:
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    process = publisher = None
    result: dict[str, Any] = {"complete": False, "local_root": str(local)}
    try:
        with (
            (local / "publisher.log").open("a") as publog,
            (local / "eval.log").open("a") as evallog,
        ):
            publisher = subprocess.Popen(
                publisher_command, stdout=publog, stderr=subprocess.STDOUT, env=env
            )
            process = subprocess.Popen(
                command, stdout=evallog, stderr=subprocess.STDOUT, env=env
            )
            started = time.time()
            while process.poll() is None and not stopped:
                if publisher.poll() is not None:
                    reason = "checkpoint publisher exited"
                    break
                if (guard / "fatal.json").exists():
                    reason = "tool infrastructure circuit breaker"
                    break
                if health.exists():
                    state = json.loads(health.read_text())
                    reason = publication_problem(
                        state, started, time.time(), args.max_durable_lag
                    )
                    if reason:
                        break
                elif time.time() - started > args.max_durable_lag:
                    reason = "checkpoint publisher never emitted a heartbeat"
                    break
                time.sleep(5)
            if stopped:
                reason = "supervisor received termination"
            if reason:
                stop_process(process)
            eval_rc = process.wait()
            stop_process(publisher)
            final_pub = subprocess.run(
                publisher_command + ["--once"],
                stdout=publog,
                stderr=subprocess.STDOUT,
                env=env,
                timeout=args.max_durable_lag,
            )
        # Keep diagnostics available even if the shared filesystem is still failing.
        for name in ("eval.log", "publisher.log", "checkpoint-status.json"):
            source = local / name
            if source.exists():
                import shutil

                shutil.copyfile(source, args.output / name)
        if guard.exists():
            import shutil

            shutil.copytree(guard, args.output / "tool-guard", dirs_exist_ok=True)
        result = {
            "eval_exit": eval_rc,
            "publisher_exit": final_pub.returncode,
            "reason": reason,
            "finished_at": time.time(),
            "local_root": str(local),
        }
        paths = sorted(durable.glob("*.eval"))
        if len(paths) != 1:
            result["reason"] = (
                result["reason"] or "expected exactly one durable archive"
            )
        elif final_pub.returncode == 0:
            result["archive"] = str(paths[0])
            result["validation"] = validate_archive(paths[0], expected)
        complete = (
            not result["reason"]
            and eval_rc == 0
            and final_pub.returncode == 0
            and len(paths) == 1
            and result.get("validation", {}).get("valid", False)
        )
        result["complete"] = complete
        atomic_json(local / "result.json", result)
        atomic_json(args.output / "result.json", result)
        print(json.dumps(result, sort_keys=True), flush=True)
        if complete:
            return 0
        if (
            not result["reason"]
            and eval_rc == 0
            and final_pub.returncode == 0
            and result.get("validation", {}).get("continuable", False)
        ):
            return 3
        return 2
    except (
        OSError,
        ValueError,
        zipfile.BadZipFile,
        subprocess.TimeoutExpired,
    ) as error:
        result.update(
            complete=False,
            reason=f"{type(error).__name__}: {error}",
            finished_at=time.time(),
        )
        atomic_json(local / "result.json", result)
        try:
            atomic_json(args.output / "result.json", result)
        except OSError as durable_error:
            print(
                f"Durable failure status unavailable: {durable_error}; "
                f"local result: {local / 'result.json'}",
                file=sys.stderr,
                flush=True,
            )
        return 2
    finally:
        stop_process(process)
        stop_process(publisher)


if __name__ == "__main__":
    raise SystemExit(main())
