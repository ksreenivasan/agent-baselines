#!/usr/bin/env python3
"""Offline Inspect recorder/publisher soak; run on an allocated compute node."""

from __future__ import annotations

import argparse
import asyncio
import errno
import gc
import hashlib
import json
import random
import resource
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from inspect_ai.event import ModelEvent, ToolEvent
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalPlan,
    EvalSample,
    EvalSpec,
    EvalStats,
)
from inspect_ai.log._recorders.eval import EvalRecorder
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.tool import ToolCall

import checkpoint_eval_logs as publisher


def memory() -> dict[str, int | None]:
    rss = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("VmRSS:")
    )
    cgroup = None
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        hierarchy, controllers, relative = line.split(":", 2)
        if hierarchy == "0":
            candidate = Path("/sys/fs/cgroup") / relative.lstrip("/") / "memory.current"
        elif "memory" in controllers.split(","):
            candidate = (
                Path("/sys/fs/cgroup/memory")
                / relative.lstrip("/")
                / "memory.usage_in_bytes"
            )
        else:
            continue
        if candidate.exists():
            cgroup = int(candidate.read_text())
    return {"rss_bytes": rss, "cgroup_bytes": cgroup}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sample(index: int, turns: int, chunk_bytes: int) -> EvalSample:
    rng = random.Random(index)
    messages = [ChatMessageUser(content=f"Offline long trace {index}")]
    events = []
    for turn in range(turns):
        call_id = f"{index}-{turn}"
        query = {"query": f"synthetic query {index} {turn}"}
        response = rng.randbytes(chunk_bytes // 2).hex()
        assistant = ChatMessageAssistant(
            content=rng.randbytes(chunk_bytes // 8).hex(),
            tool_calls=[ToolCall(id=call_id, function="web_search", arguments=query)],
        )
        events.append(
            ModelEvent(
                model="offline/synthetic",
                input=list(messages),
                tools=[],
                tool_choice="auto",
                config=GenerateConfig(),
                output=ModelOutput(),
            )
        )
        messages.extend(
            [assistant, ChatMessageTool(content=response, tool_call_id=call_id)]
        )
        events.append(
            ToolEvent(
                id=call_id, function="web_search", arguments=query, result=response
            )
        )
    messages.append(ChatMessageAssistant(content="synthetic final answer"))
    return EvalSample(
        id=f"soak-{index:04}",
        epoch=1,
        input="offline",
        target="offline",
        messages=messages,
        events=events,
    )


async def run(args: argparse.Namespace, live: Path) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    durable = args.output_dir / "checkpoints"
    durable.mkdir(exist_ok=True)
    if list(durable.glob("*.eval")):
        raise ValueError("Use a fresh output directory for each soak")
    created = datetime.now(timezone.utc).isoformat()
    spec = EvalSpec(
        created=created,
        task="offline-logging-soak",
        run_id="offline-soak",
        dataset=EvalDataset(samples=args.samples),
        model="offline/synthetic",
        config=EvalConfig(),
    )
    recorder = EvalRecorder(str(live))
    location = Path(await recorder.log_init(spec, str(live / "soak.eval")))
    await recorder.log_start(spec, EvalPlan())
    status = publisher.CheckpointStatus()
    points = []
    durations = []
    injected = False
    start = time.monotonic()
    for index in range(args.samples):
        await recorder.log_sample(spec, sample(index, args.turns, args.chunk_bytes))
        # The campaign uses a one-sample log buffer.
        await recorder.flush(spec)
        if (index + 1) % 25 == 0 or index + 1 == args.samples:
            gc.collect()
            points.append({"samples": index + 1, **memory()})
            checkpoint_start = time.monotonic()
            assert publisher.checkpoint(live, durable, status) == 1, status.errors
            durations.append(time.monotonic() - checkpoint_start)
            if not injected:
                previous_hash = digest(durable / location.name)
                original_copy = publisher._copy_and_fsync

                def fail_copy(source: Path, destination: Path) -> None:
                    if destination.parent == durable:
                        destination.write_bytes(b"injected partial write")
                        raise OSError(errno.ENOSPC, "offline soak injected failure")
                    original_copy(source, destination)

                with patch.object(publisher, "_copy_and_fsync", fail_copy):
                    assert publisher.checkpoint(live, durable, status) == 0
                assert digest(durable / location.name) == previous_hash
                assert publisher.checkpoint(live, durable, status) == 1
                injected = True
            print(json.dumps(points[-1]), flush=True)
    await recorder.log_finish(
        spec, "success", EvalStats(started_at=created), None, None, header_only=True
    )
    assert publisher.checkpoint(live, durable, status) == 1
    found = []
    with zipfile.ZipFile(durable / location.name) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            if name.startswith("samples/") and name.endswith(".json"):
                found.append(json.loads(archive.read(name))["id"])
    expected = {f"soak-{index:04}" for index in range(args.samples)}
    assert len(found) == args.samples and set(found) == expected
    payload_bytes = args.samples * args.turns * args.chunk_bytes * 1.25
    initial_rss, final_rss = points[0]["rss_bytes"], points[-1]["rss_bytes"]
    assert initial_rss is not None and final_rss is not None
    rss_growth = final_rss - initial_rss
    # A loose guard against retaining an entire shard's message contents; this
    # offline probe does not establish bounds for provider/client memory.
    assert rss_growth < max(64 * 1024 * 1024, payload_bytes / 2), points
    report = {
        "samples": args.samples,
        "turns_per_sample": args.turns,
        "tool_bytes_per_turn": args.chunk_bytes,
        "synthetic_message_bytes": payload_bytes,
        "archive_bytes": (durable / location.name).stat().st_size,
        "elapsed_seconds": time.monotonic() - start,
        "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "memory_points": points,
        "rss_growth_after_warmup_bytes": rss_growth,
        "checkpoint_seconds": durations,
        "exact_ids_verified": True,
        "failed_copy_preserved_previous_archive": injected,
        "recovery_verified": status.complete,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--turns", type=int, default=15)
    parser.add_argument("--chunk-bytes", type=int, default=32768)
    args = parser.parse_args()
    if min(args.samples, args.turns, args.chunk_bytes) <= 0:
        parser.error("sizes must be positive")
    with tempfile.TemporaryDirectory(
        prefix="hle-logging-soak-", dir="/tmp"
    ) as directory:
        print(json.dumps(asyncio.run(run(args, Path(directory))), indent=2))


if __name__ == "__main__":
    main()
