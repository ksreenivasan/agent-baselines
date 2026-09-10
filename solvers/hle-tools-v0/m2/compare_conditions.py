#!/usr/bin/env python3
"""Compare accepted HLE direct/tools conditions; never generate or score answers.

Example:
  python compare_conditions.py --expected-manifest manifests/all.json \
    --tools-aggregate final/tools --direct-aggregate final/direct --output paired
For historical GPT/Gemini controls, replace --direct-aggregate with --direct-audit
controls/gpt-gemini-direct-audit/LANE.json. Both inputs must cover all 2158 IDs.
Native scores and generation hashes are checked against their bound archives.
Run in the existing frozen HLE environment; the output records its NumPy version,
PCG64 generator, seed, cell order and quantile method for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from inspect_ai.log import read_eval_log

from aggregate_campaign import checked_file, read_json, require
from agent_baselines.evals.hle_tools_v0.recovery import (
    digest,
    disposition,
    file_digest,
    generation_digest,
    iter_samples,
    read_json_header,
)

COUNT = 2158
REVISION = "021a3d71f516a7ac28ceb8d284969902edf1edeb"
RESAMPLES = 10000
SEED = 20260910
QUALIFICATIONS = [
    "This is a descriptive comparison of accepted outcomes, not an identical-load "
    "causal effect or a first-attempt-only estimate.",
    "GLM/K2 direct controls use concurrency 12 while tools use concurrency 3. "
    "Shared endpoint load, source revisions, budgets, and recovery opportunities differ.",
    "Historical GPT/Gemini direct repairs regenerated solver answers; tools retain historical "
    "answers and use separately recorded judge-only and infrastructure recovery. "
    "Model-name normalization does not equate source or opportunity histories.",
    "The interval describes question-level variation conditional on the selected "
    "generations; it does not measure repeated-generation uncertainty.",
]


def binding(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_digest(path)}


def jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def indexed(rows: list[dict], ids: list[str]) -> dict[str, dict]:
    result = {}
    for row in rows:
        sample_id = row.get("id")
        require(
            isinstance(sample_id, str)
            and sample_id in ids
            and sample_id not in result
            and type(row.get("epoch")) is int
            and row["epoch"] == 1,
            "unknown/duplicate ID or invalid epoch",
        )
        require(type(row.get("correct")) is bool, "correct must be a strict boolean")
        require(
            not row.get("error") and not row.get("invalidated"),
            "invalidated or failed outcome",
        )
        assert isinstance(sample_id, str)
        result[sample_id] = row
    require(set(result) == set(ids), "condition does not contain every expected ID")
    return result


def model_identity(model: str) -> str:
    # Only this deployed adapter alias differs between the two conditions.
    if model == "k2-vllm/IFM/K2-Horizon-375B-A23B":
        return "vllm/IFM/K2-Horizon-375B-A23B"
    return model


def native_scores(rows: dict[str, dict], *, direct: bool, model: str) -> None:
    """Verify accepted score/answer identity without changing valid I or limits."""
    archives: dict[tuple[str, str], dict[str, dict]] = {}
    for sample_id, row in rows.items():
        archives.setdefault((row["archive"], row["archive_sha256"]), {})[
            sample_id
        ] = row
    for (filename, checksum), expected in archives.items():
        path = checked_file(filename, checksum)
        header = (
            read_eval_log(path, header_only=True).model_dump(
                mode="json", exclude_none=True
            )
            if path.suffix == ".eval"
            else read_json_header(path)
        )
        spec = header["eval"]
        args = spec["task_args"]
        require(
            header.get("status") in {"success", "started", "cancelled"}
            and not header.get("invalidated")
            and not header.get("error")
            and spec["model"] == model
            and spec["task"].split("/")[-1]
            == ("hle_direct" if direct else "hle_tools_v0")
            and args.get("dataset_revision") == REVISION
            and args.get("dataset_variant") == "standard"
            and args.get("fixture") is False,
            "native source condition/header mismatch",
        )
        seen = set()
        for sample in iter_samples(path):
            sample_id = str(sample["id"])
            if sample_id not in expected:
                continue
            require(
                sample_id not in seen
                and type(sample.get("epoch", 1)) is int
                and sample.get("epoch", 1) == 1,
                "duplicate native ID or invalid native epoch",
            )
            seen.add(sample_id)
            require(
                disposition(sample)[0] == "retain_score", "unaccepted native sample"
            )
            score = sample.get("scores", {})
            require(
                set(score) == {"hle_scorer"}
                and score["hle_scorer"].get("value") in {"C", "I"}
                and not score["hle_scorer"]
                .get("metadata", {})
                .get("hle_judge_repair_failed"),
                "invalid native score",
            )
            saved = expected[sample_id]
            require(
                saved["correct"] == (score["hle_scorer"]["value"] == "C")
                and saved["generation_sha256"] == generation_digest(sample),
                "native score or generation differs from accepted evidence",
            )
            if direct:
                require(
                    not any(
                        event.get("event") == "tool"
                        or (event.get("event") == "model" and event.get("tools"))
                        for event in sample.get("events", [])
                    ),
                    "direct source contains tools",
                )
        require(seen == set(expected), "native source lacks selected IDs")


def load_aggregate(directory: Path, ids: list[str], *, direct: bool) -> dict:
    summary_path = directory / "summary.json"
    summary = read_json(summary_path)
    require(
        summary.get("complete") is True
        and summary.get("samples") == COUNT
        and summary.get("dataset_revision") == REVISION
        and summary.get("dataset_ids_sha256") == digest(ids),
        "incomplete or incompatible aggregate",
    )
    rows = indexed(jsonl(directory / "outcomes.jsonl"), ids)
    counts = Counter(row["correct"] for row in rows.values())
    require(
        type(summary.get("correct")) is int
        and type(summary.get("incorrect")) is int
        and summary["correct"] == counts[True]
        and summary["incorrect"] == counts[False]
        and summary.get("accuracy") == counts[True] / COUNT,
        "aggregate totals disagree with per-ID evidence",
    )
    shards = json.loads((directory / "shards.json").read_text())
    selected: dict[str, tuple[str, str]] = {}
    for shard in shards:
        for sample_id in shard["selected_ids"]:
            require(
                sample_id not in selected, "overlapping aggregate source selections"
            )
            selected[sample_id] = (str(Path(shard["path"]).resolve()), shard["sha256"])
    require(set(selected) == set(ids), "aggregate source selections are incomplete")
    for sample_id, row in rows.items():
        require(
            (str(Path(row["archive"]).resolve()), row["archive_sha256"])
            == selected[sample_id],
            "outcome is not bound to its selected source",
        )
    if direct:
        require(summary.get("task") == "hle_direct", "expected a direct aggregate")
        protocol = read_json(
            checked_file(summary["protocol"], summary["protocol_sha256"])
        )
        require(
            protocol["dataset_revision"] == REVISION
            and protocol["dataset_variant"] == "standard",
            "direct protocol dataset mismatch",
        )
        config_ref = protocol["lanes"][summary["lane"]]["config"]
        config = read_json(checked_file(config_ref["path"], config_ref["sha256"]))
        require(config_ref["sha256"] == summary["config_sha256"], "config mismatch")
        stratum_key = "stratum"
    else:
        config = summary.get("configuration", {})
        require(
            config.get("task") == "hle_tools"
            and config.get("run_phase") == "production"
            and config.get("search_backend") == "keenable"
            and config.get("model") == summary.get("model"),
            "expected a production tools aggregate",
        )
        stratum_key = "kind"
    require(
        config.get("dataset_revision") == REVISION, "model config revision mismatch"
    )
    require(
        summary.get("source_counts")
        == dict(Counter(row[stratum_key] for row in rows.values())),
        "aggregate source counts disagree",
    )
    native_scores(rows, direct=direct, model=config["model"])
    return {
        "model": config["model"],
        "rows": rows,
        "provenance": {
            "format": "strict_direct_aggregate" if direct else "strict_tools_aggregate",
            "files": [
                binding(directory / name)
                for name in ("summary.json", "outcomes.jsonl", "shards.json")
            ],
            "summary": summary,
            "shards": shards,
        },
    }


def load_direct_audit(path: Path, ids: list[str]) -> dict:
    audit = read_json(path)
    require(
        audit.get("status")
        == "coverage_verified_with_lineage_and_opportunity_qualifications"
        and audit.get("selected_count") == COUNT
        and not audit.get("unresolved_ids")
        and not audit.get("selected_problem_counts")
        and audit.get("selected_tool_events") == 0
        and audit.get("selected_actual_solver_inputs_verified") == COUNT
        and audit.get("selected_canonical_judge_prompts_verified") == COUNT,
        "historical direct audit is incomplete or unaccepted",
    )
    dataset = audit["dataset"]
    require(
        dataset["revision"] == REVISION
        and dataset["variant"] == "standard"
        and dataset["count"] == COUNT,
        "historical direct dataset mismatch",
    )
    frozen = dataset["frozen_manifest"]
    require(
        read_json(checked_file(frozen["path"], frozen["sha256"]))["ids"] == ids,
        "historical direct membership/order mismatch",
    )
    sources = audit["sources"]
    models = set()
    for source in sources:
        spec = source["header"]["eval"]
        require(
            source.get("parse_valid") is True
            and spec["task"].split("/")[-1] == "hle_direct"
            and spec["task_args"]["dataset_revision"] == REVISION
            and spec["task_args"]["dataset_variant"] == "standard",
            "historical source is not the pinned direct condition",
        )
        models.add(spec["model"])
    require(len(models) == 1, "historical source models differ")
    reference = audit["per_id_evidence"]
    evidence_path = checked_file(reference["path"], reference["sha256"])
    normalized = []
    evidence = jsonl(evidence_path)
    for record in evidence:
        selected = record["selected"]
        require(
            selected.get("eligible") is True
            and selected.get("disposition") == "retain_score"
            and not selected.get("problems")
            and not selected.get("infrastructure_reasons")
            and not selected.get("error_type")
            and selected.get("tool_events") == 0
            and selected.get("full_solver_request_input_matches") is True
            and selected.get("canonical_judge_prompt_matches") is True
            and selected.get("score") in {"C", "I"},
            "historical selected row is unaccepted",
        )
        require(
            selected["id"] == record["id"]
            and selected["epoch"] == record["epoch"]
            and selected in record["observed_attempts"],
            "historical selected row differs from observed attempts",
        )
        index = selected["source_index"]
        require(
            type(index) is int and 0 <= index < len(sources), "invalid source index"
        )
        normalized.append(
            {
                "id": record["id"],
                "epoch": record["epoch"],
                "correct": selected["score"] == "C",
                "generation_sha256": selected["generation_sha256"],
                "archive": sources[index]["path"],
                "archive_sha256": sources[index]["sha256"],
                "source_index": index,
                "selected": selected,
                "observed_attempts": record["observed_attempts"],
            }
        )
    rows = indexed(normalized, ids)
    require(
        audit["selected_score_counts"]
        == dict(Counter("C" if row["correct"] else "I" for row in rows.values())),
        "historical score totals disagree",
    )
    require(
        audit["selected_source_counts"]
        == dict(Counter(str(row["source_index"]) for row in rows.values())),
        "historical source totals disagree",
    )
    native_scores(rows, direct=True, model=next(iter(models)))
    return {
        "model": next(iter(models)),
        "rows": rows,
        "provenance": {
            "format": "qualified_historical_direct_audit",
            "files": [binding(path), binding(evidence_path)],
            "audit": audit,
        },
    }


def paired_statistics(direct: list[bool], tools: list[bool]) -> dict:
    require(len(direct) == len(tools) and len(direct) > 0, "pair lengths differ")
    require(all(type(v) is bool for v in direct + tools), "pairs must be boolean")
    # Multinomial counts are exactly the empirical question-level bootstrap
    # distribution. Each draw keeps the two outcomes of a question together.
    cells = Counter((d, t) for d, t in zip(direct, tools))
    order = [(False, False), (True, False), (False, True), (True, True)]
    n = len(direct)
    generator = np.random.Generator(np.random.PCG64(SEED))
    draws = generator.multinomial(n, [cells[cell] / n for cell in order], RESAMPLES)
    differences = (draws[:, 2] - draws[:, 1]) * (100.0 / n)
    interval = np.quantile(differences, [0.025, 0.975], method="linear")
    return {
        "samples": n,
        "direct": {"correct": sum(direct), "accuracy": sum(direct) / n},
        "tools": {"correct": sum(tools), "accuracy": sum(tools) / n},
        "tools_minus_direct_percentage_points": 100 * (sum(tools) - sum(direct)) / n,
        "discordant": {
            "tools_correct_direct_incorrect": cells[(False, True)],
            "tools_incorrect_direct_correct": cells[(True, False)],
        },
        "concordant": {
            "both_correct": cells[(True, True)],
            "both_incorrect": cells[(False, False)],
        },
        "bootstrap": {
            "resamples": RESAMPLES,
            "seed": SEED,
            "rng": "NumPy PCG64",
            "numpy_version": np.__version__,
            "method": "paired question-level empirical multinomial counts",
            "cell_order": [
                "both_incorrect",
                "direct_only",
                "tools_only",
                "both_correct",
            ],
            "confidence_level": 0.95,
            "interval": "percentile",
            "quantile_method": "linear",
            "tools_minus_direct_percentage_points_ci": interval.tolist(),
        },
    }


def compare(
    expected_manifest: Path,
    tools_aggregate: Path,
    output: Path,
    *,
    direct_aggregate: Path | None = None,
    direct_audit: Path | None = None,
) -> dict:
    require(
        (direct_aggregate is None) != (direct_audit is None), "choose one direct input"
    )
    manifest = read_json(expected_manifest)
    ids = manifest["ids"]
    require(
        len(ids) == COUNT
        and len(set(ids)) == COUNT
        and all(isinstance(value, str) for value in ids)
        and manifest.get("dataset_revision") == REVISION
        and manifest.get("dataset_variant") == "standard",
        "expected manifest must contain 2158 unique pinned standard IDs",
    )
    require(not output.exists(), "comparison output already exists")
    tools = load_aggregate(tools_aggregate, ids, direct=False)
    if direct_aggregate is not None:
        direct = load_aggregate(direct_aggregate, ids, direct=True)
    else:
        assert direct_audit is not None
        direct = load_direct_audit(direct_audit, ids)
    require(
        model_identity(tools["model"]) == model_identity(direct["model"]),
        "direct and tools solver models differ",
    )
    result = {
        "version": 1,
        "complete": True,
        "dataset_revision": REVISION,
        "dataset_variant": "standard",
        "dataset_ids_sha256": digest(ids),
        "expected_manifest": binding(expected_manifest),
        "comparison_code": binding(Path(__file__)),
        "models": {"direct": direct["model"], "tools": tools["model"]},
        **paired_statistics(
            [direct["rows"][sid]["correct"] for sid in ids],
            [tools["rows"][sid]["correct"] for sid in ids],
        ),
        "qualifications": QUALIFICATIONS,
        "provenance": {
            "direct": direct["provenance"],
            "tools": tools["provenance"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as temp:
        staging = Path(temp) / "comparison"
        staging.mkdir(mode=0o700)
        paired = staging / "paired-outcomes.jsonl"
        with paired.open("w") as stream:
            for sid in ids:
                stream.write(
                    json.dumps(
                        {
                            "id": sid,
                            "epoch": 1,
                            "direct": direct["rows"][sid],
                            "tools": tools["rows"][sid],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        result["paired_outcomes_sha256"] = file_digest(paired)
        (staging / "comparison.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
        for path in staging.iterdir():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        os.rename(staging, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-manifest", type=Path, required=True)
    parser.add_argument("--tools-aggregate", type=Path, required=True)
    direct = parser.add_mutually_exclusive_group(required=True)
    direct.add_argument("--direct-aggregate", type=Path)
    direct.add_argument("--direct-audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    result = compare(
        args.expected_manifest,
        args.tools_aggregate,
        args.output,
        direct_aggregate=args.direct_aggregate,
        direct_audit=args.direct_audit,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
