import importlib.util
import json
from pathlib import Path

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.event import InfoEvent
from inspect_ai.log import (
    EvalError,
    read_eval_log,
    read_eval_log_samples,
    write_eval_log,
)
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import match

from agent_baselines.evals.hle_tools_v0 import scorer as judge_module
from agent_baselines.evals.hle_tools_v0.recovery import file_digest, generation_digest

SCRIPT = (
    Path(__file__).resolve().parents[2] / "solvers/hle-tools-v0/m2/rescore_saved.py"
)
spec = importlib.util.spec_from_file_location("rescore_saved", SCRIPT)
repair_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair_module)


def source_ledger(tmp_path, completion):
    log = eval(
        Task(
            dataset=[Sample(id="saved-id", input="question", target="42")],
            scorer=match(),
        ),
        model=get_model("mockllm/model", custom_outputs=[completion], memoize=False),
        display="none",
        log_dir=str(tmp_path / "initial"),
    )[0]
    sample = log.samples[0]
    sample.scores = None
    sample.output = ModelOutput.from_content("mockllm/model", completion)
    sample.error = EvalError(
        message="EqualityJudgment invalid JSON",
        traceback="scorer.py EqualityJudgment",
        traceback_ansi="",
    )
    sample.events.append(
        InfoEvent(source="original_judge_attempt", data="original malformed")
    )
    log.status = "started"
    log.results = log.reductions = None
    log.eval.scorers[0].name = "hle_scorer"
    log.eval.scorers[0].options = {
        "judge_model": "mockllm/judge",
        "judge_reasoning_effort": "medium",
    }
    source = tmp_path / "judge-only.eval"
    write_eval_log(log, source)
    native = next(
        read_eval_log_samples(
            source, all_samples_required=False, resolve_attachments=True
        )
    )
    ledger = {
        "judge_only_shard": {
            "path": str(source),
            "sha256": file_digest(source),
            "ids": ["saved-id"],
        },
        "samples": [
            {
                "id": "saved-id",
                "disposition": "judge_only",
                "generation_sha256": generation_digest(
                    native.model_dump(mode="json", exclude_none=True)
                ),
            }
        ],
    }
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps(ledger))
    return path, native


def mock_judge(monkeypatch, outputs):
    calls = []

    class Judge:
        name = "mockllm/judge"

        async def generate(self, **kwargs):
            calls.append(kwargs)
            text = outputs[min(len(calls) - 1, len(outputs) - 1)]
            return ModelOutput.from_content(self.name, text)

    monkeypatch.setattr(judge_module, "get_model", lambda model: Judge())
    return calls


VALID = '{"extracted_final_answer":"42","reasoning":"match","correct":"yes","confidence":100}'


@pytest.mark.parametrize("completion", ["saved full answer", ""])
def test_only_judge_runs_and_empty_generation_is_preserved(
    tmp_path, monkeypatch, completion
):
    ledger, original = source_ledger(tmp_path, completion)
    calls = mock_judge(monkeypatch, ["invalid JSON", VALID])
    result = repair_module.repair(ledger, tmp_path / "repair")
    assert result["complete"]
    assert len(calls) == 2
    assert f"[response]: {completion}" in calls[0]["input"]
    restored = read_eval_log(result["archive"], resolve_attachments=True).samples[0]
    assert repair_module.stable_digest(restored) == repair_module.stable_digest(
        original
    )
    assert restored.output.completion == completion
    assert restored.error is None
    assert set(restored.scores) == {"hle_scorer"}
    old = [
        e
        for e in restored.events
        if getattr(e, "source", None) == "original_judge_attempt"
    ]
    new = [
        e for e in restored.events if getattr(e, "source", None) == "hle_judge_attempt"
    ]
    assert len(old) == 1 and len(new) == 2


def test_exhausted_judge_retains_attempts_without_publishing_success(
    tmp_path, monkeypatch
):
    ledger, original = source_ledger(tmp_path, "")
    calls = mock_judge(monkeypatch, ["invalid JSON"])
    output = tmp_path / "repair"
    result = repair_module.repair(ledger, output)
    assert not result["complete"]
    assert len(calls) == 3
    assert not (output / "repaired.eval").exists()
    restored = read_eval_log(
        result["attempts"][0]["path"], resolve_attachments=True
    ).samples[0]
    assert repair_module.stable_digest(restored) == repair_module.stable_digest(
        original
    )
    assert restored.error and not restored.scores
    assert (
        len(
            [
                e
                for e in restored.events
                if getattr(e, "source", None) == "hle_judge_attempt"
            ]
        )
        == 3
    )


def test_source_checksum_mismatch_stops_before_judge_call(tmp_path, monkeypatch):
    ledger, _ = source_ledger(tmp_path, "saved")
    value = json.loads(ledger.read_text())
    value["judge_only_shard"]["sha256"] = "changed"
    ledger.write_text(json.dumps(value))
    calls = mock_judge(monkeypatch, [VALID])
    with pytest.raises(ValueError, match="checksum"):
        repair_module.repair(ledger, tmp_path / "repair")
    assert calls == []
