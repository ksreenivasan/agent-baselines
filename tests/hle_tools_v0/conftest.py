"""Small native-schema fixtures shared by recovery and publication tests."""

import pytest
from inspect_ai.log import EvalSample


@pytest.fixture
def judge_provider_error_row():
    def make(*, sample_id="b", completion="", error_type="RateLimitError"):
        # Mirrors pinned Inspect's native scorer span and failed ModelEvent.
        error = f"{error_type}: provider unavailable"
        return EvalSample.model_validate(
            {
                "id": sample_id,
                "epoch": 1,
                "input": "question",
                "target": "42",
                "output": {"model": "mockllm/solver", "completion": completion},
                "error": {
                    "message": error,
                    "traceback": (
                        'File "/source/agent_baselines/evals/hle_tools_v0/scorer.py", line 54, in score\n'
                        "    output = await judge.generate(\n" + error
                    ),
                    "traceback_ansi": "",
                },
                "events": [
                    {
                        "event": "span_begin",
                        "id": "scorers",
                        "type": "scorers",
                        "name": "scorers",
                    },
                    {
                        "event": "span_begin",
                        "id": "hle",
                        "parent_id": "scorers",
                        "type": "scorer",
                        "name": "hle_scorer",
                    },
                    {
                        "event": "model",
                        "span_id": "hle",
                        "model": "mockllm/judge",
                        "input": [],
                        "tools": [],
                        "tool_choice": "auto",
                        "config": {},
                        "output": {},
                        "error": error,
                    },
                    {"event": "span_end", "id": "hle"},
                    {"event": "span_end", "id": "scorers"},
                ],
            }
        ).model_dump(mode="json", exclude_none=True)

    return make
