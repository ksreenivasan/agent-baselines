# HLE-Verified direct condition

This fork maintains two separate evaluation conditions over the same pinned HLE-Verified Gold data and deterministic manifests:

| Condition | Task | Model generations | Model-visible tools | Submission path |
|---|---|---:|---|---|
| Direct/no-tools | `agent_baselines/evals/hle_verified_direct/task.py@hle_verified_direct` | exactly 1 | none | response is the JSON object |
| With-tools | `agent_baselines/evals/hle_tools_v0/task.py@hle_tools_v0` | at most 15 | Exa search, controlled fetch, stateful sandboxed Python, inherited submit on the final turn | inherited `submission` string containing the same JSON object |

Both conditions use:

- `skylenage-ai/HLE-Verified@0bc83643672d4f68a5f89998617a639d85e7318b`;
- the card-recommended 668-row Gold subset;
- the same protected smoke and pilot manifests;
- the same text/image serialization and answer-type metadata;
- the same decoded schema: `answer`, `confidence`, and `explanation`;
- deterministic multiple-choice comparison and normalized/semantic exact-answer judging.

Their scores are **separate scaffold conditions**. Direct and with-tools results must not be merged or presented as interchangeable HLE scores.

The direct task performs one solver-model generation. Its scorer may separately call the pinned semantic judge when an exact answer is not a normalized string match; that evaluator call is not a solver tool or a second attempt by the evaluated model.

## Offline check

Use the existing pinned HLE tools environment:

```bash
PYTHONPATH=. uv run --project solvers/hle-tools-v0 --python 3.11 --frozen -- \
  pytest -q tests/hle_tools_v0/test_direct.py
```

## Protected live smoke shape

Live runs require `HLE_VERIFIED_DATA_PATH`, runtime-injected provider credentials, and a protected manifest. On macOS, wrap the command in `caffeinate -i`. Do not commit dataset files, manifest IDs, model output, judge output, or `.eval` logs.

See `protocols/hle-verified-direct-v0.yaml` and `protocols/hle-tools-v0.yaml` for the exact conditions.
