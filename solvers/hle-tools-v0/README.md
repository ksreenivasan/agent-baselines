# hle-tools-v0

Minimal InspectAI adapter for the pinned HLE-Verified tool-use condition.

- Dataset: `skylenage-ai/HLE-Verified@0bc83643672d4f68a5f89998617a639d85e7318b`.
- Evaluation pool: the card-recommended `Gold subset` (668 of 2,500 rows).
- The loader reads the final `question`, `answer`, `answer_type`, and image from each row's structured `json` field.
- Raw datasets, manifests, and run logs stay under ignored `protected/` paths.
- The upstream dataset is public and ungated, but declares no license in its card or repository; do not redistribute it from this project.
- Generated Python runs in the no-network Docker sandbox in `sandbox/`.
- Model keys are injected at runtime by `run_with_secrets.py`; key values are never printed.
- Search uses Exa when `~/secrets_and_keys/exa.key` exists. Without it, only fixture search is enabled and live runs are blocked.

## Offline checks

```bash
uv sync --project solvers/hle-tools-v0 --python 3.11
PYTHONPATH=. uv run --project solvers/hle-tools-v0 --frozen -- pytest -q tests/hle_tools_v0
```

## Pinned data and manifests

Set `HLE_VERIFIED_DATA_PATH` to the protected snapshot directory containing all 15 Parquet parts. Deterministic protected manifests are generated with:

```bash
PYTHONPATH=. uv run --project solvers/hle-tools-v0 --python 3.11 --frozen -- \
  python -m agent_baselines.evals.hle_tools_v0.cli build-manifests \
  --data-path "$HLE_VERIFIED_DATA_PATH" \
  --output-dir protected/manifests-hle-verified-gold
```

Only the sanitized balance table and manifest hashes belong in Git.

## Host-awake requirement

Inspect 0.3.203 enforces attempt and sample time limits with AnyIO cancellation scopes. Those scopes cannot advance while the macOS host is asleep. Wrap live evaluation in `caffeinate -i` so wall-clock observations correspond to runnable event-loop time:

```bash
caffeinate -i python3 solvers/hle-tools-v0/run_with_secrets.py ...
```

Provider SDK timeouts remain defense in depth; `caffeinate` does not replace them. The preserved aborted pilot was interrupted by host idle sleep, not evidence of a 42-minute OpenAI request.

## No-inference preflight

```bash
HLE_VERIFIED_DATA_PATH=protected/hle-verified-0bc83643672d4f68a5f89998617a639d85e7318b \
PYTHONPATH=. uv run --project solvers/hle-tools-v0 --frozen -- \
  python -m agent_baselines.evals.hle_tools_v0.cli preflight
```

See `PLAN.md`, `protocols/hle-tools-v0.yaml`, and `reports/HLE_VERIFIED_MIGRATION.md` for the controlled condition and evidence review.
