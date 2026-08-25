# hle-tools-v0

Minimal InspectAI adapter for the pinned 2,500-question HLE tool-use condition.

- Raw HLE data and run logs stay outside Git.
- Generated Python runs in the no-network Docker sandbox in `sandbox/`.
- Model keys are injected at runtime by `run_with_secrets.py`; key values are never printed.
- Search uses Exa when `~/secrets_and_keys/exa.key` exists. Without it, only fixture search is enabled and live HLE runs are blocked.

## Offline checks

```bash
uv sync --project solvers/hle-tools-v0 --python 3.11
PYTHONPATH=. uv run --project solvers/hle-tools-v0 --frozen -- pytest -q tests/hle_tools_v0
```

## No-inference preflight

```bash
PYTHONPATH=. uv run --project solvers/hle-tools-v0 --frozen -- \
  python -m agent_baselines.evals.hle_tools_v0.cli preflight
```

See `PLAN.md` and `protocols/hle-tools-v0.yaml` for the controlled run condition.
