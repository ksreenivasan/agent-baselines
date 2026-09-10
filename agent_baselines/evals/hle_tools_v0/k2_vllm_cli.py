"""Register the scoped K2 provider before Inspect resolves the CLI model."""

from agent_baselines.evals.hle_tools_v0 import k2_vllm  # noqa: F401
from inspect_ai._cli.main import main

if __name__ == "__main__":
    main()
