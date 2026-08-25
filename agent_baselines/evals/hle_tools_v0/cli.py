import argparse
import json
import os
import subprocess
from pathlib import Path

from agent_baselines.evals.hle_tools_v0.dataset import load_hle_dataset


def preflight() -> int:
    key_dir = Path.home() / "secrets_and_keys"
    hle_path = os.environ.get("HLE_DATA_PATH")
    checks = {
        "fixture_rows": len(load_hle_dataset(fixture=True)),
        "docker_available": subprocess.run(
            ["docker", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
        == 0,
        "hle_data_available": bool(hle_path and Path(hle_path).expanduser().is_file()),
        "exa_key_available": (key_dir / "exa.key").is_file(),
        "tavily_key_available": (key_dir / "tavily.key").is_file(),
        "provider_key_files": {
            name: (key_dir / filename).is_file()
            for name, filename in {
                "openai": "openai.key",
                "anthropic": "anthropic.key",
                "google": "gemini.key",
                "openrouter": "openrouter.key",
            }.items()
        },
    }
    checks["live_smoke_blockers"] = [
        name
        for name, blocked in {
            "gated HLE data file is unavailable": not checks["hle_data_available"],
            "Exa/Tavily credential is unavailable": not (
                checks["exa_key_available"] or checks["tavily_key_available"]
            ),
        }.items()
        if blocked
    ]
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight"])
    args = parser.parse_args()
    raise SystemExit(preflight())


if __name__ == "__main__":
    main()
