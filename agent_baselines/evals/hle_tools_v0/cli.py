import argparse
import json
import os
import subprocess
from pathlib import Path

from agent_baselines.evals.hle_tools_v0.dataset import load_hle_dataset
from agent_baselines.evals.hle_tools_v0.manifest import build_manifests


def preflight() -> int:
    key_dir = Path.home() / "secrets_and_keys"
    hle_path = os.environ.get("HLE_DATA_PATH")
    hle_revision = os.environ.get("HLE_DATASET_REVISION")
    hle_verified_path = os.environ.get("HLE_VERIFIED_DATA_PATH")
    checks = {
        "fixture_rows": len(load_hle_dataset(fixture=True)),
        "docker_available": subprocess.run(
            ["docker", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
        == 0,
        "hle_standard_data_available": bool(
            hle_path and Path(hle_path).expanduser().exists()
        ),
        "hle_standard_revision_pinned": bool(
            hle_revision
            and len(hle_revision) == 40
            and all(character in "0123456789abcdef" for character in hle_revision)
        ),
        "hle_verified_data_available": bool(
            hle_verified_path and Path(hle_verified_path).expanduser().exists()
        ),
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
            "standard HLE snapshot is unavailable": not checks[
                "hle_standard_data_available"
            ],
            "standard HLE snapshot revision is not pinned": not checks[
                "hle_standard_revision_pinned"
            ],
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
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    manifest_parser = subparsers.add_parser("build-manifests")
    manifest_parser.add_argument("--data-path", required=True)
    manifest_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        raise SystemExit(preflight())
    audit = build_manifests(args.data_path, args.output_dir)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
