#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

KEYS = {
    "openai": ("openai.key", "OPENAI_API_KEY"),
    "anthropic": ("anthropic.key", "ANTHROPIC_API_KEY"),
    "google": ("gemini.key", "GOOGLE_API_KEY"),
    "openrouter": ("openrouter.key", "OPENROUTER_API_KEY"),
    "exa": ("exa.key", "EXA_API_KEY"),
}


def _read_key(path: Path, variable: str) -> str:
    text = path.read_text().strip()
    if text.startswith("export "):
        text = text[7:].strip()
    if "=" in text:
        name, text = text.split("=", 1)
        if name.strip() != variable:
            raise RuntimeError(f"{path.name} defines an unexpected variable")
    value = text.strip().strip("'\"")
    if not value:
        raise RuntimeError(f"{path.name} is empty")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", action="append", choices=sorted(KEYS), required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a command is required")

    key_dir = Path.home() / "secrets_and_keys"
    env = os.environ.copy()
    for provider in args.provider:
        filename, variable = KEYS[provider]
        path = key_dir / filename
        if not path.is_file():
            raise RuntimeError(f"authorized key file is unavailable: {path}")
        env[variable] = _read_key(path, variable)
    os.execvpe(args.command[0], args.command, env)


if __name__ == "__main__":
    main()
