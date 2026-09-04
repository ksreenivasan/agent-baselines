#!/usr/bin/env python3
import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Callable

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


def _is_inspect_eval(command: list[str]) -> bool:
    return any(
        Path(token).name == "inspect" and command[index + 1] == "eval"
        for index, token in enumerate(command[:-1])
    )


def _smoke_or_stop(
    command: list[str],
    *,
    skip: bool,
    smoke_runner: Callable[[list[str]], list[str]] | None = None,
) -> None:
    if not _is_inspect_eval(command):
        return
    if skip:
        warnings.warn(
            "HLE smoke test explicitly skipped; evaluation is proceeding without "
            "tool or endpoint validation.",
            RuntimeWarning,
        )
        return
    try:
        if smoke_runner is None:
            from smoke import run_smoke_test

            smoke_runner = run_smoke_test
        checks = smoke_runner(command)
    except Exception as error:
        detail = str(error) or type(error).__name__
        warnings.warn(
            f"HLE smoke test failed; evaluation stopped: {detail}",
            RuntimeWarning,
        )
        raise SystemExit(2) from error
    print(f"HLE smoke test passed: {', '.join(checks)}", file=sys.stderr)


def main() -> None:
    argv = sys.argv[1:]
    skip_smoke_test = "--skip-smoke-test" in argv
    argv = [argument for argument in argv if argument != "--skip-smoke-test"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", action="append", choices=sorted(KEYS), required=True
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
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
    os.environ.update(env)
    _smoke_or_stop(args.command, skip=skip_smoke_test)
    os.execvpe(args.command[0], args.command, env)


if __name__ == "__main__":
    main()
