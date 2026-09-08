from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import tempfile
import warnings
from pathlib import Path
from typing import Literal, overload

from typing_extensions import override

from inspect_ai.util import (
    ExecResult,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    SandboxEnvironmentLimits,
    sandboxenv,
)
from inspect_ai.util._sandbox._cli import SANDBOX_CLI


_SECRET_NAMES = {
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "EXA_API_KEY",
    "KEENABLE_API_KEY",
    "VLLM_API_KEY",
}


@sandboxenv(name="m2-enroot")
class M2EnrootSandbox(SandboxEnvironment):
    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        return {"default": cls()}

    @override
    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        for environment in environments.values():
            sandbox = environment.as_type(cls)
            await sandbox.close()
            sandbox.directory.cleanup()

    def __init__(self) -> None:
        super().__init__()
        self.container = os.environ["M2_ENROOT_CONTAINER"]
        self.directory = tempfile.TemporaryDirectory(
            prefix="hle-sandbox-", ignore_cleanup_errors=True
        )
        self.root = Path(self.directory.name)
        (self.root / "workspace").mkdir()
        (self.root / "tmp").mkdir()
        (self.root / "empty").mkdir()
        rootfs = Path(os.environ["ENROOT_DATA_PATH"]) / self.container
        (rootfs / "mnt" / "weka").mkdir(parents=True, exist_ok=True)
        (rootfs / "home").mkdir(parents=True, exist_ok=True)
        self._namespace_process: asyncio.subprocess.Process | None = None
        self._namespace_socket = self.root / "tmp" / ".m2-exec.sock"

    @override
    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        if user is not None:
            warnings.warn(
                "The user parameter is ignored by the M2 Enroot sandbox.",
                UserWarning,
            )

        await self._ensure_namespace()
        container_cwd = self._container_path(cwd)
        clean_env = {
            "HOME": "/tmp",
            "PATH": "/opt/inspect_tool_support/bin:/usr/local/bin:/usr/bin:/bin",
            "MPLBACKEND": "Agg",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        clean_env.update(
            {
                key: value
                for key, value in (env or {}).items()
                if key not in _SECRET_NAMES
            }
        )
        if cmd == [SANDBOX_CLI, "exec"] and input is not None:
            raw_input = input.decode() if isinstance(input, bytes) else input
            rpc = json.loads(raw_input)
            method = rpc["method"]
            params = rpc.get("params") or {}
            action_names = {
                "exec_remote_start": "remote_start",
                "exec_remote_poll": "remote_poll",
                "exec_remote_write_stdin": "remote_write_stdin",
                "exec_remote_close_stdin": "remote_close_stdin",
                "exec_remote_kill": "remote_kill",
            }
            if method in action_names:
                remote_env = dict(clean_env)
                remote_env.update(
                    {
                        key: value
                        for key, value in (params.get("env") or {}).items()
                        if key not in _SECRET_NAMES
                    }
                )
                request = {
                    "action": action_names[method],
                    "pid": params.get("pid"),
                    "command": params.get("command"),
                    "cwd": self._container_path(params.get("cwd")),
                    "env": remote_env,
                    "input": params.get("input"),
                    "stdin_open": params.get("stdin_open", False),
                    "data": params.get("data"),
                }
                result = await self._request(request, timeout)
                stdout = json.dumps(
                    {"jsonrpc": "2.0", "id": rpc.get("id"), "result": result}
                )
                return ExecResult(success=True, returncode=0, stdout=stdout, stderr="")
        request = {
            "cmd": cmd,
            "cwd": container_cwd,
            "env": clean_env,
            "input": base64.b64encode(
                input.encode() if isinstance(input, str) else input
            ).decode("ascii")
            if input is not None
            else None,
            "timeout": timeout,
            "output_limit": SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE,
        }
        response = await self._request(request, timeout)
        stdout = base64.b64decode(response["stdout"]).decode("utf-8", errors="replace")
        stderr = base64.b64decode(response["stderr"]).decode("utf-8", errors="replace")
        return ExecResult(
            success=response["returncode"] == 0,
            returncode=response["returncode"],
            stdout=stdout,
            stderr=stderr,
        )

    async def close(self) -> None:
        process = self._namespace_process
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def _ensure_namespace(self) -> None:
        process = self._namespace_process
        if process is not None and process.returncode is None:
            return
        uid = str(os.getuid())
        gid = str(os.getgid())
        server_source = Path(__file__).with_name("m2_enroot_exec_server.py")
        server_target = self.root / "tmp" / ".m2-exec-server.py"
        server_target.write_bytes(server_source.read_bytes())
        self._namespace_process = await asyncio.create_subprocess_exec(
            *self._enroot_prefix(),
            "unshare",
            "-Un",
            f"--map-user={uid}",
            f"--map-group={gid}",
            "--keep-caps",
            "env",
            "-i",
            "HOME=/tmp",
            "PATH=/usr/local/bin:/usr/bin:/bin",
            "python",
            "/tmp/.m2-exec-server.py",
            "/tmp/.m2-exec.sock",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        for _ in range(200):
            if self._namespace_socket.exists():
                return
            if self._namespace_process.returncode is not None:
                stderr = await self._namespace_process.stderr.read()
                raise RuntimeError(
                    f"sandbox namespace holder exited: {stderr.decode(errors='replace')}"
                )
            await asyncio.sleep(0.05)
        await self.close()
        raise TimeoutError("sandbox executor did not become ready")

    async def _request(
        self, request: dict[str, object], timeout: int | None
    ) -> dict[str, object]:
        async def exchange() -> dict[str, object]:
            reader, writer = await asyncio.open_unix_connection(
                str(self._namespace_socket)
            )
            try:
                encoded = json.dumps(request).encode("utf-8")
                writer.write(struct.pack("!Q", len(encoded)) + encoded)
                await writer.drain()
                size = struct.unpack("!Q", await reader.readexactly(8))[0]
                return json.loads((await reader.readexactly(size)).decode("utf-8"))
            finally:
                writer.close()
                await writer.wait_closed()

        if timeout is None:
            return await exchange()
        return await asyncio.wait_for(exchange(), timeout=timeout + 10)

    def _enroot_prefix(self) -> list[str]:
        return [
            "enroot",
            "start",
            "-m",
            f"{self.root / 'workspace'}:/workspace:bind,rw",
            "-m",
            f"{self.root / 'tmp'}:/tmp:bind,rw",
            "-m",
            f"{self.root / 'empty'}:/mnt/weka:bind,ro",
            "-m",
            f"{self.root / 'empty'}:/home:bind,ro",
            self.container,
        ]

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        if file == SANDBOX_CLI:
            payload = contents.encode() if isinstance(contents, str) else contents
            result = await self.exec(
                [
                    "sh",
                    "-c",
                    'mkdir -p -- "$(dirname -- "$1")" && base64 -d > "$1"',
                    "m2-write-file",
                    file,
                ],
                input=base64.b64encode(payload).decode("ascii"),
                timeout=600,
            )
            if not result.success:
                raise RuntimeError(f"failed to inject sandbox tools: {result.stderr}")
            return
        path = self._host_path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(contents, str):
            path.write_text(contents, encoding="utf-8", newline="")
        else:
            path.write_bytes(contents)

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        path = self._host_path(file)
        if path.stat().st_size > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
            raise ValueError(f"sandbox file exceeds read limit: {file}")
        if text:
            with path.open("r", encoding="utf-8", newline="") as handle:
                return handle.read()
        return path.read_bytes()

    def default_polling_interval(self) -> float:
        return 0.2

    def _container_path(self, path: str | None) -> str:
        if path is None:
            return "/workspace"
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate.as_posix()
        return (Path("/workspace") / candidate).as_posix()

    def _host_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            if candidate == Path("/workspace"):
                return self.root / "workspace"
            try:
                relative = candidate.relative_to("/workspace")
            except ValueError as exc:
                raise PermissionError(
                    f"sandbox file access is restricted to /workspace: {path}"
                ) from exc
            return self.root / "workspace" / relative
        return self.root / "workspace" / candidate
