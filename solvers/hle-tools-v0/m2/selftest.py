from __future__ import annotations

from pathlib import Path

import anyio
from inspect_ai.tool._sandbox_tools_utils.sandbox import sandbox_with_injected_tools
from inspect_ai.util._sandbox._cli import SANDBOX_CLI

from agent_baselines.evals.hle_tools_v0.m2_enroot_sandbox import M2EnrootSandbox


async def main() -> None:
    sandbox = M2EnrootSandbox()
    try:
        await sandbox.write_file("probe.py", "print('sandbox-write-ok')\n")
        result = await sandbox.exec(
            [
                "python",
                "-c",
                (
                    "import os,socket; "
                    "print('uid', os.getuid()); "
                    "print('openai_secret', bool(os.environ.get('OPENAI_API_KEY'))); "
                    "print('keenable_secret', bool(os.environ.get('KEENABLE_API_KEY'))); "
                    "print('cap_eff', next(x.split()[1] for x in open('/proc/self/status') if x.startswith('CapEff:'))); "
                    "s=socket.socket(); s.settimeout(1); "
                    "print('network', s.connect_ex(('1.1.1.1',443))); "
                    "exec(open('probe.py').read())"
                ),
            ],
            env={"OPENAI_API_KEY": "sentinel", "KEENABLE_API_KEY": "sentinel"},
            timeout=30,
        )
        print(result.stdout, end="")
        if not result.success:
            raise RuntimeError(result.stderr)
        lines = set(result.stdout.splitlines())
        if (
            "openai_secret False" not in lines
            or "keenable_secret False" not in lines
            or "network 101" not in lines
            or "cap_eff 0000000000000000" not in lines
        ):
            raise RuntimeError("sandbox isolation checks failed")
        if await sandbox.read_file("probe.py") != "print('sandbox-write-ok')\n":
            raise RuntimeError("sandbox file round trip failed")
        injected = await sandbox_with_injected_tools(sandbox=sandbox)
        tool_result = await injected.exec([SANDBOX_CLI, "--help"], timeout=30)
        if not tool_result.success:
            raise RuntimeError(f"sandbox tool injection failed: {tool_result.stderr}")
        print("sandbox-tool-injection-ok")
        server = await sandbox.exec(
            [
                "sh",
                "-c",
                (
                    "python -m http.server 18473 --bind 127.0.0.1 "
                    ">/tmp/m2-http.log 2>&1 & echo $! >/tmp/m2-http.pid"
                ),
            ],
            timeout=30,
        )
        if not server.success:
            raise RuntimeError(f"sandbox namespace server failed: {server.stderr}")
        client = await sandbox.exec(
            [
                "python",
                "-c",
                (
                    "import socket,time; "
                    "exec(\"for attempt in range(50):\\n"
                    " try:\\n"
                    "  connection=socket.create_connection(('127.0.0.1',18473),timeout=1); connection.close(); break\\n"
                    " except OSError:\\n"
                    "  time.sleep(0.1)\\n"
                    "else:\\n"
                    " raise RuntimeError('loopback server did not become ready')\")"
                ),
            ],
            timeout=30,
        )
        if not client.success:
            raise RuntimeError(f"sandbox namespace persistence failed: {client.stderr}")
        await sandbox.exec(
            ["sh", "-c", "kill \"$(cat /tmp/m2-http.pid)\""], timeout=30
        )
        print("sandbox-namespace-persistence-ok")
        host_paths = [Path.home(), Path.home() / "secrets_and_keys"]
        for host_path in host_paths:
            visible = await sandbox.exec(["test", "-e", str(host_path)], timeout=30)
            if visible.success:
                raise RuntimeError("host path unexpectedly visible in sandbox")
        masked = await sandbox.exec(
            [
                "python",
                "-c",
                (
                    "from pathlib import Path; "
                    "assert not any(Path('/mnt/weka').iterdir()); "
                    "assert not any(Path('/home').iterdir())"
                ),
            ],
            timeout=30,
        )
        if not masked.success:
            raise RuntimeError("host path masks are not empty")
        print("sandbox-host-paths-hidden")
    finally:
        await sandbox.close()
        sandbox.directory.cleanup()


anyio.run(main)
