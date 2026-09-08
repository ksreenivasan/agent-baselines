from __future__ import annotations

import asyncio
import base64
import ctypes
import fcntl
import json
import os
import socket
import struct
import sys
from typing import Any


class RemoteProcess:
    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.seq = 0
        self.killed = False


REMOTE_PROCESSES: dict[int, RemoteProcess] = {}


async def collect(
    stream: asyncio.StreamReader | None, buffer: bytearray
) -> None:
    if stream is None:
        return
    while chunk := await stream.read(65536):
        buffer.extend(chunk)


def remote_output(record: RemoteProcess) -> tuple[str, str, int]:
    record.seq += 1
    stdout = bytes(record.stdout).decode("utf-8", errors="replace")
    stderr = bytes(record.stderr).decode("utf-8", errors="replace")
    record.stdout.clear()
    record.stderr.clear()
    return stdout, stderr, record.seq


async def remote_request(request: dict[str, Any]) -> dict[str, Any]:
    action = request["action"]
    if action == "remote_start":
        process = await asyncio.create_subprocess_shell(
            request["command"],
            cwd=request["cwd"],
            env=request["env"],
            stdin=asyncio.subprocess.PIPE
            if request.get("stdin_open") or request.get("input") is not None
            else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        record = RemoteProcess(process)
        REMOTE_PROCESSES[process.pid] = record
        asyncio.create_task(collect(process.stdout, record.stdout))
        asyncio.create_task(collect(process.stderr, record.stderr))
        initial = request.get("input")
        if initial is not None and process.stdin is not None:
            process.stdin.write(initial.encode("utf-8"))
            await process.stdin.drain()
        if not request.get("stdin_open") and process.stdin is not None:
            process.stdin.close()
        return {"pid": process.pid}

    record = REMOTE_PROCESSES[int(request["pid"])]
    process = record.process
    if action == "remote_write_stdin":
        if process.stdin is None:
            raise RuntimeError("remote process stdin is closed")
        process.stdin.write(request["data"].encode("utf-8"))
        await process.stdin.drain()
    elif action == "remote_close_stdin":
        if process.stdin is not None and not process.stdin.is_closing():
            process.stdin.close()
    elif action == "remote_kill":
        record.killed = True
        if process.returncode is None:
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                pass
            await process.wait()

    stdout, stderr, seq = remote_output(record)
    result: dict[str, Any] = {"seq": seq, "stdout": stdout, "stderr": stderr}
    if action == "remote_poll":
        if record.killed:
            result.update(state="killed", exit_code=None)
        elif process.returncode is None:
            result.update(state="running", exit_code=None)
        else:
            result.update(state="completed", exit_code=process.returncode)
    return result


class CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class CapData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


def drop_capabilities() -> None:
    header = CapHeader(version=0x20080522, pid=0)
    data = (CapData * 2)()
    if ctypes.CDLL(None, use_errno=True).capset(ctypes.byref(header), data) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


async def read_frame(reader: asyncio.StreamReader) -> dict[str, Any]:
    size = struct.unpack("!Q", await reader.readexactly(8))[0]
    return json.loads((await reader.readexactly(size)).decode("utf-8"))


async def write_frame(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    writer.write(struct.pack("!Q", len(encoded)) + encoded)
    await writer.drain()


async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        request = await read_frame(reader)
        if str(request.get("action", "")).startswith("remote_"):
            await write_frame(writer, await remote_request(request))
            return
        stdin = base64.b64decode(request["input"]) if request.get("input") else None
        try:
            process = await asyncio.create_subprocess_exec(
                *request["cmd"],
                cwd=request["cwd"],
                env=request["env"],
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            communicate = process.communicate(stdin)
            timeout = request.get("timeout")
            try:
                stdout, stderr = await (
                    asyncio.wait_for(communicate, timeout) if timeout else communicate
                )
                returncode = process.returncode
            except asyncio.TimeoutError:
                process.kill()
                stdout, stderr = await process.communicate()
                stderr += b"command timed out"
                returncode = 124
        except FileNotFoundError as error:
            stdout = b""
            stderr = str(error).encode("utf-8", errors="replace")
            returncode = 127
        limit = int(request["output_limit"])
        await write_frame(
            writer,
            {
                "returncode": returncode,
                "stdout": base64.b64encode(stdout[:limit]).decode("ascii"),
                "stderr": base64.b64encode(stderr[:limit]).decode("ascii"),
            },
        )
    except Exception as error:
        await write_frame(
            writer,
            {
                "returncode": 1,
                "stdout": "",
                "stderr": base64.b64encode(
                    f"sandbox executor error: {type(error).__name__}: {error}".encode()
                ).decode("ascii"),
            },
        )
    finally:
        writer.close()
        await writer.wait_closed()


async def main() -> None:
    control = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interface = struct.pack("16sh", b"lo", 0)
    flags = struct.unpack("16sh", fcntl.ioctl(control, 0x8913, interface))[1]
    fcntl.ioctl(control, 0x8914, struct.pack("16sh", b"lo", flags | 0x1))
    control.close()
    drop_capabilities()
    socket_path = sys.argv[1]
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass
    server = await asyncio.start_unix_server(handle, path=socket_path)
    os.chmod(socket_path, 0o600)
    async with server:
        await server.serve_forever()


asyncio.run(main())
