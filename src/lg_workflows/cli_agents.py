"""Run subscription-backed coding agents (`claude -p`, `agy -p`, Cursor `agent -p`) as subprocesses.

`claude` and `agy` report usage cumulatively per session; `cursor` reports it per call.
"""

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv()

Backend = Literal["claude", "agy", "cursor"]
# browse: only the tools of the MCP servers in `mcp_config` (e.g. a headless browser); no shell, files or web fetch.
Mode = Literal["readonly", "edit", "browse"]

# Subscription quotas are small and the CLIs are heavy; never run two of the same at once.
_LOCKS: dict[str, threading.Semaphore] = {b: threading.Semaphore(1) for b in ("claude", "agy", "cursor")}

CLAUDE_EDIT_ALLOWED_TOOLS = "Read,Edit,Write,Glob,Grep,Bash(git diff *),Bash(git status *),Bash(git log *)"
CLAUDE_EDIT_DISALLOWED_TOOLS = "Bash(git push *),Bash(rm *),Bash(sudo *)"
CLAUDE_BROWSE_DISALLOWED_TOOLS = "Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch"


class CLIAgentError(RuntimeError):
    pass


@dataclass
class CLIAgentResult:
    backend: Backend
    result: Any
    session_id: str
    session_usage: dict = field(default_factory=dict)


def _claude_cmd(prompt: str, mode: Mode, session_id: str | None, fork: bool,
                schema: dict | None, add_dirs: tuple[str, ...], mcp_config: str | None = None) -> list[str]:
    cmd = ["claude", "-p", prompt,
           "--model", os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5"),
           "--output-format", "json"]
    if mode == "readonly":
        cmd += ["--permission-mode", "plan"]
    elif mode == "browse":
        if not mcp_config:
            raise ValueError("browse mode needs mcp_config")
        servers = json.loads(Path(mcp_config).read_text())["mcpServers"]
        cmd += ["--permission-mode", "default", "--strict-mcp-config", "--mcp-config", mcp_config,
                "--allowedTools", ",".join(f"mcp__{name}" for name in servers),
                "--disallowedTools", CLAUDE_BROWSE_DISALLOWED_TOOLS]
    else:
        cmd += ["--permission-mode", "acceptEdits",
                "--allowedTools", CLAUDE_EDIT_ALLOWED_TOOLS,
                "--disallowedTools", CLAUDE_EDIT_DISALLOWED_TOOLS]
    if session_id:
        cmd += ["--resume", session_id] + (["--fork-session"] if fork else [])
    if schema:
        cmd += ["--json-schema", json.dumps(schema)]
    for d in add_dirs:
        cmd += ["--add-dir", d]
    return cmd


def _agy_cmd(prompt: str, mode: Mode, session_id: str | None, fork: bool,
             schema: dict | None, add_dirs: tuple[str, ...], mcp_config: str | None = None) -> list[str]:
    if fork:
        raise ValueError("agy does not support forking a conversation")
    if mode == "browse" or mcp_config:
        raise ValueError("agy backend does not support browse mode")
    cmd = ["agy", "-p", prompt,
           "--model", os.environ.get("AGY_MODEL", "gemini-3.8-flash-medium"),
           "--output-format", "json",
           "--mode", "plan" if mode == "readonly" else "accept-edits"]
    if session_id:
        cmd += ["--conversation", session_id]
    if schema:
        cmd += ["--json-schema", json.dumps(schema)]
    for d in add_dirs:
        cmd += ["--add-dir", d]
    return cmd


def _cursor_cmd(prompt: str, mode: Mode, session_id: str | None, fork: bool,
                schema: dict | None, add_dirs: tuple[str, ...], mcp_config: str | None = None) -> list[str]:
    # Without --mode, `agent -p` gets every tool including shell and has no allowlist flag.
    if mode != "readonly":
        raise ValueError("cursor backend is read-only")
    if fork or schema or add_dirs or mcp_config:
        raise ValueError("cursor backend does not support fork, schema, add_dirs or mcp_config")
    cmd = ["agent", "-p", prompt,
           "--model", os.environ.get("CURSOR_MODEL", "claude-opus-5-5-medium"),
           "--output-format", "json",
           "--mode", "ask",
           "--trust"]
    if session_id:
        cmd += ["--resume", session_id]
    return cmd


def _parse_claude(stdout: str) -> CLIAgentResult:
    data = json.loads(stdout)
    if data.get("is_error"):
        raise CLIAgentError(f"claude: {str(data.get('result'))[:1500]}")
    return CLIAgentResult(
        backend="claude",
        result=data.get("structured_output") or data.get("result"),
        session_id=data["session_id"],
        session_usage={
            "cost_usd": data.get("total_cost_usd"),
            "num_turns": data.get("num_turns"),
            **(data.get("usage") or {}),
        },
    )


def _parse_agy(stdout: str) -> CLIAgentResult:
    data = json.loads(stdout.strip().splitlines()[-1])
    if data.get("status") != "SUCCESS":
        raise CLIAgentError(f"agy: status={data.get('status')} {str(data.get('response'))[:1500]}")
    return CLIAgentResult(
        backend="agy",
        result=data.get("structured_output") or (data.get("response") or "").strip(),
        session_id=data["conversation_id"],
        session_usage={
            "duration_seconds": data.get("duration_seconds"),
            "num_turns": data.get("num_turns"),
            **(data.get("usage") or {}),
        },
    )


def _parse_cursor(stdout: str) -> CLIAgentResult:
    data = json.loads(stdout.strip().splitlines()[-1])
    if data.get("is_error"):
        raise CLIAgentError(f"cursor: {str(data.get('result'))[:1500]}")
    return CLIAgentResult(
        backend="cursor",
        result=(data.get("result") or "").strip(),
        session_id=data["session_id"],
        session_usage={"duration_ms": data.get("duration_ms"), **(data.get("usage") or {})},
    )


_BUILDERS = {"claude": _claude_cmd, "agy": _agy_cmd, "cursor": _cursor_cmd}
_PARSERS = {"claude": _parse_claude, "agy": _parse_agy, "cursor": _parse_cursor}


def run_cli_agent(
    backend: Backend,
    prompt: str,
    cwd: str,
    *,
    mode: Mode = "readonly",
    session_id: str | None = None,
    fork: bool = False,
    schema: dict | None = None,
    add_dirs: tuple[str, ...] = (),
    mcp_config: str | None = None,
    timeout: int = 900,
) -> CLIAgentResult:
    """Run one prompt; pass `session_id` to continue that session (and reuse its cache)."""
    cmd = _BUILDERS[backend](prompt, mode, session_id, fork, schema, add_dirs, mcp_config)
    with _LOCKS[backend]:
        try:
            proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise CLIAgentError(f"{backend}: timed out after {timeout}s") from e
    try:
        return _PARSERS[backend](proc.stdout)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        tail = (proc.stderr or proc.stdout)[-1500:]
        raise CLIAgentError(f"{backend}: exit {proc.returncode}, unparseable output: {tail}") from e
