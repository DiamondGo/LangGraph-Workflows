import json
import subprocess

import pytest

from lg_workflows import cli_agents
from lg_workflows.agent_bridge import invoke_cli_agent
from lg_workflows.cli_agents import CLIAgentError, run_cli_agent

CLAUDE_OK = json.dumps({"is_error": False, "result": "hi", "session_id": "s-1",
                        "total_cost_usd": 0.04, "num_turns": 1, "usage": {"cache_read_input_tokens": 10}})
AGY_OK = json.dumps({"conversation_id": "c-1", "status": "SUCCESS", "response": "hi\n",
                     "duration_seconds": 2.0, "num_turns": 1, "usage": {"input_tokens": 7000}})


@pytest.fixture
def fake_run(monkeypatch):
    calls = []

    def install(stdout, returncode=0, stderr=""):
        def run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)
        monkeypatch.setattr(cli_agents.subprocess, "run", run)
        return calls

    return install


def test_claude_readonly_resume(fake_run):
    calls = fake_run(CLAUDE_OK)
    res = run_cli_agent("claude", "q", cwd="/tmp", session_id="s-0")
    cmd = calls[0]
    assert cmd[cmd.index("--permission-mode") + 1] == "plan"
    assert cmd[cmd.index("--resume") + 1] == "s-0"
    assert "--allowedTools" not in cmd
    assert (res.result, res.session_id, res.session_usage["cost_usd"]) == ("hi", "s-1", 0.04)


def test_claude_edit_uses_allowlist(fake_run):
    calls = fake_run(CLAUDE_OK)
    run_cli_agent("claude", "q", cwd="/tmp", mode="edit")
    cmd = calls[0]
    assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"
    assert "--allowedTools" in cmd and "--disallowedTools" in cmd


def test_claude_structured_output_preferred(fake_run):
    fake_run(json.dumps({"is_error": False, "result": "{}", "structured_output": {"a": 1}, "session_id": "s"}))
    assert run_cli_agent("claude", "q", cwd="/tmp", schema={"type": "object"}).result == {"a": 1}


def test_claude_error_raises(fake_run):
    fake_run(json.dumps({"is_error": True, "result": "quota exceeded", "session_id": "s"}))
    with pytest.raises(CLIAgentError, match="quota exceeded"):
        run_cli_agent("claude", "q", cwd="/tmp")


def test_agy_parse_and_flags(fake_run):
    calls = fake_run("some log line\n" + AGY_OK)
    res = run_cli_agent("agy", "q", cwd="/tmp", session_id="c-0")
    cmd = calls[0]
    assert cmd[cmd.index("--mode") + 1] == "plan"
    assert cmd[cmd.index("--conversation") + 1] == "c-0"
    assert (res.result, res.session_id) == ("hi", "c-1")


def test_agy_failure_status(fake_run):
    fake_run(json.dumps({"conversation_id": "c", "status": "ERROR", "response": "boom"}))
    with pytest.raises(CLIAgentError, match="boom"):
        run_cli_agent("agy", "q", cwd="/tmp")


def test_unparseable_output(fake_run):
    fake_run("not json", returncode=1, stderr="auth required")
    with pytest.raises(CLIAgentError, match="auth required"):
        run_cli_agent("claude", "q", cwd="/tmp")


def test_agy_fork_rejected():
    with pytest.raises(ValueError):
        run_cli_agent("agy", "q", cwd="/tmp", session_id="c", fork=True)


CURSOR_OK = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "hi",
                        "session_id": "k-1", "duration_ms": 1500,
                        "usage": {"inputTokens": 4, "cacheReadTokens": 9883}})


def test_cursor_readonly_flags_and_parse(fake_run):
    calls = fake_run(CURSOR_OK)
    res = run_cli_agent("cursor", "q", cwd="/tmp", session_id="k-0")
    cmd = calls[0]
    assert cmd[0] == "agent"
    assert cmd[cmd.index("--mode") + 1] == "ask"
    assert "--trust" in cmd and "--force" not in cmd and "--yolo" not in cmd
    assert cmd[cmd.index("--resume") + 1] == "k-0"
    assert (res.result, res.session_id, res.session_usage["cacheReadTokens"]) == ("hi", "k-1", 9883)


def test_cursor_rejects_edit():
    with pytest.raises(ValueError, match="read-only"):
        run_cli_agent("cursor", "q", cwd="/tmp", mode="edit")


def test_invoke_updates_backend_session_fields(fake_run):
    fake_run(AGY_OK)
    result, update = invoke_cli_agent("agy", "q", {"workdir": "/tmp"}, policy="readonly")
    assert result == "hi"
    assert update == {"agy_session_id": "c-1", "agy_usage": update["agy_usage"]}


def test_invoke_rejects_unknown_policy():
    with pytest.raises(ValueError):
        invoke_cli_agent("claude", "q", {"workdir": "/tmp"}, policy="yolo")
