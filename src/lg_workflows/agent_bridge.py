"""Expose the CLI agents to LangGraph both as graph nodes and as LLM tools.

Nodes and tools share the same state fields, so a Claude session opened by a tool call
is resumed (with its prompt cache) by a later Claude node, and vice versa.

Policies:
  readonly - plan mode, never touches files
  auto     - may edit files without asking (rely on worktree isolation + verification)
  ask      - pause with interrupt() for approval before editing

Nodes read `config["configurable"]["edit_policy"]` (default "auto"); tools read
`tool_edit_policy` (default "readonly") because their task text is written by a model.
"""

from typing import Annotated, Any, Callable

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict  # same TypedDict as MessagesState, so they can be combined

from .cli_agents import Backend, run_cli_agent

POLICIES = ("readonly", "auto", "ask")


def _last(_old, new):
    return new


class AgentSessions(TypedDict, total=False):
    workdir: str
    # Reducers let parallel branches / tool calls write these in the same step.
    claude_session_id: Annotated[str | None, _last]
    agy_session_id: Annotated[str | None, _last]
    cursor_session_id: Annotated[str | None, _last]
    claude_usage: Annotated[dict, _last]
    agy_usage: Annotated[dict, _last]
    cursor_usage: Annotated[dict, _last]  # last call only; cursor reports usage per call


def resolve_policy(config: RunnableConfig, default_policy: str, key: str = "edit_policy") -> str:
    return (config.get("configurable") or {}).get(key, default_policy)


def invoke_cli_agent(
    backend: Backend,
    task: str,
    state: dict,
    *,
    policy: str,
    schema: dict | None = None,
) -> tuple[Any, dict]:
    """Return (result, state_update)."""
    if policy not in POLICIES:
        raise ValueError(f"edit_policy must be one of {POLICIES}, got {policy!r}")
    if policy == "ask":
        decision = interrupt({"backend": backend, "task": task, "workdir": state["workdir"],
                              "question": "允许该 agent 修改文件吗？回复 yes 批准"})
        if decision != "yes":
            return "用户拒绝了这次修改", {}
    res = run_cli_agent(
        backend,
        task,
        cwd=state["workdir"],
        mode="readonly" if policy == "readonly" else "edit",
        session_id=state.get(f"{backend}_session_id"),
        schema=schema,
    )
    return res.result, {f"{backend}_session_id": res.session_id, f"{backend}_usage": res.session_usage}


def cli_agent_node(
    backend: Backend,
    build_task: Callable[[dict], str],
    *,
    output_key: str,
    default_policy: str = "auto",
    fixed_policy: str | None = None,
    schema: dict | None = None,
):
    """Make a graph node that always runs `backend` on the task built from state.

    `fixed_policy` ignores the run's `edit_policy` (e.g. a review step that must stay read-only).
    """

    def node(state: dict, config: RunnableConfig) -> dict:
        policy = fixed_policy or resolve_policy(config, default_policy)
        result, update = invoke_cli_agent(backend, build_task(state), state, policy=policy, schema=schema)
        return {output_key: result, **update}

    node.__name__ = f"{backend}_node"
    return node


def _tool_result(backend: Backend, task: str, runtime: ToolRuntime, fixed_policy: str | None = None) -> Command:
    # Tool tasks are written by the model, so default to read-only.
    policy = fixed_policy or resolve_policy(runtime.config, "readonly", key="tool_edit_policy")
    result, update = invoke_cli_agent(backend, task, runtime.state, policy=policy)
    return Command(update={
        **update,
        "messages": [ToolMessage(str(result), tool_call_id=runtime.tool_call_id)],
    })


@tool
def ask_claude_code(task: str, runtime: ToolRuntime) -> Command:
    """把需要阅读、分析代码仓库的任务交给 Claude Code（在当前工作目录中执行）。
    task 要写清目标、范围和期望的输出格式。同一次工作流中多次调用会延续同一个会话。"""
    return _tool_result("claude", task, runtime)


@tool
def ask_gemini_agent(task: str, runtime: ToolRuntime) -> Command:
    """把任务交给 Gemini agent（在当前工作目录中执行），适合作为第二意见或交叉核对。
    task 要写清目标、范围和期望的输出格式。"""
    return _tool_result("agy", task, runtime)


@tool
def ask_cursor_agent(task: str, runtime: ToolRuntime) -> Command:
    """把只读的代码阅读、分析或审核任务交给 Cursor agent（Claude Opus 5.5，在当前工作目录中执行），
    适合作为高质量的第二意见。task 要写清目标、范围和期望的输出格式。"""
    return _tool_result("cursor", task, runtime, fixed_policy="readonly")
