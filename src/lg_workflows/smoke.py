"""Check that every model backend answers.  Run: uv run python -m lg_workflows.smoke"""

import os
import sys
import time

from .cli_agents import run_cli_agent
from .models import brain_llm, local_llm


def _check(name, fn) -> bool:
    t = time.time()
    try:
        out = fn()
        print(f"OK   {name:<8} {time.time() - t:5.1f}s  {str(out).strip()[:60]!r}")
        return True
    except Exception as e:
        print(f"FAIL {name:<8} {time.time() - t:5.1f}s  {type(e).__name__}: {str(e)[:200]}")
        return False


def main() -> None:
    prompt = "只回复：OK"
    results = [
        _check("local", lambda: local_llm().invoke(prompt).text),
        _check("brain", lambda: brain_llm().invoke(prompt).text),
        _check("claude", lambda: run_cli_agent("claude", prompt, cwd=os.getcwd()).result),
        _check("agy", lambda: run_cli_agent("agy", prompt, cwd=os.getcwd()).result),
        _check("cursor", lambda: run_cli_agent("cursor", prompt, cwd=os.getcwd()).result),
    ]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
