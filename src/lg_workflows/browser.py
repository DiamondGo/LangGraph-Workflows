"""Headless Chrome on persistent profiles, for workflows that read websites without an API.

Each site gets its own profile directory: Chrome locks a profile while it is open, and separate
profiles keep one site's cookies (e.g. a passed Cloudflare check) from leaking into another.
The same profile can be handed to a CLI agent through the Playwright MCP server (`mcp_config`).
"""

import json
import shutil
import subprocess
import threading
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Iterator

from playwright.sync_api import Page, sync_playwright

# Chrome encrypts cookies with the desktop keyring by default, which systemd jobs cannot reach.
CHROME_ARGS = ["--password-store=basic", "--no-first-run", "--no-default-browser-check"]
# Headless Chrome announces itself in the user agent and navigator.webdriver; Cloudflare blocks both.
STEALTH_ARGS = ["--disable-blink-features=AutomationControlled"]
VIEWPORT = {"width": 1280, "height": 900}
PLAYWRIGHT_MCP = "@playwright/mcp@0.0.83"

# One browser at a time: graph branches run in threads and each Chrome takes ~500 MB.
_LOCK = threading.Lock()


@cache
def user_agent() -> str:
    exe = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
    version = subprocess.run([exe, "--version"], capture_output=True, text=True, check=True).stdout.split()[-1]
    return f"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{version} Safari/537.36"


@contextmanager
def chrome_page(profile: Path, *, locale: str | None = None) -> Iterator[Page]:
    profile.mkdir(parents=True, exist_ok=True)
    with _LOCK, sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            str(profile), channel="chrome", headless=True, args=CHROME_ARGS + STEALTH_ARGS,
            ignore_default_args=["--enable-automation"], user_agent=user_agent(), locale=locale,
            viewport=VIEWPORT)
        try:
            yield ctx.pages[0] if ctx.pages else ctx.new_page()
        finally:
            ctx.close()


def mcp_config(profile: Path, path: Path) -> Path:
    """Write a Claude Code MCP config exposing a headless Chrome on `profile` as server "browser".

    `--caps vision` adds coordinate mouse tools, so the agent can act on what a screenshot shows when the
    page's accessibility tree does not expose the control it needs.
    """
    profile.mkdir(parents=True, exist_ok=True)
    args = ["-y", PLAYWRIGHT_MCP, "--headless", "--caps", "vision", "--browser", "chrome",
            "--user-data-dir", str(profile),
            "--user-agent", user_agent(), "--viewport-size", f"{VIEWPORT['width']},{VIEWPORT['height']}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mcpServers": {"browser": {"command": "npx", "args": args}}}, indent=1))
    return path
