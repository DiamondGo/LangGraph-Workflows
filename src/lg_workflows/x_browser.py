"""Post to X through a logged-in Chrome profile instead of the paid API.

This automates the website, which X's automation rules forbid; the account owner accepts the risk.
Verification challenges are never bypassed: they stop posting and surface as XPostError.

One-time login in a normal, visible Chrome window (close the window when done):
    uv run python -m lg_workflows.x_browser login
Check the saved session headlessly:
    uv run python -m lg_workflows.x_browser check
"""

import os
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

PROFILE_DIR = Path(os.environ.get("X_CHROME_PROFILE",
                                  Path(__file__).resolve().parents[2] / "data" / "x_chrome_profile"))
# Chrome encrypts cookies with the desktop keyring by default, which systemd jobs cannot reach.
# Login and automation must use the same setting or the session is unreadable.
CHROME_ARGS = ["--password-store=basic", "--no-first-run", "--no-default-browser-check"]

MAX_WEIGHTED_LENGTH = 280
URL_WEIGHT = 23
_LOGGED_IN = '[data-testid="SideNav_NewTweet_Button"], [data-testid="AppTabBar_Profile_Link"]'
_TEXTAREA = '[data-testid="tweetTextarea_0"]'
_POST_BUTTON = '[data-testid="tweetButton"]'
_TOAST = '[data-testid="toast"]'


class XPostError(RuntimeError):
    pass


class XLoginRequired(XPostError):
    pass


class XPostUnconfirmed(XPostError):
    """Post was clicked but X gave no confirmation: it may or may not be live, so never retry blindly."""


def _char_weight(ch: str) -> int:
    # twitter-text v3: Latin, punctuation and general symbols count 1; CJK, emoji and the rest count 2.
    cp = ord(ch)
    light = cp <= 0x10FF or 0x2000 <= cp <= 0x200D or 0x2010 <= cp <= 0x201F or 0x2032 <= cp <= 0x2037
    return 1 if light else 2


def weighted_length(text: str) -> int:
    total = 0
    for token in text.split(" "):
        for part in token.split("\n"):
            if part.startswith(("http://", "https://")):
                total += URL_WEIGHT
            else:
                total += sum(_char_weight(c) for c in unicodedata.normalize("NFC", part))
    return total + text.count(" ") + text.count("\n")


def truncate_to_weight(text: str, budget: int, ellipsis: str = "…") -> str:
    if weighted_length(text) <= budget:
        return text
    out, used = [], weighted_length(ellipsis)
    for ch in text:
        w = _char_weight(ch)
        if used + w > budget:
            break
        out.append(ch)
        used += w
    return "".join(out).rstrip() + ellipsis


class XSession:
    """A headless Chrome session on the saved profile. Use as a context manager."""

    def __init__(self, headless: bool = True):
        self.headless = headless

    def __enter__(self) -> "XSession":
        if not PROFILE_DIR.exists():
            raise XLoginRequired(f"no Chrome profile at {PROFILE_DIR}; run `python -m lg_workflows.x_browser login`")
        self._pw = sync_playwright().start()
        try:
            self._ctx = self._pw.chromium.launch_persistent_context(
                str(PROFILE_DIR), channel="chrome", headless=self.headless, args=CHROME_ARGS,
                viewport={"width": 1280, "height": 900})
        except Exception:
            self._pw.stop()
            raise
        self.page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        return self

    def __exit__(self, *exc) -> None:
        self._ctx.close()
        self._pw.stop()

    def _check_blocked(self) -> None:
        url = self.page.url
        if "/login" in url or "/i/flow/login" in url:
            raise XLoginRequired(f"logged out (at {url}); run `python -m lg_workflows.x_browser login`")
        if "/account/access" in url or "challenge" in url:
            raise XPostError(f"X asked for verification (at {url}); log in manually to resolve it")

    def handle(self) -> str:
        """Return the logged-in @handle, or raise XLoginRequired."""
        self.page.goto("https://x.com/home", wait_until="domcontentloaded")
        try:
            self.page.wait_for_selector(_LOGGED_IN, timeout=30_000)
        except PlaywrightTimeout:
            self._check_blocked()
            raise XLoginRequired(f"no logged-in session (at {self.page.url})") from None
        self._check_blocked()
        href = self.page.get_attribute('[data-testid="AppTabBar_Profile_Link"]', "href") or ""
        return "@" + href.strip("/")

    def post(self, text: str, *, dry_run: bool = False, screenshot: Path | None = None) -> str | None:
        """Post `text`. Returns the post URL when X shows it, else None. dry_run fills the box but never posts."""
        if weighted_length(text) > MAX_WEIGHTED_LENGTH:
            raise ValueError(f"post is {weighted_length(text)} weighted chars (max {MAX_WEIGHTED_LENGTH})")
        self.page.goto("https://x.com/compose/post", wait_until="domcontentloaded")
        try:
            box = self.page.wait_for_selector(_TEXTAREA, timeout=30_000)
        except PlaywrightTimeout:
            self._check_blocked()
            raise XPostError(f"compose box did not appear (at {self.page.url})") from None
        box.click()
        self.page.keyboard.type(text, delay=15)
        button = self.page.locator(_POST_BUTTON)
        button.wait_for(state="visible", timeout=10_000)
        if not button.is_enabled():
            raise XPostError("post button is disabled (text rejected?)")
        if screenshot:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(screenshot))
        if dry_run:
            return None
        button.click()
        # The home timeline has its own compose box with the same test id, so the box vanishing
        # is not a signal. X confirms with a toast linking to the post, or explains a rejection in one.
        try:
            toast = self.page.wait_for_selector(_TOAST, timeout=30_000)
        except PlaywrightTimeout:
            self._check_blocked()
            raise XPostUnconfirmed("no confirmation after clicking post; check the profile") from None
        link = toast.query_selector('a[href*="/status/"]')
        if link is None:
            raise XPostError(f"X rejected the post: {toast.inner_text().strip()[:200]}")
        href = link.get_attribute("href") or ""
        return href if href.startswith("http") else f"https://x.com{href}"


def login() -> None:
    chrome = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
    if not chrome:
        sys.exit("google-chrome not found")
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Log in to X in the Chrome window, then close it. Profile: {PROFILE_DIR}")
    subprocess.run([chrome, f"--user-data-dir={PROFILE_DIR}", *CHROME_ARGS, "https://x.com/login"], check=False)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "login":
        login()
    elif cmd == "check":
        with XSession() as s:
            print(f"logged in as {s.handle()}")
    else:
        sys.exit("usage: python -m lg_workflows.x_browser [login|check]")


if __name__ == "__main__":
    main()
