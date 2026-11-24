"""Post to X through a logged-in Chrome profile instead of the paid API.

This automates the website, which X's automation rules forbid; the account owner accepts the risk.
Verification challenges are never bypassed: they stop posting and surface as XPostError.

One-time login in a normal, visible Chrome window (close the window when done):
    uv run python -m lg_workflows.x_browser login
Check the saved session headlessly:
    uv run python -m lg_workflows.x_browser check
"""

import fcntl
import os
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

from lg_workflows.browser import STEALTH_ARGS, user_agent

PROFILE_DIR = Path(os.environ.get("X_CHROME_PROFILE",
                                  Path(__file__).resolve().parents[2] / "data" / "x_chrome_profile"))
FAILURE_DIR = PROFILE_DIR.parent / "x_screenshots"
# Chrome encrypts cookies with the desktop keyring by default, which systemd jobs cannot reach.
# Login and automation must use the same setting or the session is unreadable.
CHROME_ARGS = ["--password-store=basic", "--no-first-run", "--no-default-browser-check"]

MAX_WEIGHTED_LENGTH = 280
URL_WEIGHT = 23
_LOGGED_IN = '[data-testid="SideNav_NewTweet_Button"], [data-testid="AppTabBar_Profile_Link"]'
_TEXTAREA = '[data-testid="tweetTextarea_0"]'
_POST_BUTTON = '[data-testid="tweetButton"]'
_TOAST = '[data-testid="toast"]'
_VIEWPORT = {"width": 1280, "height": 900}
_TYPE_DELAY_MS = (90, 260)
_TYPE_PAUSE_MS = (450, 1400)
# News posting and reply jobs share one Chrome profile, and Chrome cannot open it twice at once.
LOCK_WAIT_SECONDS = 20 * 60
_STATUS_PATH = re.compile(r"^/(\w+)/status/(\d+)$")
_TWEET = 'article[data-testid="tweet"]'


class XPostError(RuntimeError):
    pass


class XLoginRequired(XPostError):
    pass


class XProfileBusy(XPostError):
    """Another job is using the Chrome profile; nothing was attempted, so it is safe to try again later."""


class XTargetMissing(XPostError):
    """The tweet to read or reply to did not show up (deleted, protected, or a transient load failure)."""


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
        self._mouse_position = (0.0, 0.0)
        self._lock = None

    def _acquire_profile_lock(self) -> None:
        path = PROFILE_DIR.with_name(PROFILE_DIR.name + ".lock")
        lock = open(path, "w")
        deadline = time.monotonic() + LOCK_WAIT_SECONDS
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._lock = lock
                return
            except BlockingIOError:
                if time.monotonic() > deadline:
                    lock.close()
                    raise XProfileBusy(f"Chrome profile still in use by another job after {LOCK_WAIT_SECONDS // 60} min") from None
                time.sleep(5)

    def _release_profile_lock(self) -> None:
        if self._lock is not None:
            self._lock.close()  # closing the file drops the flock
            self._lock = None

    def __enter__(self) -> "XSession":
        if not PROFILE_DIR.exists():
            raise XLoginRequired(f"no Chrome profile at {PROFILE_DIR}; run `python -m lg_workflows.x_browser login`")
        self._acquire_profile_lock()
        try:
            self._pw = sync_playwright().start()
            try:
                self._ctx = self._pw.chromium.launch_persistent_context(
                    str(PROFILE_DIR), channel="chrome", headless=self.headless,
                    args=CHROME_ARGS + STEALTH_ARGS, ignore_default_args=["--enable-automation"],
                    user_agent=user_agent(), viewport=_VIEWPORT)
            except Exception:
                self._pw.stop()
                raise
        except Exception:
            self._release_profile_lock()
            raise
        self.page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        self._mouse_position = (0.0, 0.0)
        return self

    def __exit__(self, *exc) -> None:
        try:
            self._ctx.close()
            self._pw.stop()
        finally:
            self._release_profile_lock()

    def _move_mouse(self, x: float, y: float) -> None:
        viewport = self.page.viewport_size or _VIEWPORT
        start_x, start_y = self._mouse_position
        # Randomized Bezier control points keep the path curved and inside the viewport.
        controls = [
            (min(max(start_x + (x - start_x) * fraction + random.uniform(-90, 90), 0), viewport["width"] - 1),
             min(max(start_y + (y - start_y) * fraction + random.uniform(-70, 70), 0), viewport["height"] - 1))
            for fraction in (0.3, 0.7)
        ]
        steps = random.randint(18, 35)
        for step in range(1, steps + 1):
            t = step / steps
            u = 1 - t
            px = u ** 3 * start_x + 3 * u ** 2 * t * controls[0][0] + 3 * u * t ** 2 * controls[1][0] + t ** 3 * x
            py = u ** 3 * start_y + 3 * u ** 2 * t * controls[0][1] + 3 * u * t ** 2 * controls[1][1] + t ** 3 * y
            self.page.mouse.move(px, py)
            self._mouse_position = (px, py)
            self.page.wait_for_timeout(random.uniform(8, 22))

    def _click(self, target) -> None:
        target.scroll_into_view_if_needed()
        bounds = target.bounding_box()
        if not bounds:
            raise XPostError("click target is not visible")
        viewport = self.page.viewport_size or _VIEWPORT
        left, top = max(0, bounds["x"]), max(0, bounds["y"])
        right = min(viewport["width"] - 1, bounds["x"] + bounds["width"])
        bottom = min(viewport["height"] - 1, bounds["y"] + bounds["height"])
        if right <= left or bottom <= top:
            raise XPostError("click target is outside the viewport")
        x = random.uniform(left + (right - left) * 0.25, left + (right - left) * 0.75)
        y = random.uniform(top + (bottom - top) * 0.3, top + (bottom - top) * 0.7)
        self._move_mouse(x, y)
        self.page.wait_for_timeout(random.uniform(150, 450))
        target.click(position={"x": x - bounds["x"], "y": y - bounds["y"]}, delay=random.uniform(70, 170))

    def _type_text(self, text: str) -> None:
        for char in text:
            self.page.keyboard.type(char)
            # Sample for every character, including CJK text inserted without key events.
            delay = random.uniform(*_TYPE_DELAY_MS)
            if char in ".!?;,\n\u3002\uff01\uff1f\uff1b\uff0c" or random.random() < 0.05:
                delay += random.uniform(*_TYPE_PAUSE_MS)
            self.page.wait_for_timeout(delay)

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

    def _open_compose(self, box) -> None:
        """Open the compose dialog. X's SPA occasionally never renders it; a fresh load fixes that, so retry once.
        Nothing is typed or sent yet, so retrying is safe. If it still fails, keep evidence for diagnosis."""
        for attempt in range(2):
            self.page.goto("https://x.com/compose/post", wait_until="domcontentloaded")
            try:
                box.wait_for(state="visible", timeout=30_000)
                return
            except PlaywrightTimeout:
                self._check_blocked()
        evidence = self._save_failure_evidence("compose")
        raise XPostError(f"compose box did not appear after 2 loads (at {self.page.url}); evidence: {evidence}")

    def _save_failure_evidence(self, name: str) -> str:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        base = FAILURE_DIR / f"fail-{name}-{stamp}"
        try:
            FAILURE_DIR.mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=f"{base}.png")
            body = self.page.inner_text("body", timeout=5_000)
            Path(f"{base}.txt").write_text(f"{self.page.url}\n{self.page.title()}\n\n{body[:4000]}")
            return f"{base}.png"
        except Exception as e:  # evidence is best effort and must not mask the real error
            return f"unavailable ({type(e).__name__})"

    def post(self, text: str, *, dry_run: bool = False, screenshot: Path | None = None) -> str | None:
        """Post `text`. Returns the post URL when X shows it, else None. dry_run fills the box but never posts."""
        if weighted_length(text) > MAX_WEIGHTED_LENGTH:
            raise ValueError(f"post is {weighted_length(text)} weighted chars (max {MAX_WEIGHTED_LENGTH})")
        # The background timeline has another editor with the same test id.
        box = self.page.locator(f'[role="dialog"] {_TEXTAREA}')
        self._open_compose(box)
        return self._fill_and_send(box, text, dry_run=dry_run, screenshot=screenshot)

    def _fill_and_send(self, box, text: str, *, dry_run: bool, screenshot: Path | None) -> str | None:
        """Type `text` into the open dialog's `box` and send it. Shared by posts and replies."""
        self._check_blocked()
        self.page.wait_for_timeout(random.uniform(600, 1800))
        self._click(box)
        self._type_text(text)
        self.page.wait_for_timeout(random.uniform(1200, 3000))
        button = self.page.locator(f'[role="dialog"] {_POST_BUTTON}')
        button.wait_for(state="visible", timeout=10_000)
        if not button.is_enabled():
            raise XPostError("post button is disabled (text rejected?)")
        if screenshot:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(screenshot))
        if dry_run:
            return None
        self._check_blocked()
        self._click(button)
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


    # JS run in the page. Reading only: it returns plain dicts and touches nothing.
    _TWEETS_JS = """(selector) => [...document.querySelectorAll(selector)].map(el => {
        const time = el.querySelector('time');
        const link = time && time.closest('a');
        const body = el.querySelector('[data-testid="tweetText"]');
        const context = [...el.querySelectorAll('a')]
            .filter(a => a.innerText.startsWith('@') && !a.closest('[data-testid="User-Name"]')
                         && !a.closest('[data-testid="tweetText"]'))
            .map(a => a.innerText.slice(1));
        return {path: link ? link.getAttribute('href') : null, datetime: time ? time.getAttribute('datetime') : null,
                text: body ? body.innerText : '', replying_to: context};
    })"""

    def _read_tweets(self) -> list[dict]:
        tweets = []
        for raw in self.page.evaluate(self._TWEETS_JS, _TWEET):
            m = _STATUS_PATH.match(raw["path"] or "")
            if m:  # promoted tweets and other cards have no status link
                tweets.append({"id": m.group(2), "handle": m.group(1), "url": f"https://x.com{raw['path']}",
                               "posted_at": raw["datetime"], "text": raw["text"].strip(),
                               "replying_to": [h.lower() for h in raw["replying_to"]]})
        return tweets

    def mentions(self, min_tweets: int = 0) -> list[dict]:
        """Tweets on the notifications Mentions tab, newest first. X shows about 15 without scrolling; with
        `min_tweets` the page is scrolled until that many are loaded or it stops growing."""
        self.page.goto("https://x.com/notifications/mentions", wait_until="domcontentloaded")
        try:
            self.page.wait_for_selector(_TWEET, timeout=30_000)
        except PlaywrightTimeout:
            self._check_blocked()
            return []  # nobody has mentioned the account
        self._check_blocked()
        self.page.wait_for_timeout(random.uniform(2000, 4000))
        seen = {t["id"]: t for t in self._read_tweets()}
        stalled = 0
        while len(seen) < min_tweets and stalled < 3:
            before = len(seen)
            self.page.mouse.wheel(0, 2500)
            self.page.wait_for_timeout(random.uniform(1800, 3200))
            for t in self._read_tweets():  # X drops far-off articles from the page, so collect as we go
                seen.setdefault(t["id"], t)
            stalled = stalled + 1 if len(seen) == before else 0
        return list(seen.values())

    def thread_context(self, status_url: str) -> dict:
        """Open a tweet and return {"focal": tweet, "ancestors": [tweets above it, root first]}."""
        status_id = status_url.rstrip("/").rsplit("/", 1)[-1]
        self.page.goto(status_url, wait_until="domcontentloaded")
        try:
            self.page.wait_for_selector(f'{_TWEET}:has(a[href$="/status/{status_id}"] time)', timeout=30_000)
        except PlaywrightTimeout:
            self._check_blocked()
            raise XTargetMissing(f"tweet {status_url} did not load") from None
        self._check_blocked()
        self.page.wait_for_timeout(random.uniform(2500, 4000))  # let the ancestors above it render
        tweets = self._read_tweets()
        for i, t in enumerate(tweets):
            if t["id"] == status_id:
                return {"focal": t, "ancestors": tweets[:i]}
        raise XTargetMissing(f"tweet {status_url} not found on its own page")

    def _open_reply(self, status_url: str, box) -> None:
        """Open the reply dialog of the tweet at `status_url`. Nothing is typed yet, so retrying is safe."""
        status_id = status_url.rstrip("/").rsplit("/", 1)[-1]
        focal = self.page.locator(f'{_TWEET}:has(a[href$="/status/{status_id}"] time)').first
        for attempt in range(2):
            self.page.goto(status_url, wait_until="domcontentloaded")
            try:
                focal.wait_for(state="visible", timeout=30_000)
                self.page.wait_for_timeout(random.uniform(1500, 3000))
                self._click(focal.locator('[data-testid="reply"]').first)
                box.wait_for(state="visible", timeout=15_000)
                return
            except PlaywrightTimeout:
                self._check_blocked()
        evidence = self._save_failure_evidence("reply")
        raise XTargetMissing(f"reply box did not open for {status_url}; evidence: {evidence}")

    def reply(self, status_url: str, text: str, *, dry_run: bool = False, screenshot: Path | None = None) -> str | None:
        """Reply to the tweet at `status_url`. Returns the reply's URL when X shows it, else None.
        dry_run fills the reply box but never sends."""
        if weighted_length(text) > MAX_WEIGHTED_LENGTH:
            raise ValueError(f"reply is {weighted_length(text)} weighted chars (max {MAX_WEIGHTED_LENGTH})")
        box = self.page.locator(f'[role="dialog"] {_TEXTAREA}')
        self._open_reply(status_url, box)
        return self._fill_and_send(box, text, dry_run=dry_run, screenshot=screenshot)


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
