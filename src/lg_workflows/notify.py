"""Alerts to the owner through the KDE/freedesktop notification area and/or a Telegram bot.

Channels come from NOTIFY_CHANNELS in .env (comma-separated: desktop, telegram; default: desktop).
Telegram needs TELEGRAM_BOT_TOKEN (from @BotFather) and TELEGRAM_CHAT_ID.

`alert()` and `resolve()` track state per key, so an ongoing problem is announced once (and again daily
while it lasts) instead of every hour, and its recovery is announced once.

    uv run python -m lg_workflows.notify chat-id   # after messaging your bot, prints your chat id
    uv run python -m lg_workflows.notify test      # sends a test message on every channel
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "alerts.json"
REMIND_AFTER = timedelta(hours=24)
APP_NAME = "lg-workflows"


def _channels() -> list[str]:
    return [c.strip() for c in os.environ.get("NOTIFY_CHANNELS", "desktop").split(",") if c.strip()]


def _desktop(title: str, body: str) -> None:
    exe = shutil.which("notify-send")
    if not exe:
        raise RuntimeError("notify-send not found")
    # Critical notifications stay on screen until dismissed.
    subprocess.run([exe, "-u", "critical", "-a", APP_NAME, title, body], check=True, timeout=10)


def _telegram(title: str, body: str) -> None:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
    r = httpx.post(f"https://api.telegram.org/bot{token}/sendMessage",
                   json={"chat_id": chat, "text": f"{title}\n\n{body}"}, timeout=20)
    r.raise_for_status()


_SENDERS = {"desktop": _desktop, "telegram": _telegram}


def send(title: str, body: str) -> list[str]:
    """Send on every configured channel. Returns per-channel errors; never raises."""
    errors = []
    for ch in _channels():
        try:
            _SENDERS[ch](title, body)
        except Exception as e:
            errors.append(f"{ch}: {type(e).__name__}: {str(e)[:200]}")
    for e in errors:
        print(f"notify failed: {e}", file=sys.stderr)
    return errors


def _load() -> dict:
    try:
        return json.loads(STATE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=1))


def alert(key: str, title: str, body: str, *, after_failures: int = 1) -> bool:
    """Record a failure for `key`; notify once `after_failures` in a row are seen, then daily. Returns True if sent."""
    state, now = _load(), datetime.now(timezone.utc)
    s = state.setdefault(key, {"failures": 0})
    s["failures"] += 1
    s["last_error"] = body
    last = s.get("notified_at")
    due = s["failures"] >= after_failures and (last is None or now - datetime.fromisoformat(last) >= REMIND_AFTER)
    if due:
        send(title, body)
        s["notified_at"] = now.isoformat()
    _save(state)
    return due


def resolve(key: str, title: str, body: str) -> bool:
    """Clear `key`; announce the recovery only if a failure was announced. Returns True if sent."""
    state = _load()
    s = state.pop(key, None)
    if s is None:
        return False
    _save(state)
    if s.get("notified_at"):
        send(title, body)
        return True
    return False


def _print_chat_ids() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        sys.exit("set TELEGRAM_BOT_TOKEN in .env first")
    r = httpx.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=20)
    r.raise_for_status()
    chats = {u["message"]["chat"]["id"]: u["message"]["chat"].get("username") or u["message"]["chat"].get("title")
             for u in r.json().get("result", []) if "message" in u}
    if not chats:
        sys.exit("no messages yet: send any message to your bot in Telegram, then rerun")
    for cid, name in chats.items():
        print(f"TELEGRAM_CHAT_ID={cid}  # {name}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "chat-id":
        _print_chat_ids()
    elif cmd == "test":
        errors = send("lg-workflows 测试通知", f"通知渠道：{', '.join(_channels())}")
        sys.exit(1 if errors else 0)
    else:
        sys.exit("usage: python -m lg_workflows.notify [test|chat-id]")


if __name__ == "__main__":
    main()
