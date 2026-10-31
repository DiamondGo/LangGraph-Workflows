"""Render digests to Markdown and hand them to publishers.

A publisher takes a finished digest and delivers it somewhere. Only the Markdown file
publisher exists today; an X.com publisher would implement the same `publish` method.
"""

from datetime import datetime
from pathlib import Path
from typing import Protocol, TypedDict

from .config import DIGEST_DIR, TOPICS
from .prompts import LABELS


class Digest(TypedDict):
    kind: str          # "hourly" | "daily"
    run_key: str
    lang: str
    title: str
    markdown: str
    local_time: str    # ISO timestamp in local time


class Publisher(Protocol):
    def publish(self, digest: Digest) -> str:
        """Deliver the digest; return where it went (path, URL, post id...)."""
        ...


class MarkdownFilePublisher:
    def __init__(self, root: Path = DIGEST_DIR):
        self.root = root

    def path_for(self, digest: Digest) -> Path:
        t = datetime.fromisoformat(digest["local_time"])
        suffix = "" if digest["lang"] == "zh" else f".{digest['lang']}"
        name = f"{t:%H}00{suffix}.md" if digest["kind"] == "hourly" else f"daily{suffix}.md"
        return self.root / f"{t:%Y-%m-%d}" / name

    def publish(self, digest: Digest) -> str:
        path = self.path_for(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(digest["markdown"], encoding="utf-8")
        return str(path)


def _age(newest_iso: str, now: datetime, labels: dict) -> str:
    minutes = max(0, int((now - datetime.fromisoformat(newest_iso)).total_seconds() // 60))
    return labels["minutes_ago"].format(n=minutes) if minutes < 120 else labels["hours_ago"].format(n=minutes // 60)


def render_hourly(local_time: datetime, lang: str, events: list[dict], errors: list[str]) -> tuple[str, str]:
    """Return (title, markdown)."""
    L = LABELS[lang]
    title = f"{L['hourly_title']} {local_time:%Y-%m-%d %H:00}"
    lines = [f"# {title}", ""]
    for topic in TOPICS:
        topic_events = sorted((e for e in events if e["topic"] == topic.key), key=lambda e: -e["score"])
        lines += [f"## {topic.name_zh if lang == 'zh' else topic.name_en}", ""]
        if not topic_events:
            lines += [L["no_events"], ""]
            continue
        for n, e in enumerate(topic_events, 1):
            meta = (f"{L['sources']}：{', '.join(e['sources'])} · {L['heat']} {e['heat']:.1f}"
                    f" · {L['freshness']} {_age(e['newest'], local_time, L)}")
            lines += [f"### {n}. {e['title']}", "", f"*{meta}*", "", e["summary"], ""]
            if e.get("comments_summary"):
                lines += [f"**{L['comments']}**（{L['comment_count'].format(n=e['hn_comments'])}）："
                          f"{e['comments_summary']}", ""]
            links = " · ".join(f"[{l['source']}]({l['url']})" for l in e["links"][:6])
            lines += [f"{L['links']}：{links}", ""]
    if errors:
        lines += [f"---", f"<details><summary>{L['errors']}（{len(errors)}）</summary>", ""]
        lines += [f"- {err}" for err in errors] + ["", "</details>", ""]
    return title, "\n".join(lines)
