"""SQLite storage: which items were already reported, cached topic labels, and event summaries."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reported_items (
    id TEXT PRIMARY KEY, url TEXT, title TEXT, source TEXT, topic TEXT,
    published TEXT, reported_at TEXT
);
CREATE TABLE IF NOT EXISTS item_topics (
    id TEXT PRIMARY KEY, topic TEXT, labeled_at TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_key TEXT, run_at TEXT, topic TEXT, lang TEXT,
    title TEXT, summary TEXT, comments_summary TEXT, score REAL,
    sources TEXT, links TEXT, hn_comments INTEGER
);
CREATE INDEX IF NOT EXISTS events_run_at ON events(run_at);
CREATE TABLE IF NOT EXISTS x_posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_key TEXT, posted_at TEXT, mode TEXT,
    anchor TEXT, text TEXT, url TEXT
);
"""


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def unreported(items: list[dict]) -> list[dict]:
    with connect() as c:
        done = {r["id"] for r in c.execute("SELECT id FROM reported_items")}
    return [i for i in items if i["id"] not in done]


def cached_topics(ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    with connect() as c:
        rows = c.execute(f"SELECT id, topic FROM item_topics WHERE id IN ({','.join('?' * len(ids))})", ids)
        return {r["id"]: r["topic"] for r in rows}


def cache_topics(labels: dict[str, str]) -> None:
    with connect() as c:
        c.executemany("INSERT OR REPLACE INTO item_topics VALUES (?, ?, ?)",
                      [(k, v, _now()) for k, v in labels.items()])


def save_run(run_key: str, run_at: str, lang: str, events: list[dict]) -> None:
    """Persist event summaries and mark their items as reported, atomically."""
    with connect() as c:
        for e in events:
            c.execute(
                "INSERT INTO events (run_key, run_at, topic, lang, title, summary, comments_summary, score,"
                " sources, links, hn_comments) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run_key, run_at, e["topic"], lang, e["title"], e["summary"], e.get("comments_summary"),
                 e["score"], json.dumps(e["sources"]), json.dumps(e["links"]), e["hn_comments"]),
            )
            c.executemany("INSERT OR IGNORE INTO reported_items VALUES (?,?,?,?,?,?,?)",
                          [(i["id"], i["url"], i["title"], i["source"], e["topic"], i["published"], _now())
                           for i in e["items"]])


def record_x_post(run_key: str, mode: str, anchor: str, text: str, url: str | None) -> None:
    with connect() as c:
        c.execute("INSERT INTO x_posts (run_key, posted_at, mode, anchor, text, url) VALUES (?,?,?,?,?,?)",
                  (run_key, _now(), mode, anchor, text, url))


def x_post_anchors_since(since_iso: str, mode: str = "live") -> list[str]:
    with connect() as c:
        rows = c.execute("SELECT anchor FROM x_posts WHERE posted_at >= ? AND mode = ?", (since_iso, mode))
        return [r["anchor"] for r in rows]


def events_between(start_iso: str, end_iso: str, lang: str) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM events WHERE run_at >= ? AND run_at < ? AND lang = ? ORDER BY score DESC",
                         (start_iso, end_iso, lang)).fetchall()
    return [{**dict(r), "sources": json.loads(r["sources"]), "links": json.loads(r["links"])} for r in rows]
