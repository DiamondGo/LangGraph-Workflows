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
CREATE TABLE IF NOT EXISTS x_replies (
    id INTEGER PRIMARY KEY AUTOINCREMENT, seen_at TEXT, mode TEXT,
    status TEXT,  -- skipped | drafted (dry_run) | replied | unconfirmed
    comment_id TEXT, comment_url TEXT, author TEXT, comment_text TEXT, parent_text TEXT,
    reply_text TEXT, reply_url TEXT, reason TEXT, model TEXT
);
CREATE INDEX IF NOT EXISTS x_replies_comment ON x_replies(comment_id);
"""


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        if "model" not in {r["name"] for r in conn.execute("PRAGMA table_info(x_replies)")}:
            conn.execute("ALTER TABLE x_replies ADD COLUMN model TEXT")  # table made before the column existed
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


def x_posts_since(since_iso: str, mode: str = "live") -> list[dict]:
    """Posts since `since_iso`, oldest first, as {"anchor", "text"}."""
    with connect() as c:
        rows = c.execute("SELECT anchor, text FROM x_posts WHERE posted_at >= ? AND mode = ? ORDER BY id",
                         (since_iso, mode))
        return [{"anchor": r["anchor"], "text": r["text"]} for r in rows]


def record_x_reply(status: str, mode: str, comment: dict, *, reply_text: str | None = None,
                   reply_url: str | None = None, reason: str | None = None, model: str | None = None) -> None:
    chain = comment.get("chain") or []
    with connect() as c:
        c.execute(
            "INSERT INTO x_replies (seen_at, mode, status, comment_id, comment_url, author, comment_text,"
            " parent_text, reply_text, reply_url, reason, model) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), mode, status, comment["id"], comment["url"], comment["handle"], comment["text"],
             chain[-1]["text"] if chain else None, reply_text, reply_url, reason, model))


def x_reply_handled_ids(revisit_old: bool = False) -> set[str]:
    """Comments never to look at again. Dry-run drafts do not count: the live run must still answer them.
    With `revisit_old`, a comment that was only skipped for being older than the age limit counts as unhandled."""
    sql = "SELECT DISTINCT comment_id FROM x_replies WHERE status != 'drafted'"
    if revisit_old:
        sql += " AND NOT (status = 'skipped' AND reason LIKE 'older than%')"
    with connect() as c:
        return {r["comment_id"] for r in c.execute(sql)}


def x_replies_to_author_on_post(handle: str, post_text: str) -> int:
    """Replies already sent to `handle` under the post whose text is `post_text`."""
    with connect() as c:
        return c.execute("SELECT COUNT(*) FROM x_replies WHERE lower(author) = ? AND parent_text = ?"
                         " AND status IN ('replied', 'unconfirmed')", (handle.lower(), post_text)).fetchone()[0]


def events_between(start_iso: str, end_iso: str, lang: str) -> list[dict]:
    with connect() as c:
        rows = c.execute("SELECT * FROM events WHERE run_at >= ? AND run_at < ? AND lang = ? ORDER BY score DESC",
                         (start_iso, end_iso, lang)).fetchall()
    return [{**dict(r), "sources": json.loads(r["sources"]), "links": json.loads(r["links"])} for r in rows]
