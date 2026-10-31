"""SQLite storage: every matching listing seen, each run's best price per item, alerts, judged canadadz posts."""

import sqlite3
from contextlib import contextmanager

from .config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    run_ts TEXT, item TEXT, store TEXT, listing_id TEXT, title TEXT, url TEXT, price REAL, method TEXT
);
CREATE INDEX IF NOT EXISTS observations_item ON observations(item, run_ts);
CREATE TABLE IF NOT EXISTS best_prices (
    run_ts TEXT, item TEXT, price REAL, store TEXT, url TEXT, PRIMARY KEY (run_ts, item)
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, item TEXT, kind TEXT, price REAL, url TEXT, message TEXT
);
CREATE TABLE IF NOT EXISTS dz_posts (
    guid TEXT PRIMARY KEY, item TEXT, judged_at TEXT, title TEXT, url TEXT, applies TEXT, price REAL, reason TEXT
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


def record_run(run_ts: str, matched: list[dict]) -> None:
    with connect() as c:
        c.executemany("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?)",
                      [(run_ts, l["item"], l["store"], l["id"], l["title"], l["url"], l["price"], l["method"])
                       for l in matched])
        best: dict[str, dict] = {}
        for l in matched:
            if l["item"] not in best or l["price"] < best[l["item"]]["price"]:
                best[l["item"]] = l
        c.executemany("INSERT OR REPLACE INTO best_prices VALUES (?,?,?,?,?)",
                      [(run_ts, k, l["price"], l["store"], l["url"]) for k, l in best.items()])


def best_prices_since(item: str, since_iso: str) -> list[float]:
    with connect() as c:
        rows = c.execute("SELECT price FROM best_prices WHERE item = ? AND run_ts >= ? ORDER BY run_ts",
                         (item, since_iso))
        return [r["price"] for r in rows]


def last_price_alert(item: str, since_iso: str) -> float | None:
    with connect() as c:
        row = c.execute("SELECT MIN(price) AS p FROM alerts WHERE item = ? AND kind = 'price' AND ts >= ?",
                        (item, since_iso)).fetchone()
        return row["p"]


def record_alert(ts: str, item: str, kind: str, price: float | None, url: str, message: str) -> None:
    with connect() as c:
        c.execute("INSERT INTO alerts (ts, item, kind, price, url, message) VALUES (?,?,?,?,?,?)",
                  (ts, item, kind, price, url, message))


def judged_dz(guids: list[str]) -> set[str]:
    if not guids:
        return set()
    with connect() as c:
        rows = c.execute(f"SELECT guid FROM dz_posts WHERE guid IN ({','.join('?' * len(guids))})", guids)
        return {r["guid"] for r in rows}


def record_dz(ts: str, post: dict) -> None:
    with connect() as c:
        c.execute("INSERT OR REPLACE INTO dz_posts VALUES (?,?,?,?,?,?,?,?)",
                  (post["guid"], post["item"], ts, post["title"], post["url"], post["applies"], post["price"],
                   post["reason"]))
