"""Fetch news items (RSS, Hacker News), HN comments and article text.

Items are plain dicts so they serialize cleanly into graph checkpoints.
"""

import hashlib
import html
import re
import time
from calendar import timegm
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import feedparser
import httpx
import trafilatura

from .config import BLOCKED_DOMAINS, HN_LOOKBACK_HOURS, MAX_ARTICLE_CHARS, MAX_HN_COMMENTS, TOPICS, USER_AGENT

HN_API = "https://hn.algolia.com/api/v1"
_TAG_RE = re.compile(r"<[^>]+>")
_TRACKING_PARAMS = re.compile(r"^(utm_|fbclid$|gclid$|cmpid$|ref$|smid$)")


def _client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=20, follow_redirects=True)


def html_to_text(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub(" ", s or ""))).strip()


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not _TRACKING_PARAMS.match(k)])
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower().removeprefix("www."),
                       parts.path.rstrip("/"), query, ""))


def item_id(url: str) -> str:
    return hashlib.sha1(normalize_url(url).encode()).hexdigest()[:16]


def domain_of(url: str) -> str:
    return urlsplit(url).netloc.lower().removeprefix("www.")


def is_blocked(domain: str) -> bool:
    return any(domain == d or domain.endswith("." + d) for d in BLOCKED_DOMAINS)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _make_item(url, title, source, domain, topic, published_ts, summary, origin, **extra) -> dict:
    return {
        "id": item_id(url), "url": url, "title": title.strip(), "source": source, "domain": domain,
        "topic": topic, "published": _iso(published_ts), "summary": summary[:600], "origin": origin,
        "hn_id": None, "hn_points": 0, "hn_comments": 0, **extra,
    }


def fetch_rss(client: httpx.Client, source: str, url: str, topic: str, since_ts: float) -> list[dict]:
    feed = feedparser.parse(client.get(url).content)
    items = []
    for e in feed.entries:
        link = e.get("link")
        if not link:
            continue
        parsed = e.get("published_parsed") or e.get("updated_parsed")
        ts = timegm(parsed) if parsed else time.time()
        if ts < since_ts:
            continue
        title, src, domain = e.get("title", ""), source, domain_of(link)
        if "news.google.com" in link:
            # Google News wraps the real link; the publisher is in <source>.
            pub = e.get("source") or {}
            domain = domain_of(pub.get("href", "")) or domain
            src = pub.get("title") or source
            title = title.removesuffix(f" - {src}")
        if is_blocked(domain):
            continue
        items.append(_make_item(link, title, src, domain, topic, ts, html_to_text(e.get("summary", "")), "rss"))
    return items


def fetch_hn(client: httpx.Client, since_ts: float) -> list[dict]:
    """Front page plus recent popular stories; topic is left for the classifier."""
    hits = client.get(f"{HN_API}/search", params={"tags": "front_page", "hitsPerPage": 60}).json()["hits"]
    hits += client.get(f"{HN_API}/search_by_date", params={
        "tags": "story", "hitsPerPage": 100,
        "numericFilters": f"created_at_i>{int(since_ts)},points>40",
    }).json()["hits"]
    items, seen = [], set()
    for h in hits:
        if h["objectID"] in seen or h.get("created_at_i", 0) < since_ts:
            continue
        seen.add(h["objectID"])
        url = h.get("url") or f"https://news.ycombinator.com/item?id={h['objectID']}"
        domain = domain_of(url)
        if is_blocked(domain):
            continue
        items.append(_make_item(
            url, h.get("title") or "", "Hacker News", domain, None, h["created_at_i"],
            html_to_text(h.get("story_text") or ""), "hn",
            hn_id=h["objectID"], hn_points=h.get("points") or 0, hn_comments=h.get("num_comments") or 0,
        ))
    return items


def fetch_all(since_ts: float) -> tuple[list[dict], list[str]]:
    """Return (items, errors). A failing source is reported, not fatal."""
    jobs = [(t.key, name, url) for t in TOPICS for name, url in t.feeds]
    items, errors = [], []
    with _client() as client, ThreadPoolExecutor(8) as pool:
        futures = {pool.submit(fetch_rss, client, name, url, topic, since_ts): name for topic, name, url in jobs}
        futures[pool.submit(fetch_hn, client, time.time() - HN_LOOKBACK_HOURS * 3600)] = "Hacker News"
        for fut, name in futures.items():
            try:
                items.extend(fut.result())
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}: {e}"[:300])
    return items, errors


def fetch_hn_comments(hn_id: str, limit: int = MAX_HN_COMMENTS) -> list[str]:
    """Breadth-first so top-level opinions come before deep replies."""
    with _client() as client:
        root = client.get(f"{HN_API}/items/{hn_id}").json()
    out, queue = [], list(root.get("children", []))
    while queue and len(out) < limit:
        node = queue.pop(0)
        text = html_to_text(node.get("text") or "")
        if text:
            out.append(text[:700])
        queue.extend(node.get("children", []))
    return out


def fetch_article_text(url: str) -> str:
    if "news.google.com" in url:
        return ""
    try:
        with _client() as client:
            resp = client.get(url, timeout=15)
        if resp.status_code != 200:
            return ""
        return (trafilatura.extract(resp.text) or "")[:MAX_ARTICLE_CHARS]
    except Exception:
        return ""
