"""canadadz.com deal posts: WordPress search as RSS, judged by the local model.

Posts are written by people, in Chinese, and often leave the variant out ("Apple Watch SE 3 8.5折"),
so keyword rules only preselect and the model decides whether the deal fits the watched item.
"""

import calendar
import re
from datetime import datetime, timedelta, timezone
from typing import Literal
from urllib.parse import quote_plus

import feedparser
import httpx
from pydantic import BaseModel

from lg_workflows.browser import user_agent
from lg_workflows.models import local_llm

from .config import DZ_MAX_AGE_DAYS, Item, has_phrase, norm_text

JUDGE_PROMPT = """You check deal posts from a Canadian deals site against a shopping watchlist entry.

Watched item: {name}
Its normal price is about {normal} CAD.

Post title: {title}
Post text: {body}

Decide:
- applies: "yes" if the deal is for exactly this item (same model, size and connectivity, new condition);
  "maybe" if it is for this model but the post does not say which size/variant, or several variants are on sale;
  "no" if it is a different model, size or variant, a used/refurbished unit, an accessory, or a bundle
  (e.g. a prebuilt PC containing the part).
- price: the deal price in CAD for the watched item as a number, or null if the post does not give one.
- reason: one short sentence in Chinese."""


class Verdict(BaseModel):
    applies: Literal["yes", "maybe", "no"]
    price: float | None = None
    reason: str = ""


def search(item: Item, now: datetime) -> list[dict]:
    r = httpx.get(f"https://www.canadadz.com/?s={quote_plus(item.dz_query)}&feed=rss2",
                  headers={"User-Agent": user_agent()}, timeout=30, follow_redirects=True)
    r.raise_for_status()
    feed = feedparser.parse(r.content)
    if feed.bozo and not feed.entries:
        raise RuntimeError(f"unparseable feed: {feed.bozo_exception}")
    since = now - timedelta(days=DZ_MAX_AGE_DAYS)
    out = []
    for e in feed.entries:
        ts = e.get("published_parsed")
        published = datetime.fromtimestamp(calendar.timegm(ts), timezone.utc) if ts else None
        title = norm_text(e.get("title", ""))
        # WordPress search also matches post bodies and substrings ("5070" in "FH50700").
        if published is None or published < since or not all(has_phrase(title, w) for w in item.dz_query.split()):
            continue
        body = re.sub(r"<[^>]+>|\s+", " ", e.get("summary", "")).strip()
        out.append({"guid": e.get("id") or e.link, "title": e.title, "url": e.link,
                    "published": published.isoformat(), "body": body[:1500]})
    return out


def judge(item: Item, post: dict, normal: float | None) -> Verdict:
    prompt = JUDGE_PROMPT.format(name=item.name, normal=f"{normal:.0f}" if normal else "unknown",
                                 title=post["title"], body=post["body"])
    return local_llm(thinking=False).with_structured_output(Verdict, method="function_calling").invoke(prompt)
