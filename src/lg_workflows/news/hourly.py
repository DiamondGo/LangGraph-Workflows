"""Hourly news digest.

fetch -> classify -> cluster_rank (per topic) -> summarize (fan-out per event) -> render -> post_x

All LLM work runs on the local Qwen; grouping uses local CPU embeddings; post_x is off unless X_POST_MODE
is set. Run:
    uv run python -m lg_workflows.news.hourly [--lang zh] [--force]
"""

import argparse
import json
import math
import operator
import time
from datetime import datetime, timezone
from typing import Annotated, Literal

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, field_validator
from typing_extensions import TypedDict

from lg_workflows.models import local_llm
from lg_workflows.news import store
from lg_workflows.news.clustering import group_items
from lg_workflows.news.config import (CHECKPOINT_DB, EVENTS_PER_TOPIC, FRESHNESS_HALF_LIFE_HOURS, LOOKBACK_HOURS,
                                      MAJOR_DOMAINS, MIN_EVENT_SCORE, MIN_EVENTS_PER_TOPIC, MINOR_SOURCE_WEIGHT, TOPIC_BY_KEY, TOPICS)
from lg_workflows.news.prompts import CLASSIFY, PROMPTS, check_lang
from lg_workflows.news.publish import MarkdownFilePublisher, Publisher, render_hourly
from lg_workflows.news.sources import fetch_all, fetch_article_text, fetch_hn_comments
from lg_workflows.news.x_post import post_x

CLASSIFY_BATCH = 40
CLASSIFY_MIN_BATCH = 10
SUMMARIZE_THINKING = True
MAX_RSS_ITEMS_PER_TOPIC = 150
TopicKey = Literal["us_politics", "china_politics", "ai", "other"]


class Label(BaseModel):
    idx: int
    topic: TopicKey


class TopicLabels(BaseModel):
    labels: list[Label]

    @field_validator("labels", mode="before")
    @classmethod
    def _decode_json_string(cls, v):
        # Qwen sometimes emits the array as a JSON-encoded string.
        return json.loads(v) if isinstance(v, str) else v


class EventSummary(BaseModel):
    title: str
    summary: str
    comments_summary: str = ""


class NewsState(TypedDict, total=False):
    run_key: str
    local_time: str
    lang: str
    items: list[dict]
    errors: Annotated[list[str], operator.add]
    candidates: list[dict]
    summaries: Annotated[list[dict], operator.add]
    digest_path: str
    x_posts: list[str]


class SummarizeInput(TypedDict):
    event: dict
    lang: str


def _merge_duplicates(items: list[dict]) -> list[dict]:
    """Same URL from several feeds -> one item; HN stats are folded into the RSS item."""
    merged: dict[str, dict] = {}
    for it in items:
        cur = merged.get(it["id"])
        if cur is None:
            merged[it["id"]] = dict(it)
            continue
        if it["origin"] == "hn":
            cur.update(hn_id=it["hn_id"], hn_points=it["hn_points"], hn_comments=it["hn_comments"])
        elif cur["origin"] == "hn":
            merged[it["id"]] = {**it, "hn_id": cur["hn_id"], "hn_points": cur["hn_points"],
                                "hn_comments": cur["hn_comments"]}
    return list(merged.values())


def fetch(state: NewsState) -> dict:
    items, errors = fetch_all(since_ts=time.time() - LOOKBACK_HOURS * 3600)
    items = store.unreported(_merge_duplicates(items))
    return {"items": items, "errors": errors}


def classify(state: NewsState) -> dict:
    """Label every item; the label replaces the feed's topic (feeds carry off-topic stories)."""
    items = state["items"]
    labels = store.cached_topics([i["id"] for i in items])
    todo = [i for i in items if i["id"] not in labels]
    model = local_llm(thinking=False).with_structured_output(TopicLabels, method="function_calling")
    new_labels: dict[str, str] = {}
    errors: list[str] = []
    pending = [todo[s:s + CLASSIFY_BATCH] for s in range(0, len(todo), CLASSIFY_BATCH)]
    while pending:
        batch = pending.pop(0)
        headlines = "\n".join(f"[{n}] {i['title']}" for n, i in enumerate(batch))
        try:
            result = model.invoke(CLASSIFY.format(headlines=headlines))
            by_idx = {l.idx: l.topic for l in result.labels}
        except Exception as e:
            # Malformed output is usually batch-specific, so halve and retry.
            if len(batch) > CLASSIFY_MIN_BATCH:
                half = len(batch) // 2
                pending[:0] = [batch[:half], batch[half:]]
            else:
                # Not cached, so the batch is retried next hour.
                errors.append(f"classify batch of {len(batch)}: {type(e).__name__}: {str(e)[:200]}")
            continue
        new_labels.update({i["id"]: by_idx.get(n, "other") for n, i in enumerate(batch)})
    store.cache_topics(new_labels)
    labels.update(new_labels)
    labeled = [{**i, "topic": labels[i["id"]]} for i in items if i["id"] in labels]
    return {"items": [i for i in labeled if i["topic"] in TOPIC_BY_KEY], "errors": errors}


def _source_weight(domain: str) -> float:
    major = any(domain == d or domain.endswith("." + d) for d in MAJOR_DOMAINS)
    return 1.0 if major else MINOR_SOURCE_WEIGHT


def _to_event(topic_key: str, members: list[dict], now: datetime) -> dict:
    # Fetchable, major-outlet reports first: they lead the links and feed the summarizer.
    members = sorted(members, key=lambda m: ("news.google.com" in m["url"], -_source_weight(m["domain"])))
    outlets = {m["domain"]: m["source"] for m in members if m["origin"] != "hn"}
    sources = sorted(set(outlets.values()))
    hn = max(members, key=lambda m: m["hn_comments"] if m["hn_id"] else -1)
    hn_points, hn_comments = (hn["hn_points"], hn["hn_comments"]) if hn["hn_id"] else (0, 0)
    newest = max(m["published"] for m in members)
    heat = (sum(_source_weight(d) for d in outlets)
            + 0.5 * math.log1p(hn_points) + 0.5 * math.log1p(hn_comments))
    age_h = (now - datetime.fromisoformat(newest)).total_seconds() / 3600
    links, seen_domains = [], set()
    for m in members:
        if m["domain"] not in seen_domains:
            seen_domains.add(m["domain"])
            links.append({"source": m["domain"] if m["origin"] == "hn" else m["source"], "url": m["url"]})
    if hn["hn_id"]:
        links.append({"source": "HN", "url": f"https://news.ycombinator.com/item?id={hn['hn_id']}"})
    return {
        "topic": topic_key, "items": members, "sources": sources or ["Hacker News"], "links": links,
        "newest": newest, "heat": heat, "score": heat * 0.5 ** (max(age_h, 0) / FRESHNESS_HALF_LIFE_HOURS),
        "hn_id": hn["hn_id"], "hn_points": hn_points, "hn_comments": hn_comments,
    }


def cluster_rank(state: NewsState) -> dict:
    now = datetime.now(timezone.utc)
    candidates = []
    for topic in TOPICS:
        topic_items = [i for i in state["items"] if i["topic"] == topic.key]
        rss = sorted((i for i in topic_items if not i["hn_id"]), key=lambda i: i["published"],
                     reverse=True)[:MAX_RSS_ITEMS_PER_TOPIC]
        items = rss + [i for i in topic_items if i["hn_id"]]
        if not items:
            continue
        events = [_to_event(topic.key, g, now) for g in group_items(items)]
        ranked = sorted(events, key=lambda e: -e["score"])
        keep = max(MIN_EVENTS_PER_TOPIC, sum(e["score"] >= MIN_EVENT_SCORE for e in ranked))
        candidates += ranked[:min(keep, EVENTS_PER_TOPIC)]
    return {"candidates": candidates}


def fan_out(state: NewsState):
    if not state["candidates"]:
        return "render"
    return [Send("summarize", {"event": e, "lang": state["lang"]}) for e in state["candidates"]]


def summarize(payload: SummarizeInput) -> dict:
    ev, lang = payload["event"], payload["lang"]
    P = PROMPTS[lang]
    parts = []
    for m in ev["items"][:4]:
        body = fetch_article_text(m["url"])[:2500] if len(parts) < 3 else ""
        parts.append(f"- [{m['source']}] {m['title']}\n  {m['summary'][:400]}" + (f"\n  Excerpt: {body}" if body else ""))
    comments = fetch_hn_comments(ev["hn_id"]) if ev["hn_id"] and ev["hn_comments"] else []
    comments_block = ("\nHacker News comments:\n" + "\n".join(f"- {c}" for c in comments))[:7000] if comments else ""
    prompt = P["summarize"].format(
        topic=TOPIC_BY_KEY[ev["topic"]].name_zh if lang == "zh" else TOPIC_BY_KEY[ev["topic"]].name_en,
        articles="\n".join(parts),
        comments=comments_block,
        comments_instruction=P["comments_instruction_on" if comments else "comments_instruction_off"],
    )
    errors, s = [], None
    # A slow or busy server times out the thinking pass; the non-thinking retry takes ~10s.
    for thinking in dict.fromkeys((SUMMARIZE_THINKING, False)):
        try:
            s = local_llm(thinking=thinking).with_structured_output(EventSummary, method="function_calling").invoke(prompt)
            break
        except Exception as e:
            errors.append(f"summarize (thinking={thinking}) {ev['items'][0]['title'][:60]!r}: "
                          f"{type(e).__name__}: {str(e)[:200]}")
    if s is not None:
        title, summary, cs = s.title, s.summary, (s.comments_summary or "").strip() if comments else ""
    else:
        first = ev["items"][0]
        title, summary, cs = first["title"], first["summary"][:300], ""
    return {"summaries": [{**ev, "title": title, "summary": summary, "comments_summary": cs}], "errors": errors}


PUBLISHERS: list[Publisher] = [MarkdownFilePublisher()]


def render(state: NewsState) -> dict:
    local_time = datetime.fromisoformat(state["local_time"])
    title, markdown = render_hourly(local_time, state["lang"], state.get("summaries", []), state.get("errors", []))
    digest = {"kind": "hourly", "run_key": state["run_key"], "lang": state["lang"], "title": title,
              "markdown": markdown, "local_time": state["local_time"]}
    paths = [p.publish(digest) for p in PUBLISHERS]
    store.save_run(state["run_key"], datetime.now(timezone.utc).isoformat(), state["lang"],
                   state.get("summaries", []))
    return {"digest_path": paths[0]}


def build_graph() -> StateGraph:
    b = StateGraph(NewsState)
    b.add_node("fetch", fetch)
    b.add_node("classify", classify)
    b.add_node("cluster_rank", cluster_rank)
    b.add_node("summarize", summarize)
    b.add_node("render", render)
    b.add_node("post_x", post_x)
    b.add_edge(START, "fetch")
    b.add_edge("fetch", "classify")
    b.add_edge("classify", "cluster_rank")
    b.add_conditional_edges("cluster_rank", fan_out, ["summarize", "render"])
    b.add_edge("summarize", "render")
    b.add_edge("render", "post_x")
    b.add_edge("post_x", END)
    return b


graph = build_graph().compile()


def run(lang: str = "zh", force: bool = False) -> str | None:
    check_lang(lang)
    now = datetime.now().astimezone()
    run_key = f"{now:%Y-%m-%dT%H}"
    thread = f"news-hourly-{run_key}-{lang}" + (f"-{int(time.time())}" if force else "")
    # The local model serves one request at a time; more parallel summaries only queue up and time out.
    cfg = {"configurable": {"thread_id": thread}, "max_concurrency": 2, "recursion_limit": 50}
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as cp:
        app = build_graph().compile(checkpointer=cp)
        snap = app.get_state(cfg)
        if snap.values.get("digest_path"):
            print(f"already done for {run_key}: {snap.values['digest_path']} (use --force to rerun)")
            return snap.values["digest_path"]
        payload = None if snap.next else {"run_key": run_key, "local_time": now.isoformat(), "lang": lang}
        if payload is None:
            print(f"resuming interrupted run {thread} at {snap.next}")
        t0 = time.time()
        for update in app.stream(payload, cfg, stream_mode="updates"):
            for node, delta in update.items():
                info = ""
                if node == "fetch":
                    info = f"{len(delta['items'])} new items, {len(delta['errors'])} source errors"
                elif node == "classify":
                    info = f"{len(delta['items'])} on-topic items"
                elif node == "cluster_rank":
                    info = f"{len(delta['candidates'])} events selected"
                elif node == "summarize":
                    info = delta["summaries"][0]["title"][:40]
                    for err in delta.get("errors", []):
                        info += f"\n    ! {err}"
                elif node == "post_x":
                    posts = (delta or {}).get("x_posts")
                    info = "off" if posts is None else "".join(f"\n    {p}" for p in posts) or "nothing to post"
                print(f"[{time.time() - t0:6.1f}s] {node}: {info}", flush=True)
        path = app.get_state(cfg).values.get("digest_path")
        print(f"digest: {path}")
        return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="zh")
    p.add_argument("--force", action="store_true", help="rerun even if this hour already has a digest")
    args = p.parse_args()
    run(args.lang, args.force)


if __name__ == "__main__":
    main()
