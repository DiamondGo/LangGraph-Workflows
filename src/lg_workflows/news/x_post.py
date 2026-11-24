"""Post the hottest freshly summarized events of an hourly run to X."""

import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from pydantic import BaseModel

from lg_workflows import notify, x_browser
from lg_workflows.models import local_llm
from lg_workflows.news import store
from lg_workflows.news.clustering import embed
from lg_workflows.news.config import (CLUSTER_THRESHOLD, X_FILL_MIN_SCORE, X_MAX_POSTS_PER_RUN, X_MIN_POST_SCORE,
                                      X_POST_GAP_SECONDS, X_QUIET_HOURS, X_REPOST_WINDOW_HOURS, X_SCREENSHOT_DIR,
                                      X_TARGET_POSTS_PER_RUN)
from lg_workflows.news.prompts import X_NOVELTY

MODES = ("off", "dry_run", "live")
ALERT_KEY = "x_post"
REPO_DIR = Path(__file__).resolve().parents[3]


def post_mode() -> str:
    mode = os.environ.get("X_POST_MODE", "off").strip().lower()
    if mode not in MODES:
        raise ValueError(f"X_POST_MODE must be one of {MODES}, got {mode!r}")
    return mode


def anchor(event: dict) -> str:
    """Source-language text identifying the story, comparable across hours."""
    m = event["items"][0]
    return f"{m['title']}. {m['summary'][:200]}"


def _link(event: dict) -> str | None:
    urls = [l["url"] for l in event["links"]]
    direct = [u for u in urls if "news.google.com" not in u]
    return (direct or urls or [None])[0]


def _fit_sentences(text: str, budget: int) -> str:
    if budget < 20:
        return ""
    sentences = [s + "。" for s in text.split("。") if s.strip()]
    out = ""
    for s in sentences:
        if x_browser.weighted_length(out + s) > budget:
            break
        out += s
    return out or x_browser.truncate_to_weight(text, budget)


def compose(event: dict) -> str:
    header = event["title"]
    link = _link(event)
    tail = f"\n\n{link}" if link else ""
    header = x_browser.truncate_to_weight(header, x_browser.MAX_WEIGHTED_LENGTH - x_browser.weighted_length(tail))
    budget = x_browser.MAX_WEIGHTED_LENGTH - x_browser.weighted_length(header + "\n\n" + tail)
    body = _fit_sentences(event["summary"], budget)
    return header + (f"\n\n{body}" if body else "") + tail


class Novelty(BaseModel):
    materially_new: bool
    reason: str = ""


def is_materially_new(event: dict, prior_texts: list[str]) -> bool:
    """Ask the LLM whether `event` adds an important new fact to stories already posted. Fails closed."""
    prompt = X_NOVELTY.format(posted="\n---\n".join(prior_texts), title=event["title"], summary=event["summary"])
    try:
        verdict = (local_llm(thinking=False).with_structured_output(Novelty, method="function_calling")
                   .invoke(prompt))
    except Exception as e:
        print(f"    ! novelty check {event['title'][:40]!r}: {type(e).__name__}: {e}"[:300])
        return False
    print(f"    novelty {event['title'][:40]!r}: {verdict.materially_new} {verdict.reason}"[:300])
    return verdict.materially_new


def select(summaries: list[dict], posted: list[dict]) -> list[dict]:
    """Pick events to post: all above X_MIN_POST_SCORE (max X_MAX_POSTS_PER_RUN), topped up to X_TARGET_POSTS_PER_RUN.
    A story similar to one already posted is skipped unless it has a materially new development."""
    # A failed summary keeps the source headline as its title; never post those.
    ok = [s for s in summaries if s["title"] != s["items"][0]["title"] and s["score"] >= X_FILL_MIN_SCORE]
    ok.sort(key=lambda s: -s["score"])
    prior = embed([p["anchor"] for p in posted]) if posted else np.zeros((0, 0))
    chosen, chosen_vecs = [], []
    for s in ok:
        if len(chosen) >= X_MAX_POSTS_PER_RUN:
            break
        if s["score"] < X_MIN_POST_SCORE and len(chosen) >= X_TARGET_POSTS_PER_RUN:
            break  # only top up to the target with below-bar events; ok is sorted, so nothing better follows
        v = embed([anchor(s)])[0]
        if chosen_vecs and float((np.array(chosen_vecs) @ v).max()) >= CLUSTER_THRESHOLD:
            continue  # same story as one picked earlier in this run
        if len(prior):
            sims = prior @ v
            related = [posted[i]["text"] for i in np.argsort(sims) if sims[i] >= CLUSTER_THRESHOLD]
            if related and not is_materially_new(s, related):
                continue
        chosen.append(s)
        chosen_vecs.append(v)
    return chosen


def post_x(state: dict) -> dict:
    mode = post_mode()
    if mode == "off":
        return {}
    run_hour = datetime.fromisoformat(state["local_time"]).hour if state.get("local_time") else datetime.now().hour
    if run_hour in X_QUIET_HOURS:
        return {"x_posts": [f"quiet hours ({run_hour:02d}:00), not posting"]}
    since = (datetime.now(timezone.utc) - timedelta(hours=X_REPOST_WINDOW_HOURS)).isoformat()
    chosen = select(state.get("summaries", []), store.x_posts_since(since))
    results = []
    if not chosen:
        return {"x_posts": results}
    try:
        with x_browser.XSession() as session:
            for n, ev in enumerate(chosen):
                if n and mode == "live":
                    time.sleep(random.uniform(*X_POST_GAP_SECONDS))
                text = compose(ev)
                shot = X_SCREENSHOT_DIR / f"{state['run_key']}-{n}.png" if mode == "dry_run" else None
                try:
                    url = session.post(text, dry_run=mode == "dry_run", screenshot=shot)
                except x_browser.XPostUnconfirmed:
                    # Possibly live: record it so the story is not posted again.
                    store.record_x_post(state["run_key"], mode, anchor(ev), text, None)
                    raise
                store.record_x_post(state["run_key"], mode, anchor(ev), text, url)
                results.append(f"{mode}: {url or shot or 'posted'} | {ev['title']}")
    except x_browser.XLoginRequired as e:
        results.append(f"stopped: {e}")
        notify.alert(ALERT_KEY, "X 发帖已停止：需要重新登录",
                     f"{e}\n\n重新登录：cd {REPO_DIR} && uv run python -m lg_workflows.x_browser login")
        return {"x_posts": results}
    except x_browser.XPostError as e:
        results.append(f"stopped: {e}")
        notify.alert(ALERT_KEY, "X 发帖已停止", str(e))
        return {"x_posts": results}
    except Exception as e:
        # The digest is already written; a posting failure must not fail the run.
        results.append(f"failed: {type(e).__name__}: {str(e)[:300]}")
        # Network hiccups happen; only a repeat is worth a notification.
        notify.alert(ALERT_KEY, "X 发帖连续失败", results[-1], after_failures=2)
        return {"x_posts": results}
    notify.resolve(ALERT_KEY, "X 发帖已恢复", f"{state['run_key']}：成功{'发出' if mode == 'live' else '试填'} {len(results)} 帖")
    return {"x_posts": results}
