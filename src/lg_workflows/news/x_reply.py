"""Reply to comments on our own X posts.

collect -> draft -> send

- collect: read the notifications Mentions tab and keep only direct comments on our own top-level posts. A comment
  on a comment, a reply under someone else's post, a Grok answer, or a plain mention is skipped.
- draft: the local model writes one short, friendly reply in the commenter's language, or declines spam, abuse,
  political arguments and prompt-injection attempts. Replies that break a rule are rewritten once, then dropped.
- send: type the reply into X's reply box and send it (live) or stop short of sending (dry_run).

Mode comes from X_REPLY_MODE in .env: off (default), dry_run or live. Every comment is recorded in the
x_replies table of data/news.db, so it is looked at once. Run:
    uv run python -m lg_workflows.news.x_reply [--mode dry_run]
"""

import argparse
import os
import random
import re
import time
from datetime import datetime, timedelta, timezone

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from lg_workflows import notify, x_browser
from lg_workflows.models import local_llm, local_model_name
from lg_workflows.news import store
from lg_workflows.news.config import (X_REPLY_QUIET_HOURS, X_REPLY_GAP_SECONDS, X_REPLY_IGNORE_HANDLES, X_REPLY_MAX_AGE_HOURS,
                                      X_REPLY_MAX_PER_AUTHOR_PER_POST, X_REPLY_MAX_PER_RUN, X_REPLY_MAX_WEIGHT,
                                      X_SCREENSHOT_DIR)
from lg_workflows.news.prompts import X_REPLY
from lg_workflows.news.x_post import REPO_DIR

MODES = ("off", "dry_run", "live")
ALERT_KEY = "x_reply"
_LINK_OR_TAG = re.compile(r"https?://|www\.|[@＠#＃]", re.IGNORECASE)
# The account never promises anything in a reply. The prompt says so; this catches the common slips.
_PROMISE = re.compile(r"(后面|之后|以后|回头|稍后|下次|马上)[^，。！？,.]{0,3}(再|会|就|补|更新|出|写|跟进)|我们?会|会继续|"
                      r"\b(will|I'll|we'll|going to) (add|cover|follow|update|share|post|look)", re.IGNORECASE)


def reply_mode() -> str:
    mode = os.environ.get("X_REPLY_MODE", "off").strip().lower()
    if mode not in MODES:
        raise ValueError(f"X_REPLY_MODE must be one of {MODES}, got {mode!r}")
    return mode


class ReplyDraft(BaseModel):
    should_reply: bool
    reply: str = ""
    reason: str = ""


class ReplyState(TypedDict, total=False):
    run_key: str
    local_time: str
    mode: str
    me: str
    candidates: list[dict]
    drafts: list[dict]
    results: list[str]


def _skip_reason(m: dict, me: str, now: datetime, *, backfill: bool = False) -> str | None:
    """Cheap checks on a mention before its thread is opened. A backfill ignores the age limit."""
    if m["handle"].lower() == me:
        return "own comment"
    if m["handle"].lower() in X_REPLY_IGNORE_HANDLES:
        return "ignored account"
    if me not in m["replying_to"]:
        return "not a reply to us"
    if not m["posted_at"]:
        return "no timestamp"
    if backfill:
        return None
    age = now - datetime.fromisoformat(m["posted_at"].replace("Z", "+00:00"))
    if age > timedelta(hours=X_REPLY_MAX_AGE_HOURS):
        return f"older than {X_REPLY_MAX_AGE_HOURS}h"
    return None


def _post_cap_reason(m: dict) -> str | None:
    """A person may get only a few replies under one post; their comments under other posts do not count."""
    if store.x_replies_to_author_on_post(m["handle"], m["chain"][-1]["text"]) >= X_REPLY_MAX_PER_AUTHOR_PER_POST:
        return f"already replied to this person {X_REPLY_MAX_PER_AUTHOR_PER_POST} times under this post"
    return None


def _thread_skip_reason(ancestors: list[dict], me: str) -> str | None:
    """Reply only to a direct comment on one of our own top-level posts, never to a comment on a comment."""
    if not ancestors:
        return "no parent tweet visible"
    if len(ancestors) > 1:
        return "comment on a comment"
    if ancestors[0]["handle"].lower() != me:
        return "parent tweet is not ours"
    return None


def _fail(stage: str, e: Exception, results: list[str]) -> dict:
    """Report a browser failure the way news posting does. The comments stay unrecorded, so the next run retries."""
    if isinstance(e, x_browser.XProfileBusy):
        results.append(f"{stage} skipped: {e}")  # the news job holds the profile; not a fault
    elif isinstance(e, x_browser.XLoginRequired):
        results.append(f"{stage} stopped: {e}")
        notify.alert(ALERT_KEY, "X 回复已停止：需要重新登录",
                     f"{e}\n\n重新登录：cd {REPO_DIR} && uv run python -m lg_workflows.x_browser login")
    elif isinstance(e, x_browser.XPostError):
        results.append(f"{stage} stopped: {e}")
        notify.alert(ALERT_KEY, "X 回复已停止", str(e))
    else:
        results.append(f"{stage} failed: {type(e).__name__}: {str(e)[:300]}")
        notify.alert(ALERT_KEY, "X 回复连续失败", results[-1], after_failures=2)
    return {"results": results}


def collect(state: ReplyState) -> dict:
    mode = reply_mode()
    if mode == "off":
        return {"mode": mode, "candidates": [], "results": []}
    hour = datetime.fromisoformat(state["local_time"]).hour
    if hour in X_REPLY_QUIET_HOURS:
        return {"mode": mode, "candidates": [], "results": [f"quiet hours ({hour:02d}:00), not replying"]}
    now = datetime.now(timezone.utc)
    handled = store.x_reply_handled_ids()
    candidates: list[dict] = []
    results: list[str] = []
    try:
        with x_browser.XSession() as session:
            me = session.handle().lstrip("@").lower()
            fresh = [m for m in session.mentions() if m["id"] not in handled]
            fresh.sort(key=lambda m: m["posted_at"] or "")  # oldest first
            results.append(f"{len(fresh)} unhandled mentions")
            for m in fresh:
                reason = _skip_reason(m, me, now)
                if reason is None:
                    time.sleep(random.uniform(2, 5))
                    try:
                        thread = session.thread_context(m["url"])
                    except x_browser.XTargetMissing as e:
                        results.append(f"retry later: {e}")  # not recorded, so the next run tries again
                        continue
                    reason = _thread_skip_reason(thread["ancestors"], me)
                    m = {**m, "chain": [{"handle": t["handle"], "text": t["text"]} for t in thread["ancestors"]]}
                    reason = reason or _post_cap_reason(m)
                if reason:
                    store.record_x_reply("skipped", mode, m, reason=reason)
                    results.append(f"skip @{m['handle']} {m['url']}: {reason}")
                else:
                    candidates.append(m)
    except Exception as e:
        return {"mode": mode, "candidates": [], **_fail("collect", e, results)}
    return {"mode": mode, "me": me, "candidates": candidates, "results": results}


_FILLER_OPENING = re.compile(
    r"^[\s\"“”'‘’(（]*(对[，,！!。 ]|对$|是的|是啊|是呀|说得对|说的对|没错|的确|确实|同意|赞同|"
    r"(exactly|indeed|true|agreed|right|yes|yeah|absolutely|definitely|totally|good point|great point)\b)",
    re.IGNORECASE)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# A Chinese number with a unit ("七个", "三成", "四分之一"). A bare "一" ("一个", "一点") is ordinary wording.
_CN_DIGITS = "零〇一二两三四五六七八九十百千万亿"
_CN_NUMBER = re.compile(rf"([{_CN_DIGITS}]+)(?:分之[{_CN_DIGITS}]+|个|家|位|名|种|条|项|成|倍|国|所|间|款|次|年|月|日|天|人|元|美元|度|轮)")
# Capitalized or all-caps Latin words: names, outlets, products.
_NAME = re.compile(r"\b[A-Z][A-Za-z0-9.+-]*")


def _script(text: str) -> str:
    """Rough writing system of `text`, to catch a reply in the wrong language. Latin covers many languages."""
    for name, pattern in (("ja", r"[\u3040-\u30ff]"), ("ko", r"[\uac00-\ud7af]"), ("zh", r"[\u4e00-\u9fff]"),
                          ("cyrillic", r"[\u0400-\u04ff]"), ("arabic", r"[\u0600-\u06ff]"),
                          ("devanagari", r"[\u0900-\u097f]"), ("thai", r"[\u0e00-\u0e7f]")):
        if re.search(pattern, text):
            return name
    return "latin"


def _sentence_starts(text: str) -> set[int]:
    return {m.end() for m in re.finditer(r"[.!?。！？]\s*", text)} | {0}


def validate_reply(text: str, comment: str, post: str) -> str | None:
    """Why `text` must not be posted as a reply to `comment` under `post`, or None. The model is told these
    rules; this enforces them and the reason is fed back to the model for one retry."""
    if not text.strip():
        return "empty reply"
    if _LINK_OR_TAG.search(text):
        return "reply contains a link, @mention or hashtag"
    if _PROMISE.search(text):
        return "reply makes a promise"
    if _FILLER_OPENING.search(text):
        return "reply opens with agreement filler (说得对/是啊/确实/对/Exactly...); start with the substance instead"
    if re.search(r"[?？]", text):
        return "reply asks a question; make a statement instead"
    if x_browser.weighted_length(text) > X_REPLY_MAX_WEIGHT:
        return (f"reply is too long ({x_browser.weighted_length(text)} weighted chars, max {X_REPLY_MAX_WEIGHT}); "
                "write one short sentence")
    if _script(text) != _script(comment):
        return f"reply must be in the commenter's language ({_script(comment)} script), not {_script(text)}"
    post_numbers = {n.replace(",", "") for n in _NUMBER.findall(post)}
    stray = [n for n in _NUMBER.findall(text) if n.replace(",", "") not in post_numbers]
    if stray:
        return f"reply uses numbers that are not in the original post: {', '.join(stray)}"
    cn_stray = [m.group() for m in _CN_NUMBER.finditer(text) if m.group(1) != "一" and m.group() not in post]
    if cn_stray:
        return f"reply uses numbers that are not in the original post: {', '.join(cn_stray)}"
    starts = _sentence_starts(text)
    post_lower = post.lower()
    names = [m.group() for m in _NAME.finditer(text) if m.start() not in starts]
    unknown = [n for n in names if n.lower().strip(".,") not in post_lower]
    if unknown:
        return f"reply names things that are not in the original post: {', '.join(unknown)}"
    return None


def write_reply(me: str, c: dict, feedback: str | None = None) -> ReplyDraft:
    chain = "\n".join(f"@{t['handle']}: {t['text']}" for t in c["chain"])
    prompt = X_REPLY.format(me=me, chain=chain, author=c["handle"], comment=c["text"])
    if feedback:
        prompt += f"\n\nYour previous reply was rejected: {feedback}. Write a new reply that fixes this."
    return (local_llm(thinking=False, temperature=0.7).with_structured_output(ReplyDraft, method="function_calling")
            .invoke(prompt))


def make_reply(me: str, c: dict) -> tuple[str | None, str | None]:
    """A reply that passes every check, or (None, why not). A rejected reply gets one rewrite with the reason."""
    feedback, problem = None, None
    for _ in range(2):
        d = write_reply(me, c, feedback)
        if not d.should_reply:
            return None, f"model declined: {d.reason}"
        text = d.reply.strip().strip('"“”')
        problem = validate_reply(text, comment=c["text"], post=c["chain"][-1]["text"])
        if problem is None:
            return text, None
        feedback = problem
    return None, problem


def draft(state: ReplyState) -> dict:
    mode, me, results = state["mode"], state.get("me", ""), list(state.get("results", []))
    drafts, answered, errors = [], set(), 0
    model = local_model_name()  # saved with every decision, for review later
    for c in state.get("candidates", []):
        if len(drafts) >= X_REPLY_MAX_PER_RUN:
            break
        key = (c["handle"].lower(), c["chain"][-1]["text"])
        if key in answered:
            continue  # their other comment under the same post waits for a later run
        try:
            text, problem = make_reply(me, c)
        except Exception as e:
            errors += 1
            results.append(f"draft failed @{c['handle']}: {type(e).__name__}: {str(e)[:200]}")
            continue
        if problem:
            store.record_x_reply("skipped", mode, c, reason=problem[:300], model=model)
            results.append(f"skip @{c['handle']} {c['url']}: {problem[:200]}")
            continue
        answered.add(key)
        drafts.append({**c, "reply": text, "model": model})
    if errors and not drafts:
        notify.alert(ALERT_KEY, "X 回复起草失败", results[-1], after_failures=2)
    return {"drafts": drafts, "results": results}


def send(state: ReplyState) -> dict:
    mode, results = state["mode"], list(state.get("results", []))
    drafts = state.get("drafts", [])
    if not drafts:
        return {"results": results}
    sent = missing = 0
    try:
        with x_browser.XSession() as session:
            for n, d in enumerate(drafts):
                if n and mode == "live":
                    time.sleep(random.uniform(*X_REPLY_GAP_SECONDS))
                shot = X_SCREENSHOT_DIR / f"reply-{state['run_key']}-{n}.png" if mode == "dry_run" else None
                try:
                    url = session.reply(d["url"], d["reply"], dry_run=mode == "dry_run", screenshot=shot)
                except x_browser.XTargetMissing as e:
                    missing += 1
                    results.append(f"retry later: {e}")
                    continue
                except x_browser.XPostUnconfirmed:
                    # Possibly live: record it so the comment is never answered twice.
                    store.record_x_reply("unconfirmed", mode, d, reply_text=d["reply"], model=d["model"])
                    raise
                store.record_x_reply("drafted" if mode == "dry_run" else "replied", mode, d,
                                     reply_text=d["reply"], reply_url=url, model=d["model"])
                sent += 1
                results.append(f"{mode}: @{d['handle']} {d['url']}\n      > {d['text']}\n      < {d['reply']}"
                               + (f"\n      screenshot: {shot}" if shot else ""))
    except Exception as e:
        return _fail("send", e, results)
    if missing and not sent:
        notify.alert(ALERT_KEY, "X 回复失败", results[-1], after_failures=2)
    elif sent:
        notify.resolve(ALERT_KEY, "X 回复已恢复", f"{state['run_key']}：{'发出' if mode == 'live' else '试填'} {sent} 条回复")
    return {"results": results}


def build_graph() -> StateGraph:
    b = StateGraph(ReplyState)
    b.add_node("collect", collect)
    b.add_node("draft", draft)
    b.add_node("send", send)
    b.add_edge(START, "collect")
    b.add_edge("collect", "draft")
    b.add_edge("draft", "send")
    b.add_edge("send", END)
    return b


graph = build_graph().compile()


def run() -> list[str]:
    now = datetime.now().astimezone()
    final = graph.invoke({"run_key": f"{now:%Y-%m-%dT%H%M}", "local_time": now.isoformat()})
    results = final.get("results", [])
    print(f"mode={final.get('mode')}")
    for line in results:
        print(f"  {line}")
    return results


BACKFILL_SCAN_TWEETS = 300
BACKFILL_REST_SECONDS = (240, 420)


def find_backlog(limit: int, mode: str) -> tuple[str, list[dict], list[str]]:
    """Scroll back through the Mentions tab for up to `limit` unanswered direct comments on our top-level posts."""
    now = datetime.now(timezone.utc)
    handled = store.x_reply_handled_ids(revisit_old=True)
    candidates: list[dict] = []
    log: list[str] = []
    with x_browser.XSession() as session:
        me = session.handle().lstrip("@").lower()
        mentions = session.mentions(min_tweets=BACKFILL_SCAN_TWEETS)
        log.append(f"{len(mentions)} mentions loaded")
        for m in mentions:  # newest first
            if len(candidates) >= limit:
                break
            if m["id"] in handled:
                continue
            reason = _skip_reason(m, me, now, backfill=True)
            if reason is None:
                time.sleep(random.uniform(2, 4))
                try:
                    thread = session.thread_context(m["url"])
                except x_browser.XTargetMissing as e:
                    log.append(f"retry later: {e}")
                    continue
                reason = _thread_skip_reason(thread["ancestors"], me)
                m = {**m, "chain": [{"handle": t["handle"], "text": t["text"]} for t in thread["ancestors"]]}
                reason = reason or _post_cap_reason(m)
            if reason:
                store.record_x_reply("skipped", mode, m, reason=reason)
            else:
                candidates.append(m)
    log.append(f"{len(candidates)} unanswered direct comments found")
    return me, candidates, log


def backfill(limit: int = 20) -> list[str]:
    """One-off catch-up: answer up to `limit` older direct comments that never got a reply. Same rules, drafting
    and checks as the regular run, but sent in small batches with rests between them, so the shared Chrome profile
    is free for the news job and the account does not reply in one long burst."""
    mode = reply_mode()
    if mode == "off":
        return ["X_REPLY_MODE is off"]
    now = datetime.now().astimezone()
    run_key = f"{now:%Y-%m-%dT%H%M}-backfill"
    results: list[str] = []
    try:
        me, pending, log = find_backlog(limit, mode)
    except Exception as e:
        return _fail("backfill", e, results)["results"]
    results += log
    batch_no = 0
    while pending:
        if datetime.now().hour in X_REPLY_QUIET_HOURS:
            results.append(f"quiet hours, {len(pending)} comments left for another time")
            break
        pending = [c for c in pending if c["id"] not in store.x_reply_handled_ids(revisit_old=True)]  # a timer run may have answered some
        drafted = draft({"mode": mode, "me": me, "candidates": pending, "results": []})
        results += drafted["results"]
        done = {d["id"] for d in drafted["drafts"]} | store.x_reply_handled_ids(revisit_old=True)
        pending = [c for c in pending if c["id"] not in done]
        if not drafted["drafts"]:
            if pending and not any(line.startswith("draft failed") for line in drafted["results"]):
                continue  # only skips this round; the next batch of candidates is still waiting
            break
        batch_no += 1
        sent = send({"run_key": f"{run_key}-{batch_no}", "mode": mode, "drafts": drafted["drafts"], "results": []})
        results += sent["results"]
        if any(line.startswith(("send stopped", "send failed", "send skipped")) for line in sent["results"]):
            results.append("stopping the backfill after a send problem")
            break
        if pending:
            time.sleep(random.uniform(*BACKFILL_REST_SECONDS))
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=MODES, help="override X_REPLY_MODE for this run")
    p.add_argument("--backfill", type=int, metavar="N", help="answer up to N older unanswered comments, in batches")
    args = p.parse_args()
    if args.mode:
        os.environ["X_REPLY_MODE"] = args.mode
    if args.backfill:
        for line in backfill(args.backfill):
            print(line, flush=True)
    else:
        run()


if __name__ == "__main__":
    main()
