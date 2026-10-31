"""Price watch: find low prices for the watchlist items and message the owner.

scan (fan-out per store) -> agent_fallback -> evaluate -> notify

- scan: Best Buy, Amazon and Newegg listings (http -> headless browser -> screenshot + local vision model),
  and fresh canadadz.com deal posts judged by the local model.
- agent_fallback: a Claude Code agent with a headless browser, for stores whose scripted tiers all failed.
- evaluate: the item's normal price is the median of its best price per run over the last 30 days
  (the watchlist's normal_price until there is enough history); alert when today's best is 5%+ below it.
- notify: price and canadadz alerts, plus an alert when a store keeps failing (lg_workflows.notify).

    uv run python -m lg_workflows.deals.watch [--dry-run]
"""

import argparse
import operator
import statistics
import time
from datetime import datetime, timedelta, timezone
from typing import Annotated

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

from lg_workflows import notify
from lg_workflows.deals import agent, canadadz, db, stores
from lg_workflows.deals.config import (AGENT_STORES, BASELINE_DAYS, DROP_ALERT_PCT, MIN_BASELINE_RUNS, REALERT_HOURS,
                                       REALERT_PCT, STORE_FAILURE_ALERT_AFTER, STORES, Item, load_watchlist)

load_dotenv()

STORE_NAMES = {"bestbuy": "Best Buy", "amazon": "Amazon", "newegg": "Newegg", "canadadz": "加拿大打折网"}


class DealsState(TypedDict, total=False):
    run_ts: str
    dry_run: bool
    listings: Annotated[list[dict], operator.add]
    dz_posts: Annotated[list[dict], operator.add]
    # {store, item, errors}; agent_fallback lists the ones it recovered in `recovered` as "store/item".
    failures: Annotated[list[dict], operator.add]
    recovered: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]
    alerts: list[dict]
    sent: list[str]


class ScanInput(TypedDict):
    store: str
    run_ts: str


def fmt_price(p: float) -> str:
    return f"${p:,.2f}"


def is_candidate(item: Item, listing: dict) -> bool:
    return (listing["item"] == item.key and listing.get("in_stock", True)
            and item.matches(listing["title"], listing["price"]))


def normal_price(item: Item, history: list[float]) -> float | None:
    return statistics.median(history) if len(history) >= MIN_BASELINE_RUNS else item.normal_price


def decide(item: Item, matched: list[dict], history: list[float], last_alert: float | None) -> dict | None:
    """Price alert for this run, or None. `history` excludes this run; `last_alert` is the cheapest recent alert."""
    if not matched:
        return None
    ranked = sorted(matched, key=lambda l: l["price"])
    best = ranked[0]
    normal = normal_price(item, history)
    hit_target = item.target_price is not None and best["price"] <= item.target_price
    dropped = normal is not None and best["price"] <= normal * (1 - DROP_ALERT_PCT)
    if not (hit_target or dropped):
        return None
    if last_alert is not None and best["price"] > last_alert * (1 - REALERT_PCT):
        return None
    others, seen = [], {(best["store"], best["id"])}
    for l in ranked[1:]:
        if (l["store"], l["id"]) not in seen and len(others) < 3:
            seen.add((l["store"], l["id"]))
            others.append(l)
    return {"kind": "price", "item": item.key, "name": item.name, "best": best, "normal": normal,
            "low": min(history) if history else None, "hit_target": hit_target, "others": others}


def format_price_alert(a: dict) -> tuple[str, str]:
    best, normal = a["best"], a["normal"]
    head = f"{fmt_price(best['price'])} @ {STORE_NAMES[best['store']]}"
    if normal:
        head += f"（常价 {fmt_price(normal)}，便宜 {1 - best['price'] / normal:.0%}）"
    tags = [t for t, on in (("达到目标价", a["hit_target"]),
                            (f"{BASELINE_DAYS}天新低", a["low"] is not None and best["price"] < a["low"])) if on]
    lines = [head + (f" {'，'.join(tags)}" if tags else ""), best["title"], best["url"]]
    if best["method"] == "agent":
        lines.append("（价格由 agent 浏览网页读取，请打开链接核实）")
    elif best["method"] == "vision":
        lines.append("（页面结构变了，价格从截图识别，链接是搜索页，请核实）")
    if a["others"]:
        lines.append("\n其他低价：")
        lines += [f"- {STORE_NAMES[l['store']]} {fmt_price(l['price'])} {l['title'][:60]}\n  {l['url']}"
                  for l in a["others"]]
    return f"降价提醒：{a['name']}", "\n".join(lines)


def format_dz_alert(p: dict) -> tuple[str, str]:
    price = f" {fmt_price(p['price'])}" if p["price"] else ""
    sure = "" if p["applies"] == "yes" else "（帖子没写清型号，可能不是你要的版本）"
    return f"打折帖：{p['name']}{price}", f"{p['title']}\n{p['reason']}{sure}\n{p['url']}"


# --- nodes -----------------------------------------------------------------------------------------

def fan_out(state: DealsState):
    return [Send("scan", {"store": s, "run_ts": state["run_ts"]}) for s in STORES]


def scan(payload: ScanInput) -> dict:
    store, now = payload["store"], datetime.fromisoformat(payload["run_ts"])
    listings, posts, failures, errors = [], [], [], []
    for item in load_watchlist():
        if store != "canadadz":
            found, errs = stores.scan(store, item)
            listings += found
            if not found:
                failures.append({"store": store, "item": item.key, "errors": errs})
            elif errs:
                errors += [f"{store}/{item.key} {e}" for e in errs]
            continue
        try:
            candidates = canadadz.search(item, now)
        except Exception as e:
            failures.append({"store": store, "item": item.key, "errors": [f"rss: {type(e).__name__}: {e}"[:300]]})
            continue
        judged = db.judged_dz([p["guid"] for p in candidates])
        history = db.best_prices_since(item.key, (now - timedelta(days=BASELINE_DAYS)).isoformat())
        normal = normal_price(item, history)
        for p in candidates:
            if p["guid"] in judged:
                continue
            try:
                v = canadadz.judge(item, p, normal)
            except Exception as e:
                errors.append(f"canadadz judge {p['title'][:40]!r}: {type(e).__name__}: {str(e)[:200]}")
                continue
            posts.append({**p, "item": item.key, "name": item.name, "applies": v.applies, "price": v.price,
                          "reason": v.reason, "normal": normal})
    return {"listings": listings, "dz_posts": posts, "failures": failures, "errors": errors}


def agent_fallback(state: DealsState) -> dict:
    items = {i.key: i for i in load_watchlist()}
    by_store: dict[str, list[Item]] = {}
    for f in state.get("failures", []):
        if f["store"] in AGENT_STORES:
            by_store.setdefault(f["store"], []).append(items[f["item"]])
    if not by_store or not agent.enabled():
        return {}
    listings, recovered, errors = [], [], []
    for store, todo in by_store.items():
        try:
            found, notes = agent.scan(store, todo)
        except Exception as e:
            errors.append(f"agent {store}: {type(e).__name__}: {str(e)[:300]}")
            continue
        listings += found
        errors += [f"agent {store} {n}" for n in notes]
        recovered += [f"{store}/{i.key}" for i in todo if any(l["item"] == i.key for l in found)]
    return {"listings": listings, "recovered": recovered, "errors": errors}


def evaluate(state: DealsState) -> dict:
    now = datetime.fromisoformat(state["run_ts"])
    since = (now - timedelta(days=BASELINE_DAYS)).isoformat()
    realert_since = (now - timedelta(hours=REALERT_HOURS)).isoformat()
    alerts, all_matched = [], []
    for item in load_watchlist():
        matched = [l for l in state.get("listings", []) if is_candidate(item, l)]
        all_matched += matched
        a = decide(item, matched, db.best_prices_since(item.key, since), db.last_price_alert(item.key, realert_since))
        if a:
            alerts.append(a)
    for p in state.get("dz_posts", []):
        cheap = p["price"] is None or p["normal"] is None or p["price"] <= p["normal"] * (1 - DROP_ALERT_PCT)
        if p["applies"] != "no" and cheap:
            alerts.append({"kind": "dz", "item": p["item"], "post": p})
    db.record_run(state["run_ts"], all_matched)
    return {"alerts": alerts}


def notify_owner(state: DealsState) -> dict:
    dry = state.get("dry_run", False)
    sent = []
    for a in state.get("alerts", []):
        title, body = format_price_alert(a) if a["kind"] == "price" else format_dz_alert(a["post"])
        if dry:
            print(f"[dry-run] {title}\n{body}\n")
        else:
            notify.send(title, body)
            url = a["best"]["url"] if a["kind"] == "price" else a["post"]["url"]
            price = a["best"]["price"] if a["kind"] == "price" else a["post"]["price"]
            db.record_alert(state["run_ts"], a["item"], a["kind"], price, url, f"{title}\n{body}")
        sent.append(title)
    if dry:
        return {"sent": sent}
    for p in state.get("dz_posts", []):
        db.record_dz(state["run_ts"], p)
    recovered = set(state.get("recovered", []))
    for store in STORES:
        failed = [f for f in state.get("failures", []) if f"{f['store']}/{f['item']}" not in recovered
                  and f["store"] == store]
        name = STORE_NAMES[store]
        if failed:
            detail = "\n".join(f"{f['item']}: " + " | ".join(f["errors"]) for f in failed)
            notify.alert(f"deals-store-{store}", f"比价：{name} 连续抓取失败", detail[:1500],
                         after_failures=STORE_FAILURE_ALERT_AFTER)
        else:
            notify.resolve(f"deals-store-{store}", f"比价：{name} 已恢复", "最近一次抓取成功。")
    return {"sent": sent}


def build_graph() -> StateGraph:
    b = StateGraph(DealsState)
    b.add_node("scan", scan)
    b.add_node("agent_fallback", agent_fallback)
    b.add_node("evaluate", evaluate)
    b.add_node("notify", notify_owner)
    b.add_conditional_edges(START, fan_out, ["scan"])
    b.add_edge("scan", "agent_fallback")
    b.add_edge("agent_fallback", "evaluate")
    b.add_edge("evaluate", "notify")
    b.add_edge("notify", END)
    return b


graph = build_graph().compile()


def _summary(state: dict) -> str:
    lines = []
    for item in load_watchlist():
        matched = [l for l in state.get("listings", []) if is_candidate(item, l)]
        per_store = {s: sorted(l["price"] for l in matched if l["store"] == s) for s in STORES if s != "canadadz"}
        parts = [f"{STORE_NAMES[s]} {fmt_price(p[0])} ({len(p)})" if p else f"{STORE_NAMES[s]} -"
                 for s, p in per_store.items()]
        lines.append(f"{item.name}: " + ", ".join(parts))
    return "\n".join(lines)


def run(dry_run: bool = False) -> dict:
    t0 = time.time()
    state = graph.invoke({"run_ts": datetime.now(timezone.utc).isoformat(), "dry_run": dry_run},
                         {"max_concurrency": 4})
    print(_summary(state))
    for f in state.get("failures", []):
        status = "recovered by agent" if f"{f['store']}/{f['item']}" in state.get("recovered", []) else "FAILED"
        print(f"  {f['store']}/{f['item']} {status}: {' | '.join(f['errors'])}")
    for e in state.get("errors", []):
        print(f"  ! {e}")
    for p in state.get("dz_posts", []):
        print(f"  canadadz [{p['applies']}] {p['title'][:70]} ({p['reason']})")
    print(f"alerts: {state.get('sent') or 'none'} [{time.time() - t0:.0f}s]")
    return state


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="print alerts instead of sending or recording them")
    run(p.parse_args().dry_run)


if __name__ == "__main__":
    main()
