"""Last-resort scraping: a Claude Code agent drives the store's headless Chrome through Playwright MCP.

Used only for (store, item) pairs whose scripted tiers all failed, one agent session per store.
Its listings go through the same title/price rules as scraped ones. DEALS_AGENT_FALLBACK=off disables it.

    uv run python -m lg_workflows.deals.agent newegg [ITEM_KEY ...]   # run the agent alone, print listings
"""

import json
import os
import sys

from lg_workflows.browser import mcp_config
from lg_workflows.cli_agents import run_cli_agent

from .config import DATA_DIR, PROFILE_DIR, Item
from .stores import search_url

AGENT_TIMEOUT = 600

PROMPT = """You are checking prices on {store} (Canada, prices in CAD) with the browser tools.
Scripted scraping of this site just failed, so its pages may have changed or be hard to read.

For each watched item below:
1. Open its search page. Wait out any "Just a moment" or loading screen and dismiss cookie or location popups.
   If the search page does not work, open the site's home page, find the search box and search for the item.
2. Collect up to 8 of the cheapest listings that are this exact item in NEW condition and in stock, with the
   price you would pay today and the product URL.

Work visually whenever the page snapshot is unclear: take a screenshot, decide from it where the search box,
buttons, prices and product links are, and click or type at those coordinates. Read prices from the screenshot
if the snapshot does not contain them, and open a product to get its URL if needed.

Do not sign in, add to cart, solve captchas or buy anything. If a page is blocked or the item is not sold,
return no listings for it and say why in "note".

{items}"""

SCHEMA = {
    "type": "object",
    "properties": {"results": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "item_key": {"type": "string"},
            "note": {"type": "string"},
            "listings": {"type": "array", "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}, "price": {"type": "number"}, "url": {"type": "string"}},
                "required": ["title", "price", "url"]}},
        },
        "required": ["item_key", "listings"]}}},
    "required": ["results"],
}


def enabled() -> bool:
    return os.environ.get("DEALS_AGENT_FALLBACK", "claude").lower() != "off"


def scan(store: str, items: list[Item]) -> tuple[list[dict], list[str]]:
    """Return (listings, notes) for `items` on `store`."""
    lines = "\n".join(f"- item_key={i.key}: {i.name}. Search page: {search_url(store, i)}" for i in items)
    workdir = DATA_DIR / "agent_work"
    cfg = mcp_config(PROFILE_DIR / store, workdir / f"mcp-{store}.json")
    res = run_cli_agent("claude", PROMPT.format(store=store, items=lines), cwd=str(workdir), mode="browse",
                        mcp_config=str(cfg), schema=SCHEMA, timeout=AGENT_TIMEOUT)
    data = res.result if isinstance(res.result, dict) else json.loads(res.result)
    keys = {i.key for i in items}
    listings, notes = [], []
    for r in data.get("results", []):
        if r.get("item_key") not in keys:
            continue
        if r.get("note"):
            notes.append(f"{r['item_key']}: {r['note']}")
        listings += [{"id": l["url"], "title": l["title"], "price": l["price"], "url": l["url"],
                      "store": store, "item": r["item_key"], "method": "agent"} for l in r.get("listings", [])]
    return listings, notes


def main() -> None:
    from .config import load_watchlist
    if len(sys.argv) < 2:
        sys.exit("usage: python -m lg_workflows.deals.agent STORE [ITEM_KEY ...]")
    items = [i for i in load_watchlist() if len(sys.argv) == 2 or i.key in sys.argv[2:]]
    listings, notes = scan(sys.argv[1], items)
    by_key = {i.key: i for i in items}
    for l in sorted(listings, key=lambda l: (l["item"], l["price"])):
        ok = "match" if by_key[l["item"]].matches(l["title"], l["price"]) else "     "
        print(f"{ok} {l['item']} ${l['price']:,.2f} {l['title'][:70]}\n      {l['url']}")
    for n in notes:
        print(f"note: {n}")


if __name__ == "__main__":
    main()
