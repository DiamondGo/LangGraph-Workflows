"""Store scrapers. Each store is tried tier by tier until one returns product listings:

    http     plain request (Best Buy's JSON API, Amazon's HTML)
    browser  headless Chrome on the store's own profile, then the same HTML parser (Cloudflare, captchas)
    vision   a screenshot of the search page read by the local vision model (layout changed, parser broke)

A tier fails when it finds no products at all; finding products that don't match the item is a result.
The last resort, a Claude Code agent driving the browser, lives in agent.py because it is slow and metered.
A listing is {store, item, id, title, url, price, method, in_stock?}. price is None when the page shows
none, which is also how Amazon and Newegg present sold-out products; Best Buy reports stock separately.

    uv run python -m lg_workflows.deals.stores newegg vision [ITEM_KEY ...]   # try one tier alone
"""

import base64
import json
import re
import time
from datetime import datetime
from typing import Callable
from urllib.parse import quote_plus

import httpx
import lxml.html
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, field_validator

from lg_workflows.browser import VIEWPORT, chrome_page, user_agent
from lg_workflows.models import local_llm

from .config import MAX_AMAZON_DETAIL_PAGES, MAX_LISTINGS, PROFILE_DIR, SCREENSHOT_DIR, Item

LOCALE = "en-CA"
VISION_TILES = 3


class StoreError(RuntimeError):
    pass


def _headers() -> dict:
    return {"User-Agent": user_agent(), "Accept-Language": "en-CA,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8"}


def parse_price(text: str | None) -> float | None:
    m = re.search(r"\d[\d,]*(?:\.\d+)?", text or "")
    return float(m.group(0).replace(",", "")) if m else None


def _text(el) -> str:
    return re.sub(r"\s+", " ", el.text_content()).strip() if el is not None else ""


def search_url(store: str, item: Item) -> str:
    q = quote_plus(item.query)
    return {
        "bestbuy": f"https://www.bestbuy.ca/en-ca/search?search={q}",
        "amazon": f"https://www.amazon.ca/s?k={q}",
        "newegg": f"https://www.newegg.ca/p/pl?d={q}",
    }[store]


# --- parsers ---------------------------------------------------------------------------------------

def parse_bestbuy(data: dict) -> list[dict]:
    return [{"id": p["sku"], "title": p["name"], "price": p.get("salePrice"),
             "url": "https://www.bestbuy.ca" + p["productUrl"]} for p in data.get("products", [])]


def parse_bestbuy_stock(text: str) -> dict[str, bool]:
    return {a["sku"]: bool(a["shipping"].get("purchasable") or a["pickup"].get("purchasable"))
            for a in json.loads(text)["availabilities"]}


def parse_amazon_search(html: str) -> list[dict]:
    if "validateCaptcha" in html or "Type the characters you see in this image" in html:
        raise StoreError("Amazon served a captcha")
    out = []
    for c in lxml.html.fromstring(html).xpath('//div[@data-component-type="s-search-result"][@data-asin!=""]'):
        h2 = c.xpath(".//h2")
        title = (h2[0].get("aria-label") or _text(h2[0])) if h2 else ""
        price = c.xpath('.//span[contains(@class,"a-price") and not(contains(@class,"a-text-price"))]'
                        '/span[@class="a-offscreen"]/text()')
        asin = c.get("data-asin")
        out.append({"id": asin, "title": title, "price": parse_price(price[0]) if price else None,
                    "url": f"https://www.amazon.ca/dp/{asin}"})
    return out


def parse_amazon_product(html: str) -> float | None:
    """Price of the featured new offer, or None (no offer, or only used ones)."""
    doc = lxml.html.fromstring(html)
    if doc.xpath('//*[@id="usedOnlyBuybox"]'):
        return None
    for box in doc.xpath('//*[@id="corePriceDisplay_desktop_feature_div" or @id="corePrice_feature_div"]'
                         '//span[contains(@class,"a-price") and not(contains(@class,"a-text-price"))]'):
        price = parse_price("".join(box.xpath('./span[@class="a-offscreen"]/text()')))
        if price is None:
            whole = "".join(box.xpath('.//span[@class="a-price-whole"]//text()')).strip(".")
            frac = "".join(box.xpath('.//span[@class="a-price-fraction"]//text()'))
            price = parse_price(f"{whole}.{frac}" if whole else "")
        if price:
            return price
    return None


def parse_newegg_search(html: str) -> list[dict]:
    if "<title>Just a moment" in html:
        raise StoreError("Newegg Cloudflare check did not clear")
    out = []
    cards = lxml.html.fromstring(html).xpath(
        '//div[contains(@class,"plp-results-section")]//div[@data-itemnumber]'
        '[contains(concat(" ",normalize-space(@class)," ")," goods-container ")]')
    for c in cards:
        a = c.xpath('.//a[contains(@class,"goods-title")]')
        price = c.xpath('.//div[contains(@class,"goods-price-current")]//span[contains(@class,"goods-price-value")]')
        if not a:
            continue
        out.append({"id": c.get("data-itemnumber"), "title": _text(a[0]), "url": a[0].get("href"),
                    "price": parse_price(_text(price[0])) if price else None})
    return out


# --- tiers -----------------------------------------------------------------------------------------

def _bestbuy_http(item: Item) -> list[dict]:
    r = httpx.get("https://www.bestbuy.ca/api/v2/json/search", headers=_headers(), timeout=30,
                  params={"query": item.query, "lang": LOCALE, "pageSize": MAX_LISTINGS, "page": 1})
    r.raise_for_status()
    listings = parse_bestbuy(r.json())
    # Search keeps showing sold-out SKUs at their old (often launch) price.
    skus = [l["id"] for l in listings if item.title_matches(l["title"])]
    if skus:
        a = httpx.get("https://www.bestbuy.ca/ecomm-api/availability/products", headers=_headers(), timeout=30,
                      params={"skus": "|".join(skus), "accept-language": LOCALE})
        a.raise_for_status()
        stock = parse_bestbuy_stock(a.content.decode("utf-8-sig"))
        for l in listings:
            l["in_stock"] = stock.get(l["id"], False)
    return listings


def _amazon_details(item: Item, listings: list[dict], fetch: Callable[[str], str]) -> None:
    """Search results often hide the price of the exact item; its product page has it."""
    todo = [l for l in listings if l["price"] is None and item.title_matches(l["title"])]
    for l in todo[:MAX_AMAZON_DETAIL_PAGES]:
        l["price"] = parse_amazon_product(fetch(l["url"]))


def _amazon_http(item: Item) -> list[dict]:
    with httpx.Client(headers=_headers(), timeout=30, follow_redirects=True) as client:
        def fetch(url: str) -> str:
            r = client.get(url)
            r.raise_for_status()
            return r.text
        listings = parse_amazon_search(fetch(search_url("amazon", item)))
        if listings:
            _amazon_details(item, listings, fetch)
        return listings


def _browser_html(page, url: str, ready: str, settle: float = 2.0) -> str:
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    try:
        page.wait_for_selector(ready, timeout=45_000)
    except Exception:
        pass  # the parser reports what the page actually is
    time.sleep(settle)
    return page.content()


def _amazon_browser(item: Item) -> list[dict]:
    with chrome_page(PROFILE_DIR / "amazon", locale=LOCALE) as page:
        listings = parse_amazon_search(_browser_html(page, search_url("amazon", item),
                                                     '[data-component-type="s-search-result"]'))
        if listings:
            _amazon_details(item, listings, lambda url: _browser_html(page, url, "#productTitle", 1.0))
        return listings


def _newegg_browser(item: Item) -> list[dict]:
    url, ready = search_url("newegg", item), ".plp-results-section .goods-container"
    with chrome_page(PROFILE_DIR / "newegg", locale=LOCALE) as page:
        # Cloudflare clears on the first visit and leaves a cookie in the profile; the page it lands on
        # can be an empty shell, so load once more before giving up.
        html = _browser_html(page, url, ready)
        try:
            if listings := parse_newegg_search(html):
                return listings
        except StoreError:
            pass
        return parse_newegg_search(_browser_html(page, url, ready, settle=4.0))


class _SeenListing(BaseModel):
    title: str
    price: float | None = None


class _SeenPage(BaseModel):
    listings: list[_SeenListing]

    @field_validator("listings", mode="before")
    @classmethod
    def _decode_json_string(cls, v):
        # Qwen sometimes emits the array as a JSON-encoded string.
        return json.loads(v) if isinstance(v, str) else v


VISION_PROMPT = """This is a screenshot of a {store} search results page for "{query}".
List every product listing you can read, top to bottom: its full title as shown, and its current
price in CAD as a number (the price you would pay, not a crossed-out one). Use null when no price is shown.
Skip ads for unrelated products only if they have no title. Return an empty list if the page shows
no products (for example a captcha, an error or a "verify you are human" page)."""


GROUND_ANCHOR_WORDS = (8, 6, 5, 4)
GROUND_PRICE_LINES = 15
_SOLD_OUT = re.compile(r"out of stock|sold out|currently unavailable|no featured offers|see options", re.I)


def _words(s: str) -> str:
    return " " + " ".join(re.findall(r"[a-z0-9]+", s.lower())) + " "


def ground_listing(title: str, price: float | None, text: str) -> tuple[str, float | None] | None:
    """Re-read a screenshot listing from the page text: (title, price) as the page has them, or None.

    The vision model only locates listings. It can invent them, give one product another's price, or misread
    a title ("5070 12GB" as "5070 Ti 16GB"), so the title is the page line its first words point to and the
    price is the first one after that line. The model's price must agree with it, allowing for dropped cents
    (stores print cents as small superscript)."""
    lines = text.splitlines()
    words = _words(title).split()
    for n in GROUND_ANCHOR_WORDS:
        if len(words) < n:
            continue
        key = " " + " ".join(words[:n]) + " "
        for i, line in enumerate(lines):
            if key not in _words(line):
                continue
            following = "\n".join(lines[i + 1:i + 1 + GROUND_PRICE_LINES])
            # A sold-out card shows no price of its own; what follows is its old price or the next card's.
            following = _SOLD_OUT.split(following, maxsplit=1)[0]
            m = re.search(r"\$(\d+(?:\.\d{2})?)(?!\d)", re.sub(r"[\s,]", "", following))
            page_price = float(m.group(1)) if m else None
            if price is None or page_price is None:
                return line.strip(), None
            if page_price == price or (price == int(price) and int(page_price) == int(price)):
                return line.strip(), page_price
    return None


def vision(store: str, item: Item) -> list[dict]:
    """Read listings from screenshots, then confirm each against the page text (see ground_listing), which is
    still there when the layout has changed beyond what the parser knows."""
    url = search_url(store, item)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{store}-{item.key}-{datetime.now():%Y%m%dT%H%M}"
    shots = []
    with chrome_page(PROFILE_DIR / store, locale=LOCALE) as page:
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        time.sleep(8)
        text = page.inner_text("body")
        # Viewport-sized tiles: one tall image is downscaled until prices are unreadable.
        height = page.evaluate("document.documentElement.scrollHeight")
        for n in range(min(VISION_TILES, max(1, height // VIEWPORT["height"]))):
            shot = SCREENSHOT_DIR / f"{stamp}-{n}.png"
            page.screenshot(path=str(shot), full_page=True, clip={"x": 0, "y": n * VIEWPORT["height"],
                                                                  **VIEWPORT})
            shots.append(shot)
    model = local_llm(thinking=False).with_structured_output(_SeenPage, method="function_calling")
    seen: dict[tuple[str, float | None], _SeenListing] = {}
    errors = []
    for shot in shots:
        img = base64.b64encode(shot.read_bytes()).decode()
        msg = HumanMessage(content=[
            {"type": "text", "text": VISION_PROMPT.format(store=store, query=item.query)},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
        ])
        try:
            page_listings = model.invoke([msg]).listings
        except Exception as e:
            errors.append(f"{shot.name}: {type(e).__name__}: {str(e)[:150]}")
            continue
        for l in page_listings:
            seen.setdefault((l.title, l.price), l)
    if errors and not seen:
        raise StoreError("vision model failed on every tile: " + " | ".join(errors))
    out: dict[str, dict] = {}
    for l in seen.values():
        if grounded := ground_listing(l.title, l.price, text):
            title, price = grounded
            out.setdefault(f"{title}|{price}", {"id": f"vision:{title[:80]}", "title": title, "url": url,
                                                "price": price})
    return list(out.values())


TIERS: dict[str, list[tuple[str, Callable[[Item], list[dict]]]]] = {
    "bestbuy": [("http", _bestbuy_http), ("vision", lambda i: vision("bestbuy", i))],
    "amazon": [("http", _amazon_http), ("browser", _amazon_browser), ("vision", lambda i: vision("amazon", i))],
    "newegg": [("browser", _newegg_browser), ("vision", lambda i: vision("newegg", i))],
}


def scan(store: str, item: Item) -> tuple[list[dict], list[str]]:
    """Return (listings, errors of the tiers tried). No listings means every tier failed."""
    errors = []
    for method, fn in TIERS[store]:
        try:
            listings = fn(item)[:MAX_LISTINGS]
        except Exception as e:
            errors.append(f"{method}: {type(e).__name__}: {str(e)[:200]}")
            continue
        if listings:
            return [{**l, "store": store, "item": item.key, "method": method} for l in listings], errors
        errors.append(f"{method}: no products on the page")
    return [], errors


def main() -> None:
    import sys

    from .config import load_watchlist
    if len(sys.argv) < 3:
        sys.exit("usage: python -m lg_workflows.deals.stores STORE TIER [ITEM_KEY ...]   (tiers: "
                 + "; ".join(f"{s}: {', '.join(t for t, _ in tiers)}" for s, tiers in TIERS.items()) + ")")
    store, tier = sys.argv[1], sys.argv[2]
    fn = dict(TIERS[store])[tier]
    for item in load_watchlist():
        if len(sys.argv) > 3 and item.key not in sys.argv[3:]:
            continue
        listings = fn(item)
        matched = sorted((l for l in listings if item.matches(l["title"], l["price"])), key=lambda l: l["price"])
        print(f"{store}/{tier} {item.key}: {len(listings)} products, {len(matched)} match")
        for l in matched[:5]:
            print(f"  ${l['price']:,.2f} {l['title'][:80]}")


if __name__ == "__main__":
    main()
