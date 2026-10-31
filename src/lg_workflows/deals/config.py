"""Watchlist loading, title matching and tunables for the deals workflow."""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DB_PATH = DATA_DIR / "deals.db"
PROFILE_DIR = DATA_DIR / "shop_chrome"
SCREENSHOT_DIR = DATA_DIR / "deals_screenshots"
WATCHLIST_PATH = Path(__file__).with_name("watchlist.toml")

STORES = ("bestbuy", "amazon", "newegg", "canadadz")
# Listing-page stores whose failures fall back to a Claude Code agent driving the store's browser profile.
AGENT_STORES = ("bestbuy", "amazon", "newegg")

# "Normal price" is the median of each run's best price over this window, once there are enough runs.
BASELINE_DAYS = 30
MIN_BASELINE_RUNS = 4
# Alert when the best price is at least this far below normal...
DROP_ALERT_PCT = 0.05
# ...and, within REALERT_HOURS of an alert for the item, only if at least REALERT_PCT cheaper than it.
REALERT_HOURS = 72
REALERT_PCT = 0.01
# Listings scraped per store and item; Amazon product pages fetched when the search shows no price.
MAX_LISTINGS = 40
MAX_AMAZON_DETAIL_PAGES = 4
# canadadz posts older than this are history, not news.
DZ_MAX_AGE_DAYS = 3
# Consecutive failed runs of a store before an alert.
STORE_FAILURE_ALERT_AFTER = 3


@dataclass(frozen=True)
class Item:
    key: str
    name: str
    query: str
    dz_query: str
    require: tuple[tuple[str, ...], ...]
    exclude: tuple[str, ...]
    min_price: float
    max_price: float
    normal_price: float | None = None
    target_price: float | None = None

    def title_matches(self, title: str) -> bool:
        t = norm_text(title)
        return all(any(has_phrase(t, p) for p in group) for group in self.require) \
            and not any(has_phrase(t, p) for p in self.exclude)

    def matches(self, title: str, price: float | None) -> bool:
        return price is not None and self.min_price <= price <= self.max_price and self.title_matches(title)


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def has_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(norm_text(phrase))}(?![a-z0-9])", text) is not None


def load_watchlist(path: Path = WATCHLIST_PATH) -> list[Item]:
    raw = tomllib.loads(path.read_text())["item"]
    return [Item(**{**r, "require": tuple(tuple(g) for g in r["require"]), "exclude": tuple(r.get("exclude", ())),
                    "dz_query": r.get("dz_query", r["query"])}) for r in raw]
