"""Topics, feeds and tunables for the hourly news digest."""

from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DB_PATH = DATA_DIR / "news.db"
CHECKPOINT_DB = DATA_DIR / "news_checkpoints.db"
DIGEST_DIR = DATA_DIR / "digests"

EVENTS_PER_TOPIC = 10
# A fresh single report from a major outlet scores ~1.0, from a minor one ~0.3.
# Topics with fewer qualifying events are topped up with the best weak ones, up to MIN_EVENTS_PER_TOPIC.
MIN_EVENT_SCORE = 0.5
MIN_EVENTS_PER_TOPIC = 3
# Posting the hottest events to X (see lg_workflows/x_browser.py). Mode comes from X_POST_MODE in .env:
# "off" (default), "dry_run" (fill the compose box and screenshot, never post) or "live".
# Events scoring >= X_MIN_POST_SCORE always qualify, up to X_MAX_POSTS_PER_RUN. If fewer than
# X_TARGET_POSTS_PER_RUN qualify, the best remaining events down to X_FILL_MIN_SCORE fill the gap.
X_MAX_POSTS_PER_RUN = 5
X_TARGET_POSTS_PER_RUN = 3
X_MIN_POST_SCORE = 1.5
X_FILL_MIN_SCORE = 0.8
X_POST_GAP_SECONDS = (30, 90)
# Local hours of the hourly run that never post: 02:00 is the last run that posts, 06:00 the first.
X_QUIET_HOURS = frozenset({3, 4, 5})
# An event this similar to anything posted in the last X_REPOST_WINDOW_HOURS counts as already posted.
X_REPOST_WINDOW_HOURS = 24
X_SCREENSHOT_DIR = DATA_DIR / "x_screenshots"
# Replying to comments on our posts (see lg_workflows/news/x_reply.py). Mode comes from X_REPLY_MODE in .env:
# "off" (default), "dry_run" (draft and fill the reply box, never send) or "live".
# Local hours with no replies: midnight to 06:00. The systemd timer does not fire then either.
X_REPLY_QUIET_HOURS = frozenset(range(0, 6))
X_REPLY_MAX_PER_RUN = 5
X_REPLY_GAP_SECONDS = (30, 90)
# Comments older than this are marked handled without a reply: late replies look odd.
X_REPLY_MAX_AGE_HOURS = 24
# At most this many replies to one person under the same post. Comments under different posts never count together.
X_REPLY_MAX_PER_AUTHOR_PER_POST = 2
X_REPLY_IGNORE_HANDLES = frozenset({"grok"})
# Replies are one short sentence, far under the 280 limit (a Chinese character weighs 2: 50 characters).
X_REPLY_MAX_WEIGHT = 100
# Only consider items published within this window; older unseen items are ignored.
LOOKBACK_HOURS = 3
# HN discussions take hours to mature, and HN is the only comment source.
HN_LOOKBACK_HOURS = 24
# Freshness half-life used in ranking.
FRESHNESS_HALF_LIFE_HOURS = 6
MAX_HN_COMMENTS = 60
MAX_ARTICLE_CHARS = 6000

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# Average cosine similarity for two clusters to count as the same event (tuned on real feeds).
CLUSTER_THRESHOLD = 0.77

# Outlets that count fully toward an event's heat; everything else counts MINOR_SOURCE_WEIGHT.
MAJOR_DOMAINS = (
    "apnews.com", "reuters.com", "bloomberg.com", "wsj.com", "nytimes.com", "washingtonpost.com",
    "cnn.com", "nbcnews.com", "cbsnews.com", "abcnews.go.com", "foxnews.com", "npr.org", "pbs.org",
    "politico.com", "thehill.com", "axios.com", "usatoday.com", "latimes.com", "nypost.com",
    "newsweek.com", "time.com", "theatlantic.com", "semafor.com", "ft.com", "economist.com",
    "bbc.com", "bbc.co.uk", "theguardian.com", "aljazeera.com", "cnbc.com", "france24.com", "dw.com",
    "scmp.com", "straitstimes.com", "japantimes.co.jp", "nikkei.com", "rfa.org", "voanews.com",
    "techcrunch.com", "theverge.com", "arstechnica.com", "technologyreview.com", "wired.com",
    "venturebeat.com", "theinformation.com", "engadget.com", "404media.co",
)
MINOR_SOURCE_WEIGHT = 0.3

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) lg-workflows-news/0.1 (personal digest)"

# Chinese state media (English editions included). Applied to every source.
BLOCKED_DOMAINS = (
    "chinadaily.com.cn", "globaltimes.cn", "cgtn.com", "xinhuanet.com", "news.cn",
    "people.cn", "peopledaily.com.cn", "ecns.cn", "china.org.cn", "cctv.com", "cnr.cn",
    "chinanews.com.cn", "gov.cn", "chinadailyhk.com", "chinadailyasia.com",
)


@dataclass(frozen=True)
class Topic:
    key: str
    name_zh: str
    name_en: str
    feeds: tuple[tuple[str, str], ...]  # (source name, RSS url)


def _gnews(query: str) -> str:
    return f"https://news.google.com/rss/search?q={query}+when:2h&hl=en-US&gl=US&ceid=US:en"


TOPICS: tuple[Topic, ...] = (
    Topic("us_politics", "美国政治", "US Politics", (
        ("NPR", "https://feeds.npr.org/1014/rss.xml"),
        ("BBC", "https://feeds.bbci.co.uk/news/world/us_and_canada/rss.xml"),
        ("The Guardian", "https://www.theguardian.com/us-news/us-politics/rss"),
        ("Politico", "https://rss.politico.com/politics-news.xml"),
        ("The Hill", "https://thehill.com/homenews/feed/"),
        ("Google News", _gnews("US+politics")),
    )),
    Topic("china_politics", "中国政治", "China Politics", (
        ("SCMP", "https://www.scmp.com/rss/4/feed"),
        ("BBC", "https://feeds.bbci.co.uk/news/world/asia/china/rss.xml"),
        ("The Guardian", "https://www.theguardian.com/world/china/rss"),
        ("NYT", "https://rss.nytimes.com/services/xml/rss/nyt/AsiaPacific.xml"),
        ("Google News", _gnews("China+politics+OR+Beijing+government")),
    )),
    Topic("ai", "AI", "AI", (
        ("TechCrunch", "https://techcrunch.com/category/artificial-intelligence/feed/"),
        ("The Verge", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
        ("Ars Technica", "https://arstechnica.com/ai/feed/"),
        ("MIT Technology Review", "https://www.technologyreview.com/topic/artificial-intelligence/feed"),
        ("Google News", _gnews("artificial+intelligence")),
    )),
)

TOPIC_BY_KEY = {t.key: t for t in TOPICS}
