"""Daily digest written by the gateway model from the day's hourly event summaries.

Run:  uv run python -m lg_workflows.news.daily [--date YYYY-MM-DD] [--lang zh]
"""

import argparse
from datetime import date, datetime, time, timedelta, timezone

from lg_workflows.models import brain_llm
from lg_workflows.news import store
from lg_workflows.news.config import TOPICS
from lg_workflows.news.prompts import LABELS, PROMPTS, check_lang
from lg_workflows.news.publish import MarkdownFilePublisher, Publisher

EVENTS_PER_TOPIC = 40
PUBLISHERS: list[Publisher] = [MarkdownFilePublisher()]


def _day_bounds_utc(day: date) -> tuple[str, str]:
    start = datetime.combine(day, time.min).astimezone()
    return start.astimezone(timezone.utc).isoformat(), (start + timedelta(days=1)).astimezone(timezone.utc).isoformat()


def build_daily(day: date, lang: str = "zh") -> str:
    check_lang(lang)
    events = store.events_between(*_day_bounds_utc(day), lang)
    if not events:
        raise SystemExit(f"no hourly events stored for {day}")
    topic_name = {t.key: (t.name_zh if lang == "zh" else t.name_en) for t in TOPICS}
    lines = []
    for t in TOPICS:
        for e in [e for e in events if e["topic"] == t.key][:EVENTS_PER_TOPIC]:
            line = f"[{topic_name[t.key]}] {e['title']}：{e['summary']}"
            if e["comments_summary"]:
                line += f"（评论：{e['comments_summary']}）" if lang == "zh" else f" (Comments: {e['comments_summary']})"
            lines.append(line)
    prompt = PROMPTS[lang]["daily"].format(date=day.isoformat(), topics="、".join(topic_name.values()),
                                           events="\n".join(lines))
    body = brain_llm().invoke(prompt).text.strip()
    if body.startswith("# "):
        # The model sometimes adds its own top-level title; the digest already has one.
        body = body.split("\n", 1)[1].lstrip() if "\n" in body else ""
    title = f"{LABELS[lang]['daily_title']} {day.isoformat()}"
    digest = {"kind": "daily", "run_key": day.isoformat(), "lang": lang, "title": title,
              "markdown": f"# {title}\n\n{body}\n",
              "local_time": datetime.combine(day, time(23, 59)).astimezone().isoformat()}
    return [p.publish(digest) for p in PUBLISHERS][0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=date.fromisoformat, default=date.today())
    p.add_argument("--lang", default="zh")
    args = p.parse_args()
    print(build_daily(args.date, args.lang))


if __name__ == "__main__":
    main()
