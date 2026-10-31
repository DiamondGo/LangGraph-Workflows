from datetime import datetime, timedelta, timezone

import numpy as np

from lg_workflows import x_browser
from lg_workflows.news import x_post


def _event(title, score=3.0, topic="ai", summary="第一句话。第二句话。", src_title=None, links=None):
    ts = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    item = {"title": src_title or f"Source {title}", "summary": "English summary", "published": ts}
    return {"topic": topic, "title": title, "summary": summary, "score": score, "items": [item],
            "links": links if links is not None else [{"source": "x", "url": "https://example.com/a"}]}


def test_weighted_length_counts_cjk_double_and_urls_as_23():
    assert x_browser.weighted_length("abc") == 3
    assert x_browser.weighted_length("中文") == 4
    assert x_browser.weighted_length("看 https://example.com/a/very/long/path") == 2 + 1 + 23


def test_compose_fits_limit_and_prefers_direct_link():
    links = [{"source": "g", "url": "https://news.google.com/rss/articles/x"},
             {"source": "d", "url": "https://direct.example/story"}]
    ev = _event("标题" * 10, summary="很长的一句话" * 20 + "。" + "第二句。", links=links)
    text = x_post.compose(ev)
    assert x_browser.weighted_length(text) <= x_browser.MAX_WEIGHTED_LENGTH
    assert text.startswith("标题") and text.endswith("https://direct.example/story")


def test_compose_keeps_whole_sentences_when_they_fit():
    text = x_post.compose(_event("标题", summary="第一句。第二句。"))
    assert "第一句。第二句。" in text


def test_post_x_skips_quiet_hours_but_posts_at_2_and_6(monkeypatch):
    monkeypatch.setenv("X_POST_MODE", "live")
    attempted = []
    monkeypatch.setattr(x_post, "select", lambda summaries, anchors: attempted.append(1) or [])
    monkeypatch.setattr(x_post.store, "x_post_anchors_since", lambda since: [])
    for hour, quiet in [(2, False), (3, True), (5, True), (6, False)]:
        attempted.clear()
        out = x_post.post_x({"summaries": [], "run_key": "k", "local_time": f"2026-10-31T{hour:02d}:00:30-07:00"})
        assert bool(attempted) is not quiet
        assert ("quiet hours" in "".join(out["x_posts"])) is quiet


def test_select_skips_failed_low_and_already_posted(monkeypatch):
    vectors = {"Source A. English summary": [1.0, 0.0], "Source B. English summary": [0.0, 1.0],
               "Source C. English summary": [0.7071, 0.7071]}
    monkeypatch.setattr(x_post, "embed", lambda texts: np.array([vectors[t] for t in texts]))
    failed = _event("Source F", src_title="Source F")
    low = _event("低分", score=0.5)
    a, b, c = _event("A"), _event("B", score=2.0), _event("C", score=4.0)
    chosen = x_post.select([failed, low, a, b, c], posted_anchors=["Source A. English summary"])
    assert [e["title"] for e in chosen] == ["C", "B"]
