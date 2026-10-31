from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from lg_workflows.news import hourly
from lg_workflows.news.clustering import average_linkage
from lg_workflows.news.prompts import check_lang
from lg_workflows.news.publish import MarkdownFilePublisher, render_hourly
from lg_workflows.news.sources import is_blocked, item_id, normalize_url


def _item(url, source="NPR", origin="rss", topic="us_politics", minutes_ago=10, **kw):
    ts = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    return {"id": item_id(url), "url": url, "title": f"t {url}", "source": source, "domain": url.split("/")[2],
            "topic": topic, "published": ts.isoformat(), "summary": "s", "origin": origin,
            "hn_id": None, "hn_points": 0, "hn_comments": 0, **kw}


def test_normalize_url_strips_tracking_and_www():
    assert normalize_url("https://WWW.Example.com/a/?utm_source=x&id=3#frag") == "https://example.com/a?id=3"
    assert item_id("https://example.com/a?utm_medium=rss") == item_id("https://www.example.com/a/")


def test_blocklist_matches_subdomains_only_on_boundary():
    assert is_blocked("global.chinadaily.com.cn")
    assert is_blocked("globaltimes.cn")
    assert not is_blocked("notchinadaily.com.cn.example.org")
    assert not is_blocked("scmp.com")


def test_merge_folds_hn_stats_into_rss_item():
    rss = _item("https://npr.org/x")
    hn = _item("https://www.npr.org/x/?utm_source=hn", source="Hacker News", origin="hn", topic=None,
               hn_id="42", hn_points=300, hn_comments=120)
    [merged] = hourly._merge_duplicates([hn, rss])
    assert merged["origin"] == "rss" and merged["topic"] == "us_politics"
    assert (merged["hn_id"], merged["hn_comments"]) == ("42", 120)


def test_event_score_prefers_more_sources_and_fresher():
    now = datetime.now(timezone.utc)
    wide = hourly._to_event("ai", [_item("https://a.com/1", "A"), _item("https://b.com/1", "B")], now)
    single = hourly._to_event("ai", [_item("https://c.com/1", "C")], now)
    stale = hourly._to_event("ai", [_item("https://d.com/1", "D", minutes_ago=600),
                                    _item("https://e.com/1", "E", minutes_ago=600)], now)
    assert wide["score"] > single["score"]
    assert wide["score"] > stale["score"]
    assert wide["sources"] == ["A", "B"]


def test_topic_labels_accept_json_string():
    parsed = hourly.TopicLabels.model_validate({"labels": '[{"idx": 0, "topic": "ai"}]'})
    assert parsed.labels[0].topic == "ai"


def test_major_outlets_outweigh_minor_ones():
    now = datetime.now(timezone.utc)
    major = hourly._to_event("ai", [_item("https://www.reuters.com/x", "Reuters")], now)
    minor = hourly._to_event("ai", [_item("https://smallblog.example/x", "Blog"),
                                    _item("https://other.example/x", "Other")], now)
    assert major["heat"] == pytest.approx(1.0) and minor["heat"] == pytest.approx(0.6)


def test_average_linkage_groups_similar_vectors_only():
    v = np.array([[1, 0, 0], [0.98, 0.2, 0], [0, 1, 0], [0, 0.97, 0.24], [0, 0, 1]], dtype=float)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    groups = sorted(sorted(g) for g in average_linkage(v, threshold=0.9))
    assert groups == [[0, 1], [2, 3], [4]]


def test_cluster_rank_tops_up_weak_topics_to_floor(monkeypatch):
    monkeypatch.setattr(hourly, "group_items", lambda items: [[i] for i in items])
    strong = [_item("https://npr.org/a")]
    weak = [_item(f"https://blog{n}.example/x", f"Blog{n}", minutes_ago=10 + n) for n in range(5)]
    c = hourly.cluster_rank({"items": strong + weak})["candidates"]
    assert [e["items"][0]["source"] for e in c] == ["NPR", "Blog0", "Blog1"]


def test_event_links_prefer_direct_urls_and_add_hn():
    now = datetime.now(timezone.utc)
    ev = hourly._to_event("ai", [
        _item("https://news.google.com/rss/articles/abc", "Google News"),
        _item("https://site.com/story", "Site", hn_id="7", hn_comments=5),
    ], now)
    assert ev["links"][0]["url"] == "https://site.com/story"
    assert ev["links"][-1]["url"].endswith("id=7")


class _FakeLLM:
    def __init__(self, thinking):
        self.thinking = thinking

    def with_structured_output(self, *a, **kw):
        return self

    def invoke(self, prompt):
        if self.thinking:
            raise TimeoutError("slow server")
        return hourly.EventSummary(title="标题", summary="摘要")


def test_summarize_retries_without_thinking_and_reports_errors(monkeypatch):
    monkeypatch.setattr(hourly, "local_llm", lambda thinking=True: _FakeLLM(thinking))
    monkeypatch.setattr(hourly, "fetch_article_text", lambda url: "")
    now = datetime.now(timezone.utc)
    ev = hourly._to_event("us_politics", [_item("https://npr.org/x")], now)
    out = hourly.summarize({"event": ev, "lang": "zh"})
    assert out["summaries"][0]["title"] == "标题"
    assert len(out["errors"]) == 1 and "TimeoutError" in out["errors"][0]


def test_parallel_summarize_errors_accumulate(monkeypatch):
    monkeypatch.setattr(hourly, "local_llm", lambda thinking=True: _FakeLLM(thinking))
    monkeypatch.setattr(hourly, "fetch_article_text", lambda url: "")
    now = datetime.now(timezone.utc)
    events = [hourly._to_event("us_politics", [_item(f"https://npr.org/{n}")], now) for n in range(3)]
    monkeypatch.setattr(hourly, "fetch", lambda s: {"items": [], "errors": ["feed down"]})
    monkeypatch.setattr(hourly, "classify", lambda s: {"items": []})
    monkeypatch.setattr(hourly, "cluster_rank", lambda s: {"candidates": events})
    monkeypatch.setattr(hourly, "render", lambda s: {"digest_path": "x"})
    monkeypatch.setattr(hourly, "post_x", lambda s: {})
    final = hourly.build_graph().compile().invoke({"run_key": "k", "local_time": now.isoformat(), "lang": "zh"})
    assert len(final["summaries"]) == 3
    assert final["errors"][0] == "feed down" and len(final["errors"]) == 4


def test_render_and_publish(tmp_path):
    now = datetime.now().astimezone()
    ev = hourly._to_event("china_politics", [_item("https://scmp.com/1", "SCMP")], datetime.now(timezone.utc))
    ev.update(title="标题", summary="摘要", comments_summary="观点")
    title, md = render_hourly(now, "zh", [ev], ["X: boom"])
    assert "## 中国政治" in md and "### 1. 标题" in md and "观点" in md and "X: boom" in md
    assert "## 美国政治" in md
    path = MarkdownFilePublisher(tmp_path).publish(
        {"kind": "hourly", "run_key": "k", "lang": "zh", "title": title, "markdown": md, "local_time": now.isoformat()})
    assert path.endswith(f"{now:%H}00.md")


def test_english_not_implemented_yet():
    check_lang("zh")
    with pytest.raises(NotImplementedError):
        check_lang("en")
