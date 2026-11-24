from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import os

import pytest
from dotenv import load_dotenv

from lg_workflows import x_browser
from lg_workflows.news import store, x_reply

load_dotenv()
# The handle of the account the tests pretend to run as. It comes from the git-ignored .env, never from the repo.
ME = os.environ.get("X_ACCOUNT_HANDLE", "example_account").lower()


@pytest.fixture(autouse=True)
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "news.db")
    monkeypatch.setattr(x_reply.notify, "alert", lambda *a, **k: True)
    monkeypatch.setattr(x_reply.notify, "resolve", lambda *a, **k: True)
    monkeypatch.setattr(x_reply.time, "sleep", lambda s: None)


def _mention(id="1", handle="fan", age_h=2, replying_to=(ME,), text="很有意思"):
    ts = (datetime.now(timezone.utc) - timedelta(hours=age_h)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    return {"id": id, "handle": handle, "url": f"https://x.com/{handle}/status/{id}", "posted_at": ts,
            "text": text, "replying_to": list(replying_to)}


def _tweet(handle, text="原帖"):
    return {"handle": handle, "text": text}


def test_skip_reasons_before_opening_the_thread():
    now = datetime.now(timezone.utc)
    assert x_reply._skip_reason(_mention(), ME, now) is None
    assert x_reply._skip_reason(_mention(handle=ME), ME, now) == "own comment"
    assert x_reply._skip_reason(_mention(handle="Grok"), ME, now) == "ignored account"
    assert x_reply._skip_reason(_mention(replying_to=("someone",)), ME, now) == "not a reply to us"
    assert "older than" in x_reply._skip_reason(_mention(age_h=30), ME, now)


def test_cap_counts_replies_to_one_person_under_one_post_only():
    def thread(post):
        return {**_mention(), "chain": [_tweet(ME, post)]}
    for i in range(x_reply.X_REPLY_MAX_PER_AUTHOR_PER_POST):
        store.record_x_reply("replied", "live", {**thread("新闻A"), "id": f"r{i}"}, reply_text="hi")
    assert "already replied" in x_reply._post_cap_reason(thread("新闻A"))
    assert x_reply._post_cap_reason(thread("新闻B")) is None          # another news post: not limited
    assert x_reply._post_cap_reason({**thread("新闻A"), "handle": "other"}) is None


def test_only_direct_comments_on_our_top_level_post():
    skip = x_reply._thread_skip_reason
    assert skip([_tweet(ME)], ME) is None
    assert skip([], ME) == "no parent tweet visible"
    assert skip([_tweet("other")], ME) == "parent tweet is not ours"
    assert skip([_tweet("other"), _tweet(ME)], ME) == "comment on a comment"
    assert skip([_tweet(ME), _tweet("fan"), _tweet(ME)], ME) == "comment on a comment"


POST = "英伟达停止向中国出口H200，Colette Kress称本季对华销售24亿美元，三成营收受影响。"


@pytest.mark.parametrize("text,comment,problem", [
    ("谢谢补充，这个时间点挺关键。", "停的是往后", None),
    ("Thanks for the detail on the timeline.", "The halt only covers future shipments.", None),
    ("", "好", "empty"),
    ("看这里 https://x.com/a", "好", "link"), ("@fan 谢谢", "好", "link"), ("好 #AI", "好", "link"),
    ("后面再补上", "好", "promise"), ("我们会跟进", "好", "promise"), ("We will add that later", "good", "promise"),
    ("确实是这样", "好", "filler"), ("对，没错", "好", "filler"), ("说得对", "好", "filler"),
    ("Exactly, that matters.", "good point", "filler"), ("是的，谢谢", "好", "filler"),
    ("很长" * 40, "好", "too long"),
    ("Thanks, that is a useful point.", "停的是往后", "language"),    # Chinese comment, English reply
    ("谢谢补充，这点很关键。", "Great summary, thanks", "language"),   # English comment, Chinese reply
    ("ありがとうございます", "いい記事ですね", None),
    ("谢谢，24亿这个数字挺关键。", "好", None),
    ("谢谢，13亿这个数字挺关键。", "好", "numbers"),
    ("谢谢，Kress 这点提得好。", "好", None),
    ("谢谢，24亿这块讲清楚了。", "好", None),
    ("谢谢补充，这是个很实用的点。", "好", None),                      # a bare 一/个 is ordinary wording
    ("谢谢，这七个国家的情况各不相同。", "好", "numbers"),             # "七个" is not in the post
    ("谢谢，原帖说的掉了七成。", "好", "numbers"),
    ("谢谢，三成这个数字很关键。", "好", None),                     # unit-bearing numeral not in this post
    ("你觉得呢？", "好", "question"), ("What do you think?", "good", "question"),
    ("谢谢，据 Chosunbiz 报道。", "好", "names"),
])
def test_validate_reply(text, comment, problem):
    result = x_reply.validate_reply(text, comment=comment, post=POST)
    assert (result is None) if problem is None else (problem in result)


def test_make_reply_retries_once_with_the_reason_then_gives_up(monkeypatch):
    seen = []
    answers = iter(["确实如此", "谢谢补充，这点很关键。"])

    def write(me, c, feedback=None):
        seen.append(feedback)
        return x_reply.ReplyDraft(should_reply=True, reply=next(answers))
    monkeypatch.setattr(x_reply, "write_reply", write)
    c = {**_mention(text="很有意思"), "chain": [_tweet(ME, POST)]}
    assert x_reply.make_reply(ME, c) == ("谢谢补充，这点很关键。", None)
    assert seen[0] is None and "filler" in seen[1]

    monkeypatch.setattr(x_reply, "write_reply", lambda me, c, feedback=None:
                        x_reply.ReplyDraft(should_reply=True, reply="确实如此"))
    text, problem = x_reply.make_reply(ME, c)
    assert text is None and "filler" in problem

    monkeypatch.setattr(x_reply, "write_reply", lambda me, c, feedback=None:
                        x_reply.ReplyDraft(should_reply=False, reason="political argument"))
    assert x_reply.make_reply(ME, c) == (None, "model declined: political argument")


class _FakeSession:
    def __init__(self, mentions, threads, fail=None):
        self._mentions, self._threads, self.fail = mentions, threads, fail
        self.replied = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def handle(self):
        return "@" + ME

    def mentions(self):
        return self._mentions

    def thread_context(self, url):
        ancestors = self._threads[url]
        if ancestors is None:
            raise x_browser.XTargetMissing("gone")
        return {"ancestors": ancestors}

    def reply(self, url, text, *, dry_run=False, screenshot=None):
        if self.fail:
            raise self.fail
        self.replied.append((url, text, dry_run))
        return None if dry_run else f"https://x.com/{ME}/status/9"


def _state(mode, **kw):
    return {"run_key": "k", "local_time": "2026-11-23T14:30:00-07:00", "mode": mode, **kw}


def test_collect_keeps_only_direct_comments_and_records_skips(monkeypatch):
    monkeypatch.setenv("X_REPLY_MODE", "live")
    direct, nested, foreign, old = _mention("1"), _mention("2", handle="b"), _mention("3", handle="c"), _mention("4", age_h=50)
    session = _FakeSession([direct, nested, foreign, old], {
        direct["url"]: [_tweet(ME)],
        nested["url"]: [_tweet(ME), _tweet("fan"), _tweet(ME)],
        foreign["url"]: [_tweet("other")],
    })
    monkeypatch.setattr(x_reply.x_browser, "XSession", lambda: session)
    out = x_reply.collect(_state("live"))
    assert [c["id"] for c in out["candidates"]] == ["1"]
    assert out["candidates"][0]["chain"] == [_tweet(ME)]
    assert store.x_reply_handled_ids() == {"2", "3", "4"}  # the candidate is recorded only once answered


def test_collect_is_off_by_default_and_never_opens_a_browser(monkeypatch):
    monkeypatch.delenv("X_REPLY_MODE", raising=False)
    monkeypatch.setattr(x_reply.x_browser, "XSession", lambda: pytest.fail("browser opened"))
    assert x_reply.collect(_state("off"))["candidates"] == []


def test_collect_retries_unloadable_threads_later(monkeypatch):
    monkeypatch.setenv("X_REPLY_MODE", "live")
    m = _mention("1")
    monkeypatch.setattr(x_reply.x_browser, "XSession", lambda: _FakeSession([m], {m["url"]: None}))
    out = x_reply.collect(_state("live"))
    assert out["candidates"] == [] and store.x_reply_handled_ids() == set()


@pytest.mark.parametrize("hour,quiet", [(0, True), (3, True), (5, True), (6, False), (19, False), (23, False)])
def test_collect_skips_midnight_to_six(monkeypatch, hour, quiet):
    monkeypatch.setenv("X_REPLY_MODE", "live")
    opened = []
    monkeypatch.setattr(x_reply.x_browser, "XSession", lambda: opened.append(1) or _FakeSession([], {}))
    out = x_reply.collect({"run_key": "k", "local_time": f"2026-11-23T{hour:02d}:30:00-07:00"})
    assert bool(opened) is not quiet
    assert ("quiet hours" in "".join(out["results"])) is quiet


def test_draft_records_declines_and_bad_output_and_replies_once_per_author(monkeypatch):
    cands = [{**_mention(str(i), handle=h), "chain": [_tweet(ME)]} for i, h in enumerate(["a", "b", "c", "a"])]
    drafts = {"a": x_reply.ReplyDraft(should_reply=True, reply="谢谢你的补充"),
              "b": x_reply.ReplyDraft(should_reply=False, reason="spam"),
              "c": x_reply.ReplyDraft(should_reply=True, reply="下次再聊 https://x.com/x")}
    monkeypatch.setattr(x_reply, "write_reply", lambda me, c, feedback=None: drafts[c["handle"]])
    out = x_reply.draft({"mode": "live", "me": ME, "candidates": cands, "results": []})
    assert [(d["handle"], d["reply"], d["model"]) for d in out["drafts"]] == [("a", "谢谢你的补充", x_reply.local_model_name())]
    assert store.x_reply_handled_ids() == {"1", "2"}  # b declined, c invalid; a's two comments are not recorded


def test_draft_stops_at_the_per_run_limit(monkeypatch):
    cands = [{**_mention(str(i), handle=f"u{i}"), "chain": [_tweet(ME)]} for i in range(10)]
    monkeypatch.setattr(x_reply, "write_reply", lambda me, c, feedback=None: x_reply.ReplyDraft(should_reply=True, reply="谢谢"))
    out = x_reply.draft({"mode": "live", "me": ME, "candidates": cands, "results": []})
    assert len(out["drafts"]) == x_reply.X_REPLY_MAX_PER_RUN


def test_send_dry_run_never_counts_as_handled(monkeypatch, tmp_path):
    monkeypatch.setattr(x_reply, "X_SCREENSHOT_DIR", tmp_path)
    session = _FakeSession([], {})
    monkeypatch.setattr(x_reply.x_browser, "XSession", lambda: session)
    d = {**_mention("1"), "chain": [_tweet(ME)], "reply": "谢谢", "model": "Qwen"}
    x_reply.send(_state("dry_run", drafts=[d]))
    assert session.replied == [(d["url"], "谢谢", True)]
    assert store.x_reply_handled_ids() == set()


def test_send_live_records_reply_and_unconfirmed_is_never_retried(monkeypatch):
    session = _FakeSession([], {})
    monkeypatch.setattr(x_reply.x_browser, "XSession", lambda: session)
    d = {**_mention("1"), "chain": [_tweet(ME)], "reply": "谢谢", "model": "Qwen"}
    x_reply.send(_state("live", drafts=[d]))
    assert store.x_reply_handled_ids() == {"1"}
    with store.connect() as c:  # kept for review: what was said, to whom, by which model
        row = c.execute("SELECT status, reply_text, model, comment_text, parent_text, reply_url FROM x_replies").fetchone()
    assert tuple(row) == ("replied", "谢谢", "Qwen", "很有意思", "原帖", f"https://x.com/{ME}/status/9")

    d2 = {**_mention("2"), "chain": [_tweet(ME)], "reply": "谢谢", "model": "Qwen"}
    session.fail = x_browser.XPostUnconfirmed("no toast")
    out = x_reply.send(_state("live", drafts=[d2]))
    assert store.x_reply_handled_ids() == {"1", "2"} and "stopped" in out["results"][-1]


def test_second_run_only_sees_new_comments(monkeypatch):
    """A comment answered in one run is never looked at again; only comments that appeared since are handled."""
    monkeypatch.setenv("X_REPLY_MODE", "live")
    first, second = _mention("1"), _mention("2", handle="other")
    threads = {first["url"]: [_tweet(ME)], second["url"]: [_tweet(ME)]}
    sent = _FakeSession([], {})
    monkeypatch.setattr(x_reply.x_browser, "XSession", lambda: sent)
    monkeypatch.setattr(x_reply, "write_reply", lambda me, c, feedback=None: x_reply.ReplyDraft(should_reply=True, reply="谢谢"))

    for visible, expected in [([first], ["1"]), ([first, second], ["2"]), ([first, second], [])]:
        reader = _FakeSession(visible, threads)
        monkeypatch.setattr(x_reply.x_browser, "XSession", lambda reader=reader: reader)
        out = x_reply.collect(_state("live"))
        assert [c["id"] for c in out["candidates"]] == expected
        drafted = x_reply.draft({**_state("live"), "me": ME, "candidates": out["candidates"], "results": []})
        monkeypatch.setattr(x_reply.x_browser, "XSession", lambda: sent)
        x_reply.send({**_state("live"), "drafts": drafted["drafts"], "results": []})
    assert [url for url, _, _ in sent.replied] == [first["url"], second["url"]]


def test_backfill_ignores_age_but_not_other_rules():
    now = datetime.now(timezone.utc)
    assert "older than" in x_reply._skip_reason(_mention(id="9", age_h=500), ME, now)
    assert x_reply._skip_reason(_mention(id="9", age_h=500), ME, now, backfill=True) is None
    assert x_reply._skip_reason(_mention(handle="grok", age_h=500), ME, now, backfill=True) == "ignored account"
    assert x_reply._skip_reason(_mention(replying_to=("x",), age_h=500), ME, now, backfill=True) == "not a reply to us"


def test_comments_skipped_only_for_age_can_be_revisited():
    store.record_x_reply("skipped", "live", {**_mention("1"), "chain": []}, reason="older than 24h")
    store.record_x_reply("skipped", "live", {**_mention("2"), "chain": []}, reason="comment on a comment")
    assert store.x_reply_handled_ids() == {"1", "2"}
    assert store.x_reply_handled_ids(revisit_old=True) == {"2"}


def test_backfill_sends_in_batches_one_comment_per_author_per_batch(monkeypatch):
    monkeypatch.setenv("X_REPLY_MODE", "live")
    monkeypatch.setattr(x_reply, "X_REPLY_QUIET_HOURS", frozenset())
    comments = [{**_mention(str(i), handle=f"u{i % 4}", age_h=100 + i), "chain": [_tweet(ME)]} for i in range(8)]
    monkeypatch.setattr(x_reply, "find_backlog", lambda limit, mode: (ME, comments[:limit], ["found"]))
    monkeypatch.setattr(x_reply, "write_reply", lambda me, c, feedback=None: x_reply.ReplyDraft(should_reply=True, reply="谢谢"))
    sessions = []
    monkeypatch.setattr(x_reply.x_browser, "XSession", lambda: sessions.append(_FakeSession([], {})) or sessions[-1])
    rests = []
    monkeypatch.setattr(x_reply.time, "sleep", rests.append)
    results = x_reply.backfill(8)
    sent = [url for s in sessions for url, _, _ in s.replied]
    assert sorted(sent) == sorted(c["url"] for c in comments) and len(set(sent)) == 8
    assert len(sessions) == 2 and all(len({u.split("/")[3] for u, _, _ in s.replied}) == len(s.replied) for s in sessions)
    assert any(r >= x_reply.BACKFILL_REST_SECONDS[0] for r in rests)  # it rested between the batches
    assert store.x_reply_handled_ids() == {str(i) for i in range(8)}


def test_backfill_stops_after_a_send_problem_and_leaves_the_rest(monkeypatch):
    monkeypatch.setenv("X_REPLY_MODE", "live")
    monkeypatch.setattr(x_reply, "X_REPLY_QUIET_HOURS", frozenset())
    comments = [{**_mention(str(i), handle=f"u{i}", age_h=100), "chain": [_tweet(ME)]} for i in range(8)]
    monkeypatch.setattr(x_reply, "find_backlog", lambda limit, mode: (ME, comments, []))
    monkeypatch.setattr(x_reply, "write_reply", lambda me, c, feedback=None: x_reply.ReplyDraft(should_reply=True, reply="谢谢"))
    session = _FakeSession([], {}, fail=x_browser.XPostError("X asked for verification"))
    monkeypatch.setattr(x_reply.x_browser, "XSession", lambda: session)
    monkeypatch.setattr(x_reply.time, "sleep", lambda s: None)
    results = x_reply.backfill(8)
    assert "stopping the backfill" in results[-1]
    assert store.x_reply_handled_ids() == set()  # nothing was sent, so everything is still unanswered


def test_one_person_can_get_replies_under_different_posts_in_the_same_run(monkeypatch):
    monkeypatch.setattr(x_reply, "write_reply", lambda me, c, feedback=None: x_reply.ReplyDraft(should_reply=True, reply="谢谢"))
    cands = [{**_mention("1", handle="a"), "chain": [_tweet(ME, "新闻A")]},
             {**_mention("2", handle="a"), "chain": [_tweet(ME, "新闻B")]},   # same person, other news: both answered
             {**_mention("3", handle="a"), "chain": [_tweet(ME, "新闻A")]}]   # same person, same news: next run
    out = x_reply.draft({"mode": "live", "me": ME, "candidates": cands, "results": []})
    assert [d["id"] for d in out["drafts"]] == ["1", "2"]
