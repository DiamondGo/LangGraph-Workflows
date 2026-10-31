from datetime import datetime, timedelta, timezone

from lg_workflows import notify


def _setup(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(notify, "STATE_PATH", tmp_path / "alerts.json")
    monkeypatch.setattr(notify, "send", lambda title, body: sent.append(title) or [])
    return sent


def test_alert_once_then_daily_and_recovery(monkeypatch, tmp_path):
    sent = _setup(monkeypatch, tmp_path)
    assert notify.alert("k", "down", "e1")
    assert not notify.alert("k", "down", "e2")
    state = notify._load()
    state["k"]["notified_at"] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    notify._save(state)
    assert notify.alert("k", "down", "e3")
    assert notify.resolve("k", "up", "ok")
    assert not notify.resolve("k", "up", "ok")
    assert sent == ["down", "down", "up"]


def test_alert_waits_for_repeated_failures(monkeypatch, tmp_path):
    sent = _setup(monkeypatch, tmp_path)
    assert not notify.alert("k", "flaky", "e", after_failures=2)
    assert not notify.resolve("k", "up", "ok")
    assert not notify.alert("k", "flaky", "e", after_failures=2)
    assert notify.alert("k", "flaky", "e", after_failures=2)
    assert sent == ["flaky"]
