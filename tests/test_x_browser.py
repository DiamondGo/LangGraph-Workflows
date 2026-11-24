import random
from unittest.mock import MagicMock

import pytest

from lg_workflows import x_browser


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(x_browser, "random", random.Random(7))
    session = x_browser.XSession()
    session.page = MagicMock()
    session.page.viewport_size = {"width": 1280, "height": 900}
    session.page.url = "https://x.com/compose/post"
    return session


def test_browser_launch_uses_normal_chrome_identity(monkeypatch, tmp_path):
    pw = MagicMock()
    monkeypatch.setattr(x_browser, "sync_playwright", lambda: MagicMock(start=lambda: pw))
    monkeypatch.setattr(x_browser, "PROFILE_DIR", tmp_path)
    agent = "Mozilla/5.0 Chrome/156.0.8078.79 Safari/537.36"
    monkeypatch.setattr(x_browser, "user_agent", lambda: agent)

    with x_browser.XSession():
        options = pw.chromium.launch_persistent_context.call_args.kwargs
        assert options["user_agent"] == agent
        assert "--disable-blink-features=AutomationControlled" in options["args"]
        assert options["ignore_default_args"] == ["--enable-automation"]
        assert options["headless"] is True
    pw.chromium.launch_persistent_context.return_value.close.assert_called_once()
    pw.stop.assert_called_once()


def test_typing_samples_a_new_delay_for_each_character(session):
    text = "abc\u4e2d\u6587\nhttps://example.com"
    session._type_text(text)

    assert [c.args[0] for c in session.page.keyboard.type.call_args_list] == list(text)
    delays = [c.args[0] for c in session.page.wait_for_timeout.call_args_list]
    assert len(delays) == len(text)
    assert len(set(delays)) == len(text)
    assert all(90 <= delay <= 1660 for delay in delays)
    assert all(delays[i] >= 540 for i, char in enumerate(text) if char in "\n.")


def test_typing_adds_occasional_pauses_without_punctuation(session):
    session._type_text("a" * 100)
    delays = [c.args[0] for c in session.page.wait_for_timeout.call_args_list]
    assert any(delay > 260 for delay in delays)
    assert any(delay <= 260 for delay in delays)


def test_mouse_follows_a_bounded_curve_and_remembers_its_position(session):
    session._move_mouse(800, 500)
    moves = [c.args for c in session.page.mouse.move.call_args_list]
    assert 18 <= len(moves) <= 35
    assert moves[-1] == (800, 500)
    assert session._mouse_position == (800, 500)
    assert all(0 <= x < 1280 and 0 <= y < 900 for x, y in moves)
    assert any(abs(y - x * 500 / 800) > 1 for x, y in moves[:-1])
    delays = [c.args[0] for c in session.page.wait_for_timeout.call_args_list]
    assert len(delays) == len(moves)
    assert all(8 <= delay <= 22 for delay in delays)
    assert len(set(delays)) > 1


@pytest.mark.parametrize("bounds", [
    {"x": 400, "y": 300, "width": 200, "height": 60},
    {"x": -50, "y": 800, "width": 200, "height": 200},
])
def test_click_moves_to_a_random_visible_point_in_target(session, bounds):
    target = MagicMock()
    target.bounding_box.return_value = bounds
    session._click(target)

    target.scroll_into_view_if_needed.assert_called_once()
    options = target.click.call_args.kwargs
    position = options["position"]
    x, y = session.page.mouse.move.call_args.args
    assert x == position["x"] + bounds["x"]
    assert y == position["y"] + bounds["y"]
    assert 0 < position["x"] < bounds["width"]
    assert 0 < position["y"] < bounds["height"]
    assert 0 <= x < 1280 and 0 <= y < 900
    assert 70 <= options["delay"] <= 170


@pytest.mark.parametrize("bounds", [None, {"x": 1300, "y": 0, "width": 100, "height": 50}])
def test_click_never_clicks_an_invisible_or_offscreen_target(session, bounds):
    target = MagicMock()
    target.bounding_box.return_value = bounds
    with pytest.raises(x_browser.XPostError):
        session._click(target)
    target.click.assert_not_called()


def _compose_page(session, monkeypatch):
    box, button = MagicMock(), MagicMock()
    session.page.locator.side_effect = lambda selector: {
        f'[role="dialog"] {x_browser._TEXTAREA}': box,
        f'[role="dialog"] {x_browser._POST_BUTTON}': button,
    }[selector]
    button.is_enabled.return_value = True
    clicks = []
    monkeypatch.setattr(session, "_click", clicks.append)
    return box, button, clicks


def test_compose_controls_are_scoped_to_dialog(session, monkeypatch):
    _compose_page(session, monkeypatch)
    session.post("test", dry_run=True)
    assert [c.args[0] for c in session.page.locator.call_args_list] == [
        f'[role="dialog"] {x_browser._TEXTAREA}',
        f'[role="dialog"] {x_browser._POST_BUTTON}',
    ]


def test_dry_run_types_exact_text_but_never_clicks_post(session, monkeypatch, tmp_path):
    box, button, clicks = _compose_page(session, monkeypatch)
    shot = tmp_path / "screenshots" / "compose.png"
    assert session.post("abc\u4e2d\u6587\n", dry_run=True, screenshot=shot) is None
    assert clicks == [box]
    assert "".join(c.args[0] for c in session.page.keyboard.type.call_args_list) == "abc\u4e2d\u6587\n"
    session.page.screenshot.assert_called_once_with(path=str(shot))
    session.page.wait_for_selector.assert_not_called()


def test_live_post_clicks_once_and_requires_confirmation(session, monkeypatch):
    box, button, clicks = _compose_page(session, monkeypatch)
    toast = session.page.wait_for_selector.return_value
    toast.query_selector.return_value.get_attribute.return_value = "/example/status/123"
    assert session.post("test") == "https://x.com/example/status/123"
    assert clicks == [box, button]


def test_x_rejection_is_not_retried(session, monkeypatch):
    box, button, clicks = _compose_page(session, monkeypatch)
    toast = session.page.wait_for_selector.return_value
    toast.query_selector.return_value = None
    toast.inner_text.return_value = "This request looks like it might be automated."
    with pytest.raises(x_browser.XPostError, match="X rejected the post"):
        session.post("test")
    assert clicks == [box, button]


def test_missing_confirmation_is_not_retried(session, monkeypatch):
    box, button, clicks = _compose_page(session, monkeypatch)
    session.page.wait_for_selector.side_effect = x_browser.PlaywrightTimeout("no toast")
    with pytest.raises(x_browser.XPostUnconfirmed):
        session.post("test")
    assert clicks == [box, button]


def test_disabled_post_button_is_not_clicked(session, monkeypatch):
    box, button, clicks = _compose_page(session, monkeypatch)
    button.is_enabled.return_value = False
    with pytest.raises(x_browser.XPostError, match="post button is disabled"):
        session.post("test")
    assert clicks == [box]


@pytest.mark.parametrize("url, error", [
    ("https://x.com/i/flow/login", x_browser.XLoginRequired),
    ("https://x.com/account/access", x_browser.XPostError),
])
def test_login_and_verification_stop_before_typing(session, monkeypatch, url, error):
    box, button, clicks = _compose_page(session, monkeypatch)
    session.page.url = url
    with pytest.raises(error):
        session.post("test")
    assert clicks == []
    session.page.keyboard.type.assert_not_called()


def test_oversize_post_is_rejected_before_opening_the_page(session):
    with pytest.raises(ValueError, match="weighted chars"):
        session.post("a" * 281)
    session.page.goto.assert_not_called()


def test_compose_box_timeout_reloads_once_then_succeeds(session, monkeypatch):
    box, button, clicks = _compose_page(session, monkeypatch)
    box.wait_for.side_effect = [x_browser.PlaywrightTimeout("slow"), None]
    session.post("test", dry_run=True)
    assert session.page.goto.call_count == 2


def test_compose_box_failing_twice_saves_evidence_and_stops(session, monkeypatch, tmp_path):
    box, button, clicks = _compose_page(session, monkeypatch)
    box.wait_for.side_effect = x_browser.PlaywrightTimeout("never")
    monkeypatch.setattr(x_browser, "FAILURE_DIR", tmp_path)
    session.page.inner_text.return_value = "Something went wrong"
    session.page.title.return_value = "X"
    with pytest.raises(x_browser.XPostError, match="did not appear after 2 loads"):
        session.post("test")
    assert session.page.goto.call_count == 2
    assert list(tmp_path.glob("fail-compose-*.txt"))
    assert not clicks
