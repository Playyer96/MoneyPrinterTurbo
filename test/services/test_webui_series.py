"""Run the real Streamlit page: series toggle, part count, chapter planning."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from app.config import config
from app.services import llm


@pytest.fixture
def ui(monkeypatch):
    monkeypatch.setattr(config, "save_config", Mock())
    monkeypatch.setattr(config, "try_save_config", Mock(return_value=True))
    page = AppTest.from_file(
        str(Path(__file__).parents[2] / "webui/Main.py"), default_timeout=60
    )
    page.session_state["ui_language"] = "en"
    return page


def widget(page, group, key):
    return next(item for item in getattr(page, group) if item.key == key)


def test_series_options_appear_only_when_series_mode_is_on(ui):
    ui.run()
    assert not [item for item in ui.number_input if item.key == "series_parts_input"]

    widget(ui, "toggle", "series_enabled").set_value(True).run()

    assert widget(ui, "number_input", "series_parts_input").value == 0
    assert widget(ui, "toggle", "series_continuity").value is True
    assert widget(ui, "text_area", "series_outline").value == ""
    assert not ui.exception


def test_planning_chapters_fills_the_editable_outline(monkeypatch, ui):
    planner = Mock(return_value=["hive setup", "harvesting honey"])
    monkeypatch.setattr(llm, "generate_series_outline", planner)

    ui.session_state["video_subject"] = "beekeeping"
    ui.run()
    widget(ui, "toggle", "series_enabled").set_value(True).run()
    widget(ui, "button", "plan_series_chapters").click().run()

    assert planner.call_args.kwargs["parts"] == 0
    assert (
        widget(ui, "text_area", "series_outline").value
        == "hive setup\nharvesting honey"
    )
    assert not ui.exception


def test_planning_without_a_subject_warns_instead_of_calling_the_model(monkeypatch, ui):
    planner = Mock()
    monkeypatch.setattr(llm, "generate_series_outline", planner)

    ui.run()
    widget(ui, "toggle", "series_enabled").set_value(True).run()
    widget(ui, "button", "plan_series_chapters").click().run()

    planner.assert_not_called()
    assert any("video subject" in str(item.value).lower() for item in ui.warning)
    assert not ui.exception
