import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


_SERVER = Path(__file__).parents[2] / "vendor" / "omnivoice" / "server.py"
_SPEC = importlib.util.spec_from_file_location("omnivoice_server", _SERVER)
assert _SPEC and _SPEC.loader
omnivoice_server = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = omnivoice_server
_SPEC.loader.exec_module(omnivoice_server)


def test_omnivoice_combines_preset_and_style_instruction(monkeypatch):
    captured = {}

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return [np.zeros(240, dtype=np.float32)]

    monkeypatch.setattr(omnivoice_server, "_load_model", lambda: FakeModel())

    result = omnivoice_server.generate_audio(
        omnivoice_server.GenerateRequest(
            text="Hello there",
            voice="narrator",
            style="energetic and playful",
        )
    )

    assert result.status_code == 200
    assert captured["instruct"] == "male, middle-aged, low pitch, energetic and playful"


def test_omnivoice_style_layers_emotion_on_top_of_a_cloned_profile(
    monkeypatch, tmp_path
):
    """A per-call style string must add emotion WITHOUT replacing the prompt.

    The previous behaviour ignored style for clones, which made cloned voices
    monotone. The clone prompt still carries the speaker's timbre and the
    new instruct string layers the delivery (angry, excited, ...) on top.
    """
    captured = {}

    class FakePrompt:
        pass

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return [[0.0]]

    fake_omnivoice = types.ModuleType("omnivoice")
    fake_models = types.ModuleType("omnivoice.models")
    fake_model_module = types.ModuleType("omnivoice.models.omnivoice")
    fake_model_module.VoiceClonePrompt = types.SimpleNamespace(
        load=lambda _path: FakePrompt()
    )
    monkeypatch.setitem(sys.modules, "omnivoice", fake_omnivoice)
    monkeypatch.setitem(sys.modules, "omnivoice.models", fake_models)
    monkeypatch.setitem(sys.modules, "omnivoice.models.omnivoice", fake_model_module)
    monkeypatch.setattr(omnivoice_server, "_load_model", lambda: FakeModel())
    monkeypatch.setattr(
        omnivoice_server,
        "_profile_prompt_path",
        lambda _voice: str(tmp_path / "clone.pt"),
    )
    monkeypatch.setattr(omnivoice_server, "_load_profile_metadata", lambda _name: {})

    omnivoice_server.generate_audio(
        omnivoice_server.GenerateRequest(
            text="Hello", voice="my_clone", style="angry and forceful"
        )
    )

    assert isinstance(captured["voice_clone_prompt"], FakePrompt)
    assert captured["instruct"] == "angry and forceful"


def test_omnivoice_profile_instruct_override_is_merged_before_call_style(
    monkeypatch, tmp_path
):
    """Profile-level instruct_override sets the register; per-call style layers on top."""
    captured = {}

    class FakePrompt:
        pass

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return [[0.0]]

    fake_omnivoice = types.ModuleType("omnivoice")
    fake_models = types.ModuleType("omnivoice.models")
    fake_model_module = types.ModuleType("omnivoice.models.omnivoice")
    fake_model_module.VoiceClonePrompt = types.SimpleNamespace(
        load=lambda _path: FakePrompt()
    )
    monkeypatch.setitem(sys.modules, "omnivoice", fake_omnivoice)
    monkeypatch.setitem(sys.modules, "omnivoice.models", fake_models)
    monkeypatch.setitem(sys.modules, "omnivoice.models.omnivoice", fake_model_module)
    monkeypatch.setattr(omnivoice_server, "_load_model", lambda: FakeModel())
    monkeypatch.setattr(
        omnivoice_server,
        "_profile_prompt_path",
        lambda _voice: str(tmp_path / "clone.pt"),
    )
    monkeypatch.setattr(
        omnivoice_server,
        "_load_profile_metadata",
        lambda _name: {"instruct_override": "enthusiastic, energetic"},
    )

    omnivoice_server.generate_audio(
        omnivoice_server.GenerateRequest(
            text="Hello", voice="my_clone", style="softer"
        )
    )

    assert isinstance(captured["voice_clone_prompt"], FakePrompt)
    # Per-call style comes AFTER the profile's register so it can shadow or
    # append to it.
    assert captured["instruct"] == "enthusiastic, energetic, softer"


def test_omnivoice_clone_without_metadata_does_not_inject_instruct(
    monkeypatch, tmp_path
):
    """Cloned profiles without instruct_override still get only the prompt."""
    captured = {}

    class FakePrompt:
        pass

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            return [[0.0]]

    fake_omnivoice = types.ModuleType("omnivoice")
    fake_models = types.ModuleType("omnivoice.models")
    fake_model_module = types.ModuleType("omnivoice.models.omnivoice")
    fake_model_module.VoiceClonePrompt = types.SimpleNamespace(
        load=lambda _path: FakePrompt()
    )
    monkeypatch.setitem(sys.modules, "omnivoice", fake_omnivoice)
    monkeypatch.setitem(sys.modules, "omnivoice.models", fake_models)
    monkeypatch.setitem(sys.modules, "omnivoice.models.omnivoice", fake_model_module)
    monkeypatch.setattr(omnivoice_server, "_load_model", lambda: FakeModel())
    monkeypatch.setattr(
        omnivoice_server,
        "_profile_prompt_path",
        lambda _voice: str(tmp_path / "clone.pt"),
    )
    monkeypatch.setattr(omnivoice_server, "_load_profile_metadata", lambda _name: {})

    omnivoice_server.generate_audio(
        omnivoice_server.GenerateRequest(text="Hello", voice="my_clone")
    )

    assert isinstance(captured["voice_clone_prompt"], FakePrompt)
    assert "instruct" not in captured
