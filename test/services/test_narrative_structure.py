from app.models.schema import VideoParams
from app.services import llm
from app.services.pipeline.series import build_series_part_prompt


def test_script_prompt_always_requires_a_complete_narrative_arc():
    prompt = llm.build_script_prompt(
        video_subject="A lost hiker",
        custom_system_prompt="Use restrained cinematic narration.",
    )

    assert "one short, catchy sentence" in prompt
    assert "concrete problems, conflict, or consequences" in prompt
    assert "satisfying emotional landing" in prompt
    assert "final sentence decisive and memorable" in prompt


def test_non_final_series_part_ends_with_a_specific_continuation():
    prompt = build_series_part_prompt(
        VideoParams(video_subject="Saving a family bakery"),
        ["The eviction notice", "A risky new recipe", "The bakery survives"],
        2,
    )

    assert "The next part will cover: The bakery survives" in prompt
    assert "specific bridge or cliffhanger" in prompt
    assert "story continues in the next part" in prompt


def test_series_outline_plans_one_complete_story_arc():
    prompt = llm.build_series_outline_prompt("Saving a family bakery", parts=3)

    assert "full series as a narrative arc" in prompt
    assert "middle chapters that develop problems and consequences" in prompt
    assert "final chapter that resolves the central conflict" in prompt


def test_final_series_part_resolves_and_closes_decisively():
    prompt = build_series_part_prompt(
        VideoParams(video_subject="Saving a family bakery"),
        ["The eviction notice", "The bakery survives"],
        2,
    )

    assert "Resolve the series' central problems" in prompt
    assert "definitive final sentence" in prompt
    assert "Do not promise another part" in prompt
