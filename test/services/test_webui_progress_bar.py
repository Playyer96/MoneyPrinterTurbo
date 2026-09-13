import ast
from pathlib import Path


WEBUI_MAIN = Path(__file__).parents[2] / "webui" / "Main.py"


def test_generation_snapshot_does_not_render_progress_bar():
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_render_generation_task_snapshot"
    )

    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "st"
        and node.func.attr == "progress"
        for node in ast.walk(function)
    )
