from __future__ import annotations

import json
from pathlib import Path


XRAY_NOTEBOOK = Path(__file__).parents[1] / "participant" / "workspace" / "xray_red/xray_red_blue_starter.ipynb"


def test_start_notebook_directs_xray_players_to_arena_challenge_pages():
    path = Path(__file__).parents[1] / "participant/workspace/start_here.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )
    assert "Open X-Ray Red and X-Ray Blue from their CTFd challenge pages" in markdown
    assert "xray_red_blue_starter.ipynb" not in markdown


def test_xray_starter_explains_roles_handoff_and_blue_scoring():
    notebook = json.loads(XRAY_NOTEBOOK.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "CTFd does not enforce team membership" in markdown
    assert "Each account must submit its own Red flag" in markdown
    assert "detect at least 5%" in markdown
    assert "flags it for review" in markdown
    assert "confidence=0.6" in code


def test_xray_redblue_notebook_and_both_launchers_allow_the_notebook():
    notebook = json.loads(XRAY_NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]
    all_cells = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    source = "\n".join(code_cells)
    assert "Red challenge" in all_cells
    assert "Blue challenge" in all_cells
    assert 'detector="disagreement"' in source
    assert "confidence=0.6" in source
    assert 'os.getenv("ARENA_TOKEN") or os.getenv("JUPYTER_TOKEN", "")' in source
    assert '"Authorization": f"Bearer {ARENA_TOKEN}"' in source
    assert "headers=xray_headers()" in source
    for cell_source in code_cells:
        compile(cell_source, "xray_red/xray_red_blue_starter.ipynb", "exec")
    repo = Path(__file__).parents[1]
    assert '"xray_red/xray_red_blue_starter.ipynb"' in (repo / "launcher/app/main.py").read_text(encoding="utf-8")
    assert '"xray_red/xray_red_blue_starter.ipynb"' in (repo / "ctfd/plugins/workspace_launcher/__init__.py").read_text(encoding="utf-8")


def test_interactive_arena_notebooks_are_static_role_specific_and_compilable():
    repo = Path(__file__).parents[1]
    red = json.loads((repo / "participant/workspace/xray_red/red_team_fgsm.ipynb").read_text(encoding="utf-8"))
    blue = json.loads((repo / "participant/workspace/xray_blue/blue_team.ipynb").read_text(encoding="utf-8"))
    helper = (repo / "participant/workspace/red_blue_arena.py").read_text(encoding="utf-8")
    for name, notebook, role in (("xray_red/red_team_fgsm.ipynb", red, "red"), ("xray_blue/blue_team.ipynb", blue, "blue")):
        code_cells = ["".join(cell.get("source", [])) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert any("import red_blue_arena as arena" in source for source in code_cells)
        assert any("arena.identity()" in source for source in code_cells)
        assert any(f"['role'] == '{role}'" in source for source in code_cells)
        for source in code_cells:
            compile(source, name, "exec")
        assert name in (repo / "launcher/app/main.py").read_text(encoding="utf-8")
        assert name in (repo / "ctfd/plugins/workspace_launcher/__init__.py").read_text(encoding="utf-8")
    compile(helper, "red_blue_arena.py", "exec")
    assert 'os.getenv("ARENA_TOKEN") or os.getenv("JUPYTER_TOKEN", "")' in helper
    assert "submit_red_attack" in helper and "submit_blue_response" in helper


def test_blue_notebook_has_visible_launch_preview_placeholder():
    path = Path(__file__).parents[1] / "participant/workspace/xray_blue/blue_team.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    preview_cells = [
        (index, cell)
        for index, cell in enumerate(cells)
        if "<!-- ARENA_BLUE_PREVIEW -->" in "".join(cell.get("source", []))
    ]
    assert len(preview_cells) == 1
    index, preview = preview_cells[0]
    assert index == 1
    assert preview["id"] == "blue-attack-preview"
    assert preview["cell_type"] == "markdown"
    assert "No validated Red image has been staged yet" in "".join(preview["source"])
    assert cells[index + 1]["id"] == "blue-smoothing"


def test_red_fgsm_notebook_requires_learner_work_and_shows_blue_only_after_success(capsys):
    from types import SimpleNamespace

    notebook_path = Path(__file__).parents[1] / "participant/workspace/xray_red/red_team_fgsm.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_by_id = {
        cell["id"]: "".join(cell["source"])
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    }
    todo = code_by_id["red-fgsm-todo"]
    assert "loss = None" in todo
    assert "gradient = None" in todo
    assert "candidate_tensor = None" in todo
    assert "raise NotImplementedError" in todo
    assert "arena.make_adversarial_candidate(" not in todo
    assert "turn_indicator_html" not in "\n".join(code_by_id.values())

    submission_code = code_by_id["red-submit-code"]

    def render(result):
        displayed = []
        arena = SimpleNamespace(
            submit_red_attack=lambda *args, **kwargs: result,
            attack_result_card_html=lambda response: "accepted result card",
        )
        namespace = {
            "arena": arena, "candidate_path": "candidate.png", "source_id": "0",
            "epsilon": 5 / 255, "HTML": lambda value: value, "display": displayed.append,
        }
        exec(compile(submission_code, notebook_path.name, "exec"), namespace)
        return displayed

    assert render({"submitted": False}) == []
    assert "Next: Blue Team" not in capsys.readouterr().out

    displays = render({"submitted": True, "flag": "flag{example}"})
    assert any("Next: Blue Team" in value for value in displays)
    assert "Red succeeded" in capsys.readouterr().out

    displayed = []
    failing_arena = SimpleNamespace(submit_red_attack=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rejected")))
    exec(compile(submission_code, notebook_path.name, "exec"), {
        "arena": failing_arena, "candidate_path": "candidate.png", "source_id": "0",
        "epsilon": 5 / 255, "HTML": lambda value: value, "display": displayed.append,
    })
    assert displayed == []
    assert "did not accept" in capsys.readouterr().out
