from pathlib import Path

from scripts.organize_participant_workspace import NOTEBOOKS, SUPPORT_FILES, organize


def test_exercise_support_files_match_workspace_sources():
    base = Path(__file__).resolve().parents[1] / "participant" / "workspace"
    for target_name, source_name in SUPPORT_FILES.items():
        assert (base / target_name).read_bytes() == (base / source_name).read_bytes()


def test_organize_existing_workspace_preserves_edits_and_is_repeatable(tmp_path):
    template = tmp_path / "template"
    workspace = tmp_path / "workspace"
    template.mkdir()
    workspace.mkdir()
    repo = Path(__file__).resolve().parents[1] / "participant" / "workspace"
    for path in repo.rglob("*.ipynb"):
        destination = template / path.relative_to(repo)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    for helper in ("red_blue_arena.py", "llm_safety_optimizer.py"):
        (template / helper).write_bytes((repo / helper).read_bytes())

    (workspace / "llm_safety_starter.ipynb").write_text("student edits", encoding="utf-8")
    (workspace / "blue_team.ipynb").write_text("older edit", encoding="utf-8")
    (workspace / "xray_blue").mkdir()
    (workspace / "xray_blue" / "blue_team.ipynb").write_text("newer edit", encoding="utf-8")
    (workspace / "start_here.ipynb").write_text(
        "[LLM](/lab/tree/llm_safety_starter.ipynb)", encoding="utf-8"
    )

    result = organize(workspace, template)
    assert result["moved"] == 1
    assert result["backed_up"] == 1
    assert (workspace / NOTEBOOKS["llm_safety_starter.ipynb"]).read_text() == "student edits"
    assert (workspace / "xray_blue/blue_team.ipynb").read_text() == "newer edit"
    assert (workspace / "previous_notebooks/blue_team.ipynb").read_text() == "older edit"
    assert (workspace / "xray_red/red_team_fgsm.ipynb").exists()
    assert 'WORKSPACE = Path(__file__).resolve().parent' in (
        workspace / "xray_red/red_blue_arena.py"
    ).read_text()
    assert "/lab/tree/llm_safety/llm_safety_starter.ipynb" in (workspace / "start_here.ipynb").read_text()
    assert organize(workspace, template) == {"moved": 0, "restored": 0, "backed_up": 0, "support": 0}
