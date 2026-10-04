import json
from pathlib import Path

import pytest

from scripts.retire_workspace_exercises import checked_path, retire


TEMPLATE = Path(__file__).resolve().parents[1] / "participant/workspace"


def test_retirement_removes_only_retired_exercises_and_preserves_student_work(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("granny", "granny2", "granny_infinity"):
        (workspace / name).mkdir()
        (workspace / name / "student-candidate.png").write_bytes(b"old image")
    for name in ("granny_starter.ipynb", "score_display.py", "t_wolf.jpg"):
        (workspace / name).write_bytes(b"old exercise")
    for name in ("patchguard", "llm_safety", "xray_red", "xray_blue"):
        (workspace / name).mkdir()
        (workspace / name / "student.ipynb").write_bytes(b"student work")
    (workspace / "previous_notebooks").mkdir()
    (workspace / "previous_notebooks/granny2_starter.1.ipynb").write_bytes(b"old")
    (workspace / "previous_notebooks/blue_team.ipynb").write_bytes(b"blue notes")
    (workspace / ".ipynb_checkpoints").mkdir()
    (workspace / ".ipynb_checkpoints/granny_starter-checkpoint.ipynb").write_bytes(b"old")
    (workspace / ".ipynb_checkpoints/start_here-checkpoint.ipynb").write_text("granny")
    (workspace / "README.md").write_text("# Granny challenge workspace\nold")
    welcome = json.loads((TEMPLATE / "start_here.ipynb").read_text(encoding="utf-8"))
    for cell in welcome["cells"]:
        if cell.get("id") == "exercise-guide":
            cell["source"] = ["Old Granny links"]
    student_cell = {"cell_type": "code", "id": "my-work", "source": ["saved = 42"],
                    "outputs": [{"output_type": "stream", "name": "stdout", "text": ["42"]}],
                    "metadata": {}, "execution_count": 1}
    welcome["cells"].append(student_cell)
    (workspace / "start_here.ipynb").write_text(json.dumps(welcome), encoding="utf-8")

    assert retire(workspace, TEMPLATE)["removed"] == 8
    for name in ("patchguard", "llm_safety", "xray_red", "xray_blue"):
        assert (workspace / name / "student.ipynb").read_bytes() == b"student work"
    assert (workspace / "previous_notebooks/blue_team.ipynb").read_bytes() == b"blue notes"
    updated = (workspace / "start_here.ipynb").read_text(encoding="utf-8")
    assert "granny" not in updated.lower()
    assert json.loads(updated)["cells"][-1] == student_cell
    assert "granny" not in (workspace / "README.md").read_text().lower()
    assert "granny" not in (workspace / ".ipynb_checkpoints/start_here-checkpoint.ipynb").read_text().lower()
    assert retire(workspace, TEMPLATE) == {"removed": 0, "updated": 0}


def test_retirement_rejects_paths_outside_the_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "other-student.ipynb"
    outside.write_bytes(b"keep")
    with pytest.raises(ValueError, match="outside the workspace"):
        checked_path(workspace.resolve(), workspace / "../other-student.ipynb")
    assert outside.read_bytes() == b"keep"


def test_retirement_cleans_verified_live_artifacts_and_preserves_other_audit_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fixtures = workspace / "audit-fixtures"
    fixtures.mkdir()
    retired = (
        "granny_infinity_stress_success.png",
        ".granny_infinity_starter.before-strict-20261001.ipynb",
        ".granny_infinity_starter.before-ssim-20261001.ipynb",
        "audit-results.json",
        "audit-fixtures/infinity.png", "audit-fixtures/apple.jpg",
        "audit-fixtures/granny2.png", "audit-fixtures/granny.png",
        "audit-fixtures/infinity-before-ssim-20261001.png",
        "audit-fixtures/infinity-0.08-before-strict.png",
    )
    retained = (
        "audit-preservation.txt", "my_candidate.png", "my_adversarial.png",
        "audit-fixtures/random_model.pt", "audit-fixtures/weak_model.pt",
        "audit-fixtures/reference_model.pt",
    )
    for name in (*retired, *retained):
        (workspace / name).write_bytes(b"existing student or audit content")
    assert retire(workspace, TEMPLATE)["removed"] == len(retired)
    assert all(not (workspace / name).exists() for name in retired)
    assert all((workspace / name).read_bytes() == b"existing student or audit content" for name in retained)
    assert retire(workspace, TEMPLATE) == {"removed": 0, "updated": 0}


@pytest.mark.parametrize("source,removed", [
    ("result = query_granny_infinity('my_granny_infinity.png')", True),
    ("query_target('t_wolf.jpg')  # http://target:8000", True),
    ("# Notes about Granny", False),
    ("result = query_granny_infinity('image.png')\n# PatchGuard training work", False),
    ("import red_blue_arena as arena", False),
])
def test_generic_checkpoint_is_removed_only_when_identified_as_retired(source, removed, tmp_path):
    workspace = tmp_path / "workspace"
    checkpoint = workspace / ".ipynb_checkpoints/Untitled-checkpoint.ipynb"
    checkpoint.parent.mkdir(parents=True)
    original = json.dumps({"cells": [{"source": [source], "cell_type": "code"}]})
    checkpoint.write_text(original, encoding="utf-8")
    retire(workspace, TEMPLATE)
    assert checkpoint.exists() is not removed
    if not removed:
        assert checkpoint.read_text(encoding="utf-8") == original


def test_retirement_does_not_follow_a_retired_directory_symlink(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "outside"
    external.mkdir()
    (external / "keep.txt").write_bytes(b"keep")
    try:
        (workspace / "granny").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is not available on this host")
    retire(workspace, TEMPLATE)
    assert (external / "keep.txt").read_bytes() == b"keep"
    assert not (workspace / "granny").is_symlink()


def test_new_workspace_contains_only_retained_exercises_and_no_removed_helpers():
    expected = {"patchguard", "llm_safety", "xray_red", "xray_blue"}
    assert {path.parent.name for path in TEMPLATE.glob("*/*.ipynb")
            if path.parent.name != ".ipynb_checkpoints"} == expected
    assert not (TEMPLATE / "score_display.py").exists()
    assert not (TEMPLATE / "t_wolf.jpg").exists()
    for path in TEMPLATE.rglob("*.ipynb"):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert "granny" not in json.dumps(notebook).lower()
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell.get("source", [])), str(path), "exec")
