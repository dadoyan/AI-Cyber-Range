import json
from pathlib import Path


def test_patchguard_starter_has_required_todos_and_local_submission_flow():
    notebook_path = Path(__file__).parents[1] / "participant/workspace/patchguard/patchguard_starter.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = ["".join(cell["source"]) for cell in notebook["cells"]]
    text = "\n".join(sources)
    assert "from range_device import select_device" in text
    assert "device=select_device()" in text
    assert "download=False" in text
    assert "def extract_windows" in text and "TODO" in text
    assert "def patchguard_predict" in text and "def patchguard_accuracy" in text
    assert "submit_patchguard_model" in text
    assert "patchguard-target:8000" in text and "/score" in text
    for stale in ("Google Drive", "Colab", "Gradescope", "Group members"):
        assert stale not in text


def test_reference_notebook_with_cleared_outputs_matches_canonical_hash():
    import hashlib

    path = Path(__file__).parents[1] / "reference/A6_PatchGuard_clean.ipynb"
    # Saved execution history is cleared; code, prose, and kernel metadata are preserved.
    # Git may check out CRLF on Windows; hash the canonical LF source bytes.
    canonical_source = path.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(canonical_source).hexdigest() == (
        "ed2db0bbc59c7559b3ffdfe0f7d935cd17e99d483cb122cd29b593c3e04c0166"
    )
