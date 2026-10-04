"""Remove retired exercise content without resetting a participant workspace.

Only retired exercise directories, known legacy support files/backups, and
range-owned welcome cells are changed. Retained exercise notebooks and
additional student cells are preserved. Safe to rerun on an existing volume.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from pathlib import Path


RETIRED_DIRECTORIES = ("granny", "granny2", "granny_infinity")
RETIRED_NOTEBOOKS = (
    "granny_starter.ipynb", "granny2_starter.ipynb", "granny_infinity_starter.ipynb",
)
RETIRED_FILES = (
    *RETIRED_NOTEBOOKS,
    "score_display.py", "t_wolf.jpg", "my_granny_infinity.png",
    "granny_infinity_stress_success.png", "audit_granny_infinity.png",
    "granny2_reference_adversarial.png",
    ".granny_infinity_starter.before-strict-20261001.ipynb",
    ".granny_infinity_starter.before-ssim-20261001.ipynb",
    # Removal of the obsolete mixed audit report was explicitly authorised.
    "audit-results.json",
    "audit-fixtures/granny.png", "audit-fixtures/granny2.png",
    "audit-fixtures/infinity.png", "audit-fixtures/apple.jpg",
    "audit-fixtures/infinity-before-ssim-20261001.png",
    "audit-fixtures/infinity-0.08-before-strict.png",
)
WELCOME_CELL_IDS = {"start-intro", "readiness-help", "readiness-check", "exercise-guide"}


def checked_path(root: Path, path: Path) -> Path:
    """Check the final parent before touching a file or following a directory."""
    if not path.parent.resolve().is_relative_to(root):
        raise ValueError(f"Refusing to modify a path outside the workspace: {path}")
    if not path.is_symlink() and not path.resolve().is_relative_to(root):
        raise ValueError(f"Refusing to follow a path outside the workspace: {path}")
    return path


def remove_path(root: Path, path: Path) -> bool:
    checked_path(root, path)
    if path.is_symlink():
        path.unlink()
        return True
    if path.is_dir():
        # The resolved absolute target was checked above. shutil never follows
        # directory symlinks, including links within the retired directory.
        shutil.rmtree(path)
        return True
    if path.exists():
        path.unlink()
        return True
    return False


def is_retired_notebook(path: Path) -> bool:
    """Identify a generic checkpoint by exercise code, not its filename."""
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"]).lower()
    except (OSError, ValueError, KeyError, TypeError):
        return False
    # A generic checkpoint can contain retained work. Never remove such a mix.
    if any(marker in source for marker in (
        "patchguard", "xray", "x-ray", "red_blue_arena", "llm_safety", "qwen",
    )):
        return False
    return (
        "query_granny2" in source or "query_granny_infinity" in source
        or ("query_target" in source and "http://target:8000" in source)
        or ("granny" in source and "t_wolf.jpg" in source)
    )


def update_welcome(root: Path, template: Path) -> bool:
    path = checked_path(root, root / "start_here.ipynb")
    if path.is_symlink():
        raise ValueError("Refusing to overwrite a linked welcome notebook")
    canonical = json.loads((template / "start_here.ipynb").read_text(encoding="utf-8"))
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        replacements = {cell["id"]: cell for cell in canonical["cells"]
                        if cell.get("id") in WELCOME_CELL_IDS}
        cells = []
        found = set()
        for cell in current["cells"]:
            cell_id = cell.get("id")
            if cell_id in replacements:
                cells.append(copy.deepcopy(replacements[cell_id]))
                found.add(cell_id)
            else:
                cells.append(cell)
        for cell in canonical["cells"]:
            if cell.get("id") in replacements and cell["id"] not in found:
                cells.append(copy.deepcopy(cell))
        current["cells"] = cells
    else:
        current = canonical
    text = json.dumps(current, indent=1, ensure_ascii=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def retire(workspace: Path, template: Path) -> dict[str, int]:
    root = workspace.resolve(strict=True)
    if not root.is_dir() or root == Path(root.anchor):
        raise ValueError("Provide a dedicated participant workspace directory")
    # Validate templates before any destructive operation.
    json.loads((template / "start_here.ipynb").read_text(encoding="utf-8"))
    readme_text = (template / "README.md").read_text(encoding="utf-8")
    retired_paths = [root / name for name in (*RETIRED_DIRECTORIES, *RETIRED_FILES)]
    for dirname in ("previous_notebooks", ".ipynb_checkpoints", "__pycache__"):
        folder = checked_path(root, root / dirname)
        if folder.is_symlink():
            continue
        if folder.is_dir():
            for path in folder.iterdir():
                if re.fullmatch(
                    r"(?:granny|granny2|granny_infinity)_starter"
                    r"(?:\.\d+|-checkpoint)?\.ipynb", path.name,
                ) or (dirname == "__pycache__" and path.name.startswith("score_display.")
                      and path.suffix == ".pyc"):
                    retired_paths.append(path)
                elif (dirname == ".ipynb_checkpoints"
                      and path.name == "Untitled-checkpoint.ipynb"
                      and not path.is_symlink()
                      and is_retired_notebook(path)):
                    retired_paths.append(path)
    # Check every resolved deletion target before deleting the first one.
    for path in retired_paths:
        checked_path(root, path)
    removed = sum(remove_path(root, path) for path in retired_paths)
    updated = int(update_welcome(root, template))
    readme = checked_path(root, root / "README.md")
    if not readme.is_symlink() and (
        not readme.exists() or readme.read_text(encoding="utf-8").startswith("# Granny challenge workspace")
    ):
        readme.write_text(readme_text, encoding="utf-8")
        updated += 1
    # A saved checkpoint can otherwise resurrect the old welcome page.
    checkpoint = checked_path(root, root / ".ipynb_checkpoints/start_here-checkpoint.ipynb")
    if checkpoint.exists() and not checkpoint.is_symlink():
        raw = checkpoint.read_text(encoding="utf-8")
        if "granny" in raw.lower():
            checkpoint.write_text((root / "start_here.ipynb").read_text(encoding="utf-8"), encoding="utf-8")
            updated += 1
    return {"removed": removed, "updated": updated}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--template", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(retire(arguments.workspace, arguments.template)))
