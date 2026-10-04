"""Move legacy root notebooks into exercise folders without losing player edits.

Run inside an existing participant container after copying the current
participant/workspace tree to /tmp/ai-cyber-workspace-template.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


NOTEBOOKS = {
    "red_team.ipynb": "xray_red/red_team.ipynb",
    "xray_red_blue_starter.ipynb": "xray_red/xray_red_blue_starter.ipynb",
    "blue_team.ipynb": "xray_blue/blue_team.ipynb",
    "llm_safety_starter.ipynb": "llm_safety/llm_safety_starter.ipynb",
}

SUPPORT_FILES = {
    "xray_red/red_blue_arena.py": "red_blue_arena.py",
    "xray_blue/red_blue_arena.py": "red_blue_arena.py",
    "llm_safety/llm_safety_optimizer.py": "llm_safety_optimizer.py",
}


def copy_missing(source: Path, destination: Path) -> bool:
    if destination.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def organize(workspace: Path, template: Path) -> dict[str, int]:
    counts = {"moved": 0, "restored": 0, "backed_up": 0, "support": 0}
    for old_name, new_name in NOTEBOOKS.items():
        old = workspace / old_name
        new = workspace / new_name
        new.parent.mkdir(parents=True, exist_ok=True)
        if old.exists():
            if new.exists():
                backup = workspace / "previous_notebooks" / old_name
                backup.parent.mkdir(parents=True, exist_ok=True)
                suffix = 1
                while backup.exists():
                    backup = workspace / "previous_notebooks" / f"{old.stem}.{suffix}{old.suffix}"
                    suffix += 1
                old.rename(backup)
                counts["backed_up"] += 1
            else:
                old.rename(new)
                counts["moved"] += 1
        elif (template / new_name).exists() and copy_missing(template / new_name, new):
            counts["restored"] += 1

    for new_name in ("patchguard/patchguard_starter.ipynb", "xray_blue/blue_starter.ipynb",
                     "xray_red/red_team_fgsm.ipynb"):
        if copy_missing(template / new_name, workspace / new_name):
            counts["restored"] += 1

    for target_name, source_name in SUPPORT_FILES.items():
        source = workspace / source_name
        if not source.exists():
            source = template / source_name
        if copy_missing(source, workspace / target_name):
            counts["support"] += 1

    for folder in ("xray_red", "xray_blue"):
        helper = workspace / folder / "red_blue_arena.py"
        original = helper.read_text(encoding="utf-8")
        updated = original.replace('WORKSPACE = Path("/workspace")',
                                   "WORKSPACE = Path(__file__).resolve().parent")
        if updated != original:
            helper.write_text(updated, encoding="utf-8")

    red_notebook = workspace / "xray_red/red_team.ipynb"
    if red_notebook.exists():
        notebook = json.loads(red_notebook.read_text(encoding="utf-8"))
        identity_cell = next((cell for cell in notebook["cells"]
                              if cell.get("id") == "red-identity"), None)
        if identity_cell is not None and not any(
            "import red_blue_arena as arena" in line for line in identity_cell["source"]
        ):
            index = next((i for i, line in enumerate(identity_cell["source"])
                          if "identity = arena.identity()" in line), None)
            if index is not None:
                identity_cell["source"][index:index] = [
                    "import red_blue_arena as arena\n",
                    "from IPython.display import display, Markdown, HTML, Image as DisplayImage\n",
                    "\n",
                ]
                red_notebook.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                                        encoding="utf-8")

    start = workspace / "start_here.ipynb"
    copy_missing(template / "start_here.ipynb", start)
    original = start.read_text(encoding="utf-8")
    updated = original
    for old_name, new_name in NOTEBOOKS.items():
        updated = updated.replace(f"/lab/tree/{old_name}", f"/lab/tree/{new_name}")
    if updated != original:
        start.write_text(updated, encoding="utf-8")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--template", type=Path, default=Path("/tmp/ai-cyber-workspace-template"))
    args = parser.parse_args()
    print(organize(args.workspace, args.template))
