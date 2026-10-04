#!/usr/bin/env python3
"""Remove only Docker containers explicitly managed by the Lab Launcher."""

from __future__ import annotations

import json
import subprocess


def main() -> int:
    try:
        result = subprocess.run(
            ["docker", "ps", "--all", "--quiet", "--filter", "label=ai.range.managed=true"],
            check=True,
            capture_output=True,
            text=True,
        )
        ids = [item for item in result.stdout.splitlines() if item.strip()]
        removed = 0
        for container_id in ids:
            inspect = subprocess.run(
                ["docker", "inspect", container_id], check=True, capture_output=True, text=True
            )
            container = json.loads(inspect.stdout)[0]
            if (container.get("Config", {}).get("Labels") or {}).get("ai.range.managed") != "true":
                continue
            print(f"Removing {container.get('Name', '').lstrip('/')} ({container.get('State', {}).get('Status')})")
            subprocess.run(["docker", "rm", "--force", container_id], check=True, capture_output=True, text=True)
            removed += 1
        print(f"Removed {removed} launcher-managed participant container(s).")
        return 0
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Docker cleanup failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
