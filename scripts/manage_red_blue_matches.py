"""Instructor CLI for pairing, inspecting, and resetting Arena matches."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import requests


def _load_local_ctfd_key() -> str:
    key = os.getenv("ARENA_ADMIN_KEY", "")
    if key:
        return key
    dotenv = Path(__file__).resolve().parents[1] / ".env"
    try:
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("CTFD_SECRET_KEY="):
                value = line.split("=", 1)[1].strip().strip("\"'")
                return value
    except OSError:
        pass
    return ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("ARENA_PUBLIC_URL", "http://localhost:8004"))
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="pair two CTFd accounts into a new match")
    create.add_argument("--red-user-id", type=int, required=True)
    create.add_argument("--red-username", required=True)
    create.add_argument("--blue-user-id", type=int, required=True)
    create.add_argument("--blue-username", required=True)
    solo = commands.add_parser("create-solo", help="let one CTFd account play both roles")
    solo.add_argument("--user-id", type=int, required=True)
    solo.add_argument("--username", required=True)
    commands.add_parser("list", help="list matches and recent activity")
    show = commands.add_parser("show", help="show a match and its interaction history")
    show.add_argument("match_id")
    reset = commands.add_parser("reset", help="clear a match's Arena history and artifacts")
    reset.add_argument("match_id")
    args = parser.parse_args(argv)

    key = _load_local_ctfd_key()
    if not key:
        parser.error("Set ARENA_ADMIN_KEY or make CTFD_SECRET_KEY available in the repository .env file.")
    headers = {"X-Arena-Admin-Key": key}
    base = args.url.rstrip("/")
    try:
        if args.command == "create":
            response = requests.post(
                f"{base}/admin/matches",
                headers=headers,
                json={
                    "red_user_id": args.red_user_id,
                    "red_username": args.red_username,
                    "blue_user_id": args.blue_user_id,
                    "blue_username": args.blue_username,
                },
                timeout=10,
            )
        elif args.command == "create-solo":
            response = requests.post(
                f"{base}/admin/matches/solo",
                headers=headers,
                json={"user_id": args.user_id, "username": args.username},
                timeout=10,
            )
        elif args.command == "list":
            response = requests.get(f"{base}/admin/matches", headers=headers, timeout=10)
        elif args.command == "show":
            response = requests.get(f"{base}/admin/matches/{args.match_id}", headers=headers, timeout=10)
        else:
            response = requests.post(f"{base}/admin/matches/{args.match_id}/reset", headers=headers, timeout=10)
        response.raise_for_status()
        print(response.text)
        return 0
    except requests.RequestException as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        print(f"Arena request failed: {detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
