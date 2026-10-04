#!/usr/bin/env python3
"""Register only the optional LLM challenge; leave all other CTFd records alone."""

from __future__ import annotations

import json
import os

from scripts.bootstrap_ctfd import (CTFdAPI, read_dotenv, setting,
                                    synchronize_llm_safety_challenge)


def main():
    values = read_dotenv()
    if (setting("LLM_SAFETY_ENABLED", values, "false") or "").lower() not in {"1", "true", "yes", "on"}:
        raise RuntimeError("Set LLM_SAFETY_ENABLED=true before registering")
    flag = setting("LLM_SAFETY_FLAG", values)
    assigned = setting("LLM_SAFETY_ALLOWED_USER_ID", values)
    token = setting("CTFD_ADMIN_TOKEN", values)
    url = setting("CTFD_URL", values)
    participant_url = setting("PARTICIPANT_URL", values)
    try:
        points = int(setting("LLM_SAFETY_POINTS", values, "400") or "400")
    except ValueError as exc:
        raise RuntimeError("LLM_SAFETY_POINTS must be positive") from exc
    if not flag or not assigned or not assigned.isdecimal() or int(assigned) <= 0:
        raise RuntimeError("Configure LLM_SAFETY_FLAG and LLM_SAFETY_ALLOWED_USER_ID")
    if not token or not url or not participant_url or points <= 0:
        raise RuntimeError("Configure CTFd admin token, URLs, and positive points")
    result = synchronize_llm_safety_challenge(
        CTFdAPI(url, token), flag=flag, points=points,
        participant_url=participant_url,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
