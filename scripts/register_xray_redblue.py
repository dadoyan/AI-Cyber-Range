#!/usr/bin/env python3
"""Register the original X-Ray exercise's Red and Blue challenges only."""

from __future__ import annotations

import json

from scripts.bootstrap_ctfd import (CTFdAPI, read_dotenv, setting,
                                    synchronize_xray_redblue_challenges)


def main() -> None:
    values = read_dotenv()
    if (setting("XRAY_REDBLUE_ENABLED", values, "false") or "").lower() not in {"1", "true", "yes", "on"}:
        raise RuntimeError("Set XRAY_REDBLUE_ENABLED=true before registering")
    red_flag = setting("XRAY_RED_FLAG", values)
    blue_flag = setting("XRAY_BLUE_FLAG", values)
    token = setting("CTFD_ADMIN_TOKEN", values)
    url = setting("CTFD_URL", values)
    participant_url = setting("PARTICIPANT_URL", values)
    try:
        red_points = int(setting("XRAY_RED_POINTS", values, "200") or "200")
        blue_points = int(setting("XRAY_BLUE_POINTS", values, "300") or "300")
    except ValueError as exc:
        raise RuntimeError("X-Ray points must be positive integers") from exc
    if not all((red_flag, blue_flag, token, url, participant_url)) or red_flag == blue_flag:
        raise RuntimeError("Configure distinct X-Ray flags, CTFd admin token, and URLs")
    if red_points <= 0 or blue_points <= 0:
        raise RuntimeError("X-Ray points must be positive integers")
    result = synchronize_xray_redblue_challenges(
        CTFdAPI(url, token), red_flag=red_flag, blue_flag=blue_flag,
        red_points=red_points, blue_points=blue_points,
        participant_url=participant_url,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
