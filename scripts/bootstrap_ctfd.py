#!/usr/bin/env python3
"""Create or update the supported range challenges in CTFd via its v1 API."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_CATEGORY = "Adversarial and Robust AI challenge"
PATCHGUARD_CHALLENGE_NAME = "PatchGuard - Adversarial Patch Defense"
PATCHGUARD_CHALLENGE_CATEGORY = CHALLENGE_CATEGORY
XRAY_RED_CHALLENGE_NAME = "X-Ray Red - Adversarial Evasion"
XRAY_BLUE_CHALLENGE_NAME = "X-Ray Blue - Lightweight Defense"
XRAY_CHALLENGE_CATEGORY = CHALLENGE_CATEGORY
LLM_SAFETY_CHALLENGE_NAME = "Breaking LLMs - Prompt Safety"
LLM_SAFETY_CHALLENGE_CATEGORY = CHALLENGE_CATEGORY
HOMEPAGE_TITLE = "AI Cyber Range"


def read_dotenv(path: Path = ROOT / ".env") -> dict[str, str]:
    """Read simple KEY=value entries, without overriding process environment."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def setting(name: str, dotenv: dict[str, str], default: str | None = None) -> str | None:
    return os.environ.get(name) or dotenv.get(name) or default


class CTFdAPI:
    def __init__(self, base_url: str, token: str, timeout: int = 20):
        self.base_url = base_url.rstrip("/") + "/api/v1"
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Token {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            self.base_url + path,
            data=body,
            headers=self.headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code in (401, 403):
                raise RuntimeError(
                    f"CTFd API authentication/authorization failed (HTTP {exc.code}); "
                    "check CTFD_ADMIN_TOKEN is an administrator access token"
                ) from exc
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"CTFd API {method} {path} failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach CTFd at {self.base_url}: {exc.reason}") from exc
        if not result.get("success", False):
            raise RuntimeError(f"CTFd API {method} {path} returned an error: {result}")
        return result


def homepage_content() -> str:
    """Return generic welcome information that does not expose participant actions."""
    return """<div class="container">
  <div class="row">
    <div class="col-md-10 offset-md-1">
      <h1>Welcome to the AI Cyber Range</h1>
      <p>A hands-on learning environment for exploring adversarial machine learning through guided exercises and notebook workspaces.</p>
      <h2>Getting started</h2>
      <p>Create a CTFd account or sign in to access challenge instructions, launch a personal Jupyter workspace, and submit flags for points.</p>
      <p>After signing in, use the navigation menu to continue.</p>
    </div>
  </div>
</div>"""


def synchronize_homepage(api: CTFdAPI) -> dict[str, Any]:
    """Update CTFd's existing index page without changing its visibility settings."""
    listing = api.request("GET", f"/pages?{urlencode({'route': 'index'})}")
    matches = [item for item in listing["data"] if item.get("route") == "index"]
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one CTFd homepage page with route 'index'. Complete CTFd setup first."
        )

    page_id = matches[0]["id"]
    current = api.request("GET", f"/pages/{page_id}")["data"]
    payload = {
        "title": HOMEPAGE_TITLE,
        "content": homepage_content(),
    }
    if all(current.get(key) == value for key, value in payload.items()):
        action = "unchanged"
    else:
        api.request("PATCH", f"/pages/{page_id}", payload)
        action = "updated"
    return {"homepage_id": page_id, "homepage_action": action, "title": HOMEPAGE_TITLE}


def synchronize_patchguard_challenge(
    api: CTFdAPI,
    *,
    flag: str,
    points: int,
    participant_url: str,
    clean_threshold: float,
    robust_threshold: float,
) -> dict[str, Any]:
    """Idempotently create or update the independent PatchGuard challenge."""
    template = (ROOT / "ctfd" / "challenges" / "patchguard.md").read_text(encoding="utf-8")
    description = (
        template.replace("{{PARTICIPANT_URL}}", participant_url.rstrip("/"))
        .replace("{{CLEAN_THRESHOLD}}", f"{clean_threshold:.3f}")
        .replace("{{ROBUST_THRESHOLD}}", f"{robust_threshold:.3f}")
    )
    payload = {
        "name": PATCHGUARD_CHALLENGE_NAME,
        "category": PATCHGUARD_CHALLENGE_CATEGORY,
        "description": description,
        "value": points,
        "type": "standard",
        "state": "visible",
    }
    query = urlencode({"name": PATCHGUARD_CHALLENGE_NAME, "view": "admin"})
    listing = api.request("GET", f"/challenges?{query}")
    matches = [item for item in listing["data"] if item.get("name") == PATCHGUARD_CHALLENGE_NAME]
    if len(matches) > 1:
        raise RuntimeError(
            f"Found {len(matches)} challenges named {PATCHGUARD_CHALLENGE_NAME!r}; remove duplicates first"
        )
    if matches:
        challenge = matches[0]
        if challenge.get("type") != "standard":
            raise RuntimeError("Existing PatchGuard challenge is not a standard challenge")
        challenge_id = challenge["id"]
        api.request("PATCH", f"/challenges/{challenge_id}", payload)
        action = "updated"
    else:
        created = api.request("POST", "/challenges", payload)
        challenge_id = created["data"]["id"]
        action = "created"

    flags_response = api.request("GET", f"/flags?{urlencode({'challenge_id': challenge_id})}")
    static_flags = [item for item in flags_response["data"] if item.get("type") == "static"]
    if len(static_flags) > 1:
        raise RuntimeError("The PatchGuard challenge has multiple static flags; remove extras in CTFd first")
    if static_flags:
        current_flag = static_flags[0]
        if current_flag.get("content") != flag:
            api.request("PATCH", f"/flags/{current_flag['id']}", {"content": flag})
            flag_action = "updated"
        else:
            flag_action = "unchanged"
    else:
        api.request("POST", "/flags", {
            "challenge_id": challenge_id,
            "type": "static",
            "content": flag,
            "data": "",
        })
        flag_action = "created"
    return {
        "challenge_id": challenge_id,
        "challenge_action": action,
        "flag_action": flag_action,
        "name": PATCHGUARD_CHALLENGE_NAME,
        "category": PATCHGUARD_CHALLENGE_CATEGORY,
        "points": points,
        "flag_matches_target": True,
        "clean_threshold": clean_threshold,
        "robust_threshold": robust_threshold,
    }


def synchronize_optional_standard_challenge(
    api: CTFdAPI,
    *,
    name: str,
    description_file: str,
    flag: str,
    points: int,
    participant_url: str,
    category: str = XRAY_CHALLENGE_CATEGORY,
    state: str = "visible",
    requirements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Idempotently create or update one standard challenge and its flag."""
    template = (ROOT / "ctfd" / "challenges" / description_file).read_text(encoding="utf-8")
    payload = {
        "name": name,
        "category": category,
        "description": template.replace("{{PARTICIPANT_URL}}", participant_url.rstrip("/")),
        "value": points,
        "type": "standard",
        "state": state,
    }
    if requirements is not None:
        payload["requirements"] = requirements
    query = urlencode({"name": name, "view": "admin"})
    listing = api.request("GET", f"/challenges?{query}")
    matches = [item for item in listing["data"] if item.get("name") == name]
    if len(matches) > 1:
        raise RuntimeError(f"Found {len(matches)} challenges named {name!r}; remove duplicates first")
    if matches:
        challenge = matches[0]
        if challenge.get("type") != "standard":
            raise RuntimeError(f"Existing challenge {name!r} is not a standard challenge")
        challenge_id = challenge["id"]
        api.request("PATCH", f"/challenges/{challenge_id}", payload)
        challenge_action = "updated"
    else:
        created = api.request("POST", "/challenges", payload)
        challenge_id = created["data"]["id"]
        challenge_action = "created"

    flags = api.request("GET", f"/flags?{urlencode({'challenge_id': challenge_id})}")["data"]
    static_flags = [item for item in flags if item.get("type") == "static"]
    if len(static_flags) > 1:
        raise RuntimeError(f"Challenge {name!r} has multiple static flags; remove extras in CTFd first")
    if static_flags:
        current_flag = static_flags[0]
        if current_flag.get("content") != flag:
            api.request("PATCH", f"/flags/{current_flag['id']}", {"content": flag})
            flag_action = "updated"
        else:
            flag_action = "unchanged"
    else:
        api.request("POST", "/flags", {
            "challenge_id": challenge_id,
            "type": "static",
            "content": flag,
            "data": "",
        })
        flag_action = "created"
    return {
        "challenge_id": challenge_id,
        "challenge_action": challenge_action,
        "flag_action": flag_action,
        "name": name,
        "category": category,
        "points": points,
        "state": state,
    }


def synchronize_xray_redblue_challenges(
    api: CTFdAPI,
    *,
    red_flag: str,
    blue_flag: str,
    red_points: int,
    blue_points: int,
    participant_url: str,
) -> dict[str, Any]:
    red = synchronize_optional_standard_challenge(
            api,
            name=XRAY_RED_CHALLENGE_NAME,
            description_file="xray_red.md",
            flag=red_flag,
            points=red_points,
            participant_url=participant_url,
        )
    blue = synchronize_optional_standard_challenge(
            api,
            name=XRAY_BLUE_CHALLENGE_NAME,
            description_file="xray_blue.md",
            flag=blue_flag,
            points=blue_points,
            participant_url=participant_url,
            requirements={"prerequisites": [red["challenge_id"]], "anonymize": "preview"},
        )
    return {"red": red, "blue": blue}


def synchronize_llm_safety_challenge(
    api: CTFdAPI, *, flag: str, points: int, participant_url: str
) -> dict[str, Any]:
    """Keep the new challenge independent of existing flags and solves."""
    return synchronize_optional_standard_challenge(
        api, name=LLM_SAFETY_CHALLENGE_NAME,
        description_file="llm_safety.md", flag=flag, points=points,
        participant_url=participant_url, category=LLM_SAFETY_CHALLENGE_CATEGORY,
    )


def main() -> int:
    dotenv = read_dotenv()
    token = setting("CTFD_ADMIN_TOKEN", dotenv)
    base_url = setting("CTFD_URL", dotenv)
    participant_url = setting("PARTICIPANT_URL", dotenv)
    patchguard_flag = setting("PATCHGUARD_FLAG", dotenv)
    patchguard_points_text = setting("PATCHGUARD_POINTS", dotenv, "300")
    patchguard_clean_threshold_text = setting("PATCHGUARD_CLEAN_THRESHOLD", dotenv)
    patchguard_robust_threshold_text = setting("PATCHGUARD_ROBUST_THRESHOLD", dotenv)
    xray_enabled = (setting("XRAY_REDBLUE_ENABLED", dotenv, "false") or "false").strip().lower() in {"1", "true", "yes", "on"}
    llm_safety_enabled = (setting("LLM_SAFETY_ENABLED", dotenv, "false") or "false").strip().lower() in {"1", "true", "yes", "on"}
    if not patchguard_flag:
        raise RuntimeError("Set PATCHGUARD_FLAG in the environment or .env")
    if not token:
        raise RuntimeError("Set CTFD_ADMIN_TOKEN in the environment or .env after first-run setup")
    if not base_url or not base_url.startswith(("http://", "https://")):
        raise RuntimeError("CTFD_URL must be an http(s) URL")
    if not participant_url or not participant_url.startswith(("http://", "https://")):
        raise RuntimeError("PARTICIPANT_URL must be an externally reachable http(s) URL")
    try:
        patchguard_points = int(patchguard_points_text or "300")
        patchguard_clean_threshold = float(patchguard_clean_threshold_text)
        patchguard_robust_threshold = float(patchguard_robust_threshold_text)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Set valid calibrated PatchGuard thresholds and PATCHGUARD_POINTS") from exc
    if patchguard_points <= 0:
        raise RuntimeError("PATCHGUARD_POINTS must be a positive integer")
    if not 0.0 <= patchguard_clean_threshold <= 1.0 or not 0.0 <= patchguard_robust_threshold <= 1.0:
        raise RuntimeError("PatchGuard accuracy thresholds must be between 0 and 1")

    api = CTFdAPI(base_url, token)
    result: dict[str, Any] = {}
    result["patchguard"] = synchronize_patchguard_challenge(
        api,
        flag=patchguard_flag,
        points=patchguard_points,
        participant_url=participant_url,
        clean_threshold=patchguard_clean_threshold,
        robust_threshold=patchguard_robust_threshold,
    )
    if xray_enabled:
        xray_red_flag = setting("XRAY_RED_FLAG", dotenv)
        xray_blue_flag = setting("XRAY_BLUE_FLAG", dotenv)
        if not xray_red_flag or not xray_blue_flag:
            raise RuntimeError("Set both XRAY_RED_FLAG and XRAY_BLUE_FLAG when XRAY_REDBLUE_ENABLED=true")
        try:
            xray_red_points = int(setting("XRAY_RED_POINTS", dotenv, "200") or "200")
            xray_blue_points = int(setting("XRAY_BLUE_POINTS", dotenv, "300") or "300")
        except ValueError as exc:
            raise RuntimeError("XRAY_RED_POINTS and XRAY_BLUE_POINTS must be positive integers") from exc
        if xray_red_points <= 0 or xray_blue_points <= 0:
            raise RuntimeError("XRAY_RED_POINTS and XRAY_BLUE_POINTS must be positive integers")
        result["xray_redblue"] = synchronize_xray_redblue_challenges(
            api,
            red_flag=xray_red_flag,
            blue_flag=xray_blue_flag,
            red_points=xray_red_points,
            blue_points=xray_blue_points,
            participant_url=participant_url,
        )
    if llm_safety_enabled:
        llm_safety_flag = setting("LLM_SAFETY_FLAG", dotenv)
        allowed_user_id = setting("LLM_SAFETY_ALLOWED_USER_ID", dotenv)
        try:
            llm_safety_points = int(setting("LLM_SAFETY_POINTS", dotenv, "400") or "400")
        except ValueError as exc:
            raise RuntimeError("LLM_SAFETY_POINTS must be a positive integer") from exc
        if not llm_safety_flag or not allowed_user_id or not allowed_user_id.isdecimal() or int(allowed_user_id) <= 0 or llm_safety_points <= 0:
            raise RuntimeError("Set LLM_SAFETY_FLAG, a positive LLM_SAFETY_ALLOWED_USER_ID, and LLM_SAFETY_POINTS")
        result["llm_safety"] = synchronize_llm_safety_challenge(
            api, flag=llm_safety_flag, points=llm_safety_points,
            participant_url=participant_url,
        )
    result["homepage"] = synchronize_homepage(api)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
