"""Per-account Blue access backed by a shared successful Red artifact."""

from __future__ import annotations

import logging
import os

import requests


logger = logging.getLogger("ctfd.workspace_launcher.xray_gate")
RED_NAME = "X-Ray Red - Adversarial Evasion"
BLUE_NAME = "X-Ray Blue - Lightweight Defense"


def active_red_artifact(payload: dict) -> bool:
    """Accept only the evaluator's currently published successful Red artifact."""
    result = payload.get("result")
    return (
        payload.get("phase") in {"blue", "complete"}
        and isinstance(result, dict)
        and result.get("success") is True
        and bool(result.get("run_id"))
        and bool(payload.get("artifact_url"))
    )


def red_solved_by_user(user) -> bool:
    """Require this CTFd user, not merely someone in the class, to solve Red."""
    if user is None:
        return False
    from CTFd.models import Challenges, Solves

    red = Challenges.query.filter_by(name=RED_NAME).first()
    return bool(red and Solves.query.filter_by(challenge_id=red.id, user_id=user.id).first())


def reconcile_blue_visibility() -> bool:
    """Keep Red as Blue's prerequisite and verify the shared round artifact."""
    from CTFd.cache import clear_challenges
    from CTFd.models import Challenges, Solves, db

    red = Challenges.query.filter_by(name=RED_NAME).first()
    blue = Challenges.query.filter_by(name=BLUE_NAME).first()
    if red is None or blue is None:
        return False

    solves = Solves.query.filter_by(challenge_id=red.id).count()
    ready = False
    if solves:
        evaluator_url = os.getenv("XRAY_REDBLUE_URL", "http://xray-redblue:5000").rstrip("/")
        try:
            response = requests.get(f"{evaluator_url}/api/round/latest", timeout=3)
            if response.status_code == 404:
                ready = False
            else:
                response.raise_for_status()
                ready = active_red_artifact(response.json())
        except (requests.RequestException, ValueError):
            logger.warning("Could not verify the active X-Ray Red artifact", exc_info=True)
            ready = False

    requirements = {"prerequisites": [red.id], "anonymize": "preview"}
    if blue.state != "visible" or blue.requirements != requirements:
        blue.state = "visible"
        blue.requirements = requirements
        db.session.commit()
        clear_challenges()
        logger.info("X-Ray Blue now requires each user's Red solve (%d total Red solve(s))", solves)
    return ready
