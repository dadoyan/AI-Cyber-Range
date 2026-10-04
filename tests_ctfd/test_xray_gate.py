"""Each account needs its own Red solve plus the shared live Red artifact."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import requests


GATE_PATH = Path(__file__).resolve().parents[1] / "ctfd/plugins/workspace_launcher/xray_gate.py"
spec = importlib.util.spec_from_file_location("xray_gate_under_test", GATE_PATH)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class Query:
    def __init__(self, items):
        self.items = items

    def filter_by(self, **conditions):
        return Query([item for item in self.items if all(getattr(item, k) == v for k, v in conditions.items())])

    def first(self):
        return self.items[0] if self.items else None

    def count(self):
        return len(self.items)


def test_blue_gate_requires_solve_and_current_successful_artifact():
    red = SimpleNamespace(id=6, name=gate.RED_NAME, state="visible")
    blue = SimpleNamespace(id=7, name=gate.BLUE_NAME, state="hidden", requirements=None)
    solves = []
    commits = []
    cache_clears = []
    models = ModuleType("CTFd.models")
    models.Challenges = SimpleNamespace(query=Query([red, blue]))
    models.Solves = SimpleNamespace(query=Query(solves))
    models.db = SimpleNamespace(session=SimpleNamespace(commit=lambda: commits.append(True)))
    cache = ModuleType("CTFd.cache")
    cache.clear_challenges = lambda: cache_clears.append(True)

    def response(payload):
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None, json=lambda: payload)

    with patch.dict(sys.modules, {"CTFd.models": models, "CTFd.cache": cache}):
        with patch.object(gate.requests, "get") as get:
            assert gate.reconcile_blue_visibility() is False
            assert blue.state == "visible"
            assert blue.requirements == {"prerequisites": [red.id], "anonymize": "preview"}
            get.assert_not_called()
            solves.append(SimpleNamespace(challenge_id=red.id, user_id=1))
            assert gate.red_solved_by_user(None) is False
            assert gate.red_solved_by_user(SimpleNamespace(id=1)) is True
            assert gate.red_solved_by_user(SimpleNamespace(id=2)) is False
            get.return_value = response({"phase": "red", "result": {"success": True}})
            assert gate.reconcile_blue_visibility() is False
            get.return_value = response({
                "phase": "blue", "result": {"success": True, "run_id": "active"},
                "artifact_url": "/api/round/latest/artifact?run_id=active",
            })
            assert gate.reconcile_blue_visibility() is True
            assert blue.state == "visible"
            assert blue.requirements == {"prerequisites": [red.id], "anonymize": "preview"}
            get.side_effect = requests.ConnectionError("offline")
            assert gate.reconcile_blue_visibility() is False
            assert blue.state == "visible"
            assert blue.requirements == {"prerequisites": [red.id], "anonymize": "preview"}
    assert len(commits) == len(cache_clears) == 1
