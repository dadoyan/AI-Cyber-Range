import json
from urllib.error import HTTPError
from io import BytesIO
import unittest
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

from scripts.bootstrap_ctfd import (
    CTFdAPI,
    main,
    synchronize_patchguard_challenge,
    synchronize_xray_redblue_challenges,
    synchronize_llm_safety_challenge,
)

class InMemoryCTFd:
    def __init__(self):
        self.challenges = []
        self.flags = []
        self.next_id = 1
        self.calls = []

    def request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        if method == "GET" and path.startswith("/challenges?"):
            return {"success": True, "data": list(self.challenges)}
        if method == "POST" and path == "/challenges":
            item = {"id": self.next_id, **payload}
            self.next_id += 1
            self.challenges.append(item)
            return {"success": True, "data": item}
        if method == "PATCH" and path.startswith("/challenges/"):
            challenge_id = int(path.rsplit("/", 1)[1])
            item = next(item for item in self.challenges if item["id"] == challenge_id)
            item.update(payload)
            return {"success": True, "data": item}
        if method == "GET" and path.startswith("/flags?"):
            query = parse_qs(urlsplit(path).query)
            challenge_id = int(query["challenge_id"][0])
            return {
                "success": True,
                "data": [
                    item for item in self.flags
                    if int(item["challenge_id"]) == challenge_id
                ],
            }
        if method == "POST" and path == "/flags":
            item = {"id": self.next_id, **payload}
            self.next_id += 1
            self.flags.append(item)
            return {"success": True, "data": item}
        if method == "PATCH" and path.startswith("/flags/"):
            flag_id = int(path.rsplit("/", 1)[1])
            item = next(item for item in self.flags if item["id"] == flag_id)
            item.update(payload)
            return {"success": True, "data": item}
        raise AssertionError((method, path, payload))

class BootstrapTests(unittest.TestCase):
    @patch("scripts.bootstrap_ctfd.os.environ", {})
    @patch("scripts.bootstrap_ctfd.read_dotenv")
    @patch("scripts.bootstrap_ctfd.CTFdAPI")
    @patch("scripts.bootstrap_ctfd.synchronize_homepage", return_value={"homepage_action": "unchanged"})
    @patch("builtins.print")
    def test_main_synchronizes_only_supported_challenges_idempotently(
        self, _print, _homepage, api_class, read_dotenv
    ):
        read_dotenv.return_value = {
            "PATCHGUARD_FLAG": "RANGE{patchguard}",
            "CTFD_ADMIN_TOKEN": "test-token",
            "CTFD_URL": "http://ctfd",
            "PARTICIPANT_URL": "http://lab",
            "PATCHGUARD_CLEAN_THRESHOLD": "0.6",
            "PATCHGUARD_ROBUST_THRESHOLD": "0.3",
            "XRAY_REDBLUE_ENABLED": "true",
            "XRAY_RED_FLAG": "RANGE{xray_red}",
            "XRAY_BLUE_FLAG": "RANGE{xray_blue}",
            "LLM_SAFETY_ENABLED": "true",
            "LLM_SAFETY_FLAG": "RANGE{llm}",
            "LLM_SAFETY_ALLOWED_USER_ID": "2",
        }
        api = InMemoryCTFd()
        api_class.return_value = api

        self.assertEqual(main(), 0)
        first_ids = {item["name"]: item["id"] for item in api.challenges}
        self.assertEqual(main(), 0)

        self.assertEqual({item["name"]: item["id"] for item in api.challenges}, first_ids)
        self.assertEqual(set(first_ids), {
            "PatchGuard - Adversarial Patch Defense",
            "X-Ray Red - Adversarial Evasion",
            "X-Ray Blue - Lightweight Defense",
            "Breaking LLMs - Prompt Safety",
        })
        self.assertEqual(len(api.flags), 4)
        blue = next(item for item in api.challenges if item["name"].startswith("X-Ray Blue"))
        self.assertEqual(blue["requirements"]["prerequisites"], [first_ids["X-Ray Red - Adversarial Evasion"]])

    def test_llm_safety_bootstrap_is_independent_and_idempotent(self):
        api = InMemoryCTFd()
        first = synchronize_llm_safety_challenge(
            api, flag="flag{llm}", points=400, participant_url="http://localhost:7000"
        )
        second = synchronize_llm_safety_challenge(
            api, flag="flag{llm}", points=400, participant_url="http://localhost:7000"
        )
        self.assertEqual(first["challenge_action"], "created")
        self.assertEqual(second["flag_action"], "unchanged")
        self.assertEqual(len(api.challenges), 1)
        self.assertEqual(len(api.flags), 1)
        self.assertIn("6 of 10", api.challenges[0]["description"])
        self.assertIn("llm_safety/llm_safety_starter.ipynb", api.challenges[0]["description"])
        self.assertIn(
            "](/workspace-launch?notebook=llm_safety/llm_safety_starter.ipynb)",
            api.challenges[0]["description"],
        )
        self.assertNotIn("localhost:7000/workspace-launch", api.challenges[0]["description"])

    def test_bootstrap_creates_then_updates_without_duplicates(self):
        api = InMemoryCTFd()
        first = synchronize_patchguard_challenge(
            api,
            flag="RANGE{shared_flag}",
            points=200,
            participant_url="https://range.example/jupyter",
            clean_threshold=0.6,
            robust_threshold=0.3,
        )
        second = synchronize_patchguard_challenge(
            api,
            flag="RANGE{shared_flag}",
            points=200,
            participant_url="https://range.example/jupyter",
            clean_threshold=0.6,
            robust_threshold=0.3,
        )

        self.assertEqual(first["challenge_action"], "created")
        self.assertEqual(second["challenge_action"], "updated")
        self.assertEqual(len(api.challenges), 1)
        self.assertEqual(len(api.flags), 1)
        self.assertEqual(api.challenges[0]["category"], "Adversarial and Robust AI challenge")
        self.assertEqual(api.challenges[0]["value"], 200)
        self.assertEqual(api.challenges[0]["type"], "standard")
        self.assertIn("Open the PatchGuard workspace", api.challenges[0]["description"])
        self.assertNotIn("https://range.example/jupyter", api.challenges[0]["description"])
        self.assertEqual(api.flags[0]["content"], "RANGE{shared_flag}")
        self.assertEqual(second["flag_action"], "unchanged")

    def test_bootstrap_synchronizes_changed_flag_and_points(self):
        api = InMemoryCTFd()
        synchronize_patchguard_challenge(api, flag="RANGE{old}", points=200, participant_url="http://lab/jupyter", clean_threshold=0.6, robust_threshold=0.3)
        synchronize_patchguard_challenge(api, flag="RANGE{new}", points=350, participant_url="http://lab/jupyter", clean_threshold=0.6, robust_threshold=0.3)
        self.assertEqual(len(api.challenges), 1)
        self.assertEqual(len(api.flags), 1)
        self.assertEqual(api.challenges[0]["value"], 350)
        self.assertEqual(api.flags[0]["content"], "RANGE{new}")

    def test_patchguard_bootstrap_is_independent_and_idempotent(self):
        api = InMemoryCTFd()
        first = synchronize_patchguard_challenge(
            api,
            flag="RANGE{patchguard}",
            points=300,
            participant_url="http://localhost:7000",
            clean_threshold=0.6,
            robust_threshold=0.3,
        )
        second = synchronize_patchguard_challenge(
            api,
            flag="RANGE{patchguard}",
            points=300,
            participant_url="http://localhost:7000",
            clean_threshold=0.6,
            robust_threshold=0.3,
        )
        assert first["challenge_action"] == "created"
        assert second["challenge_action"] == "updated"
        assert second["flag_action"] == "unchanged"
        assert first["name"] == "PatchGuard - Adversarial Patch Defense"
        assert len(api.challenges) == len(api.flags) == 1
        challenge = api.challenges[0]
        assert challenge["category"] == "Adversarial and Robust AI challenge"
        assert challenge["value"] == 300
        assert challenge["type"] == "standard"
        assert "0.600" in challenge["description"]
        assert "0.300" in challenge["description"]
        assert "patchguard%2Fpatchguard_starter.ipynb" in challenge["description"]
        assert api.flags[0]["content"] == "RANGE{patchguard}"

    def test_xray_redblue_challenges_are_independent_and_idempotent(self):
        api = InMemoryCTFd()
        first = synchronize_xray_redblue_challenges(
            api,
            red_flag="RANGE{xray_red}",
            blue_flag="RANGE{xray_blue}",
            red_points=200,
            blue_points=300,
            participant_url="http://localhost:7000",
        )
        second = synchronize_xray_redblue_challenges(
            api,
            red_flag="RANGE{xray_red}",
            blue_flag="RANGE{xray_blue}",
            red_points=200,
            blue_points=300,
            participant_url="http://localhost:7000",
        )
        assert first["red"]["challenge_action"] == "created"
        assert first["blue"]["challenge_action"] == "created"
        assert second["red"]["challenge_action"] == "updated"
        assert second["blue"]["challenge_action"] == "updated"
        assert len(api.challenges) == len(api.flags) == 2
        challenges = {item["name"]: item for item in api.challenges}
        assert challenges["X-Ray Red - Adversarial Evasion"]["value"] == 200
        assert challenges["X-Ray Blue - Lightweight Defense"]["value"] == 300
        assert "xray_red/red_team_fgsm.ipynb" in challenges["X-Ray Red - Adversarial Evasion"]["description"]
        assert "xray_blue/blue_team.ipynb" in challenges["X-Ray Blue - Lightweight Defense"]["description"]
        assert "class-wide" not in challenges["X-Ray Red - Adversarial Evasion"]["description"]
        assert "static" not in challenges["X-Ray Blue - Lightweight Defense"]["description"].lower()
        assert "returns your X-Ray Red flag" in challenges["X-Ray Red - Adversarial Evasion"]["description"]
        assert "validated Red image appears at the top" in challenges["X-Ray Blue - Lightweight Defense"]["description"]
        assert "successful defense returns your X-Ray Blue flag" in challenges["X-Ray Blue - Lightweight Defense"]["description"]
        assert challenges["X-Ray Red - Adversarial Evasion"]["state"] == "visible"
        assert challenges["X-Ray Blue - Lightweight Defense"]["state"] == "visible"
        assert challenges["X-Ray Blue - Lightweight Defense"]["requirements"] == {
            "prerequisites": [first["red"]["challenge_id"]], "anonymize": "preview"
        }
        assert "](/workspace-launch?notebook=xray_red/red_team_fgsm.ipynb)" in challenges["X-Ray Red - Adversarial Evasion"]["description"]
        assert "](/workspace-launch?notebook=xray_blue/blue_team.ipynb)" in challenges["X-Ray Blue - Lightweight Defense"]["description"]
        assert all("localhost:7000/workspace-launch" not in challenge["description"] for challenge in challenges.values())
        assert {item["content"] for item in api.flags} == {"RANGE{xray_red}", "RANGE{xray_blue}"}
        assert second["red"]["flag_action"] == "unchanged"
        assert second["blue"]["flag_action"] == "unchanged"

    @patch("scripts.bootstrap_ctfd.urlopen")
    def test_api_client_sends_supported_token_header(self, mock_urlopen):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"success": True, "data": []}).encode()

        mock_urlopen.return_value = Response()
        api = CTFdAPI("http://localhost:8001", "admin-token")
        result = api.request("GET", "/challenges")
        self.assertEqual(result["data"], [])
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Token admin-token")

    @patch("scripts.bootstrap_ctfd.urlopen")
    def test_api_client_reports_rejected_admin_token(self, mock_urlopen):
        mock_urlopen.side_effect = HTTPError(
            "http://localhost:8001/api/v1/challenges",
            401,
            "Unauthorized",
            {},
            BytesIO(b"unauthorized"),
        )
        api = CTFdAPI("http://localhost:8001", "invalid-token")
        with self.assertRaisesRegex(RuntimeError, "administrator access token"):
            api.request("GET", "/challenges")

if __name__ == "__main__":
    unittest.main()
