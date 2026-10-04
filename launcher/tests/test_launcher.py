from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import struct
import tarfile
import zlib
from types import SimpleNamespace

import pytest
from docker.errors import NotFound
from starlette.requests import Request

from launcher.app import main


def configure_db(monkeypatch, tmp_path):
    path = tmp_path / "launcher.sqlite3"
    monkeypatch.setattr(main, "DB_PATH", path)
    main.init_db()
    return path


def insert_workspace(username="student1", user_id=3, port=9001):
    with main.db_connection() as conn:
        conn.execute(
            "INSERT INTO workspaces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"u{user_id}", user_id, username, f"ai-range-u{user_id}", port,
             "test-token", "Running", "created", "updated"),
        )


def make_request(path="/"):
    return Request(
        {
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "query_string": b"",
            "headers": [],
            "server": ("testserver", 80),
            "client": ("testclient", 123),
        }
    )


def test_xray_blue_handoff_copies_active_round_into_personal_workspace(monkeypatch, tmp_path):
    round_id = "a" * 32
    red_run_id = "b" * 32
    artifact = b"PK\x03\x04test-npz"
    template = tmp_path / "blue_starter.ipynb"
    template.write_bytes(b'{"cells": []}')
    monkeypatch.setattr(main, "XRAY_BLUE_TEMPLATE", template)
    state = {"round_id": round_id, "red_run_id": red_run_id, "phase": "blue"}
    latest = {"phase": "blue", "result": {"run_id": red_run_id, "success": True}}
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        data = latest if url.endswith("/api/round/latest") else state
        if url.endswith("/api/round/latest/successes"):
            return SimpleNamespace(content=artifact, raise_for_status=lambda: None)
        return SimpleNamespace(json=lambda: data, raise_for_status=lambda: None)

    class Container:
        def __init__(self):
            self.packed = None

        def exec_run(self, command):
            assert command == ["test", "-f", "/workspace/xray_blue/blue_starter.ipynb"]
            return SimpleNamespace(exit_code=1)

        def put_archive(self, path, packed):
            assert path == "/workspace"
            self.packed = packed
            return True

    container = Container()
    client = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(main.requests, "get", fake_get)
    monkeypatch.setattr(main, "docker_client", lambda: client)
    monkeypatch.setattr(main, "managed_container", lambda docker, row: container)
    main.prepare_xray_blue_workspace({"workspace_id": "u13"})
    assert calls == [
        f"{main.XRAY_REDBLUE_URL}/api/round",
        f"{main.XRAY_REDBLUE_URL}/api/round/latest",
        f"{main.XRAY_REDBLUE_URL}/api/round/latest/successes",
        f"{main.XRAY_REDBLUE_URL}/api/round",
    ]
    with tarfile.open(fileobj=io.BytesIO(container.packed)) as archive:
        names = archive.getnames()
        assert "xray_blue/blue_starter.ipynb" in names
        assert archive.extractfile(f"xray_blue/rounds/{round_id}/successful_red_examples.npz").read() == artifact
        manifest = json.load(archive.extractfile("xray_blue/active_round.json"))
    assert manifest["red_run_id"] == red_run_id
    assert manifest["sha256"] == hashlib.sha256(artifact).hexdigest()


def test_xray_blue_archive_rejects_path_escape():
    with pytest.raises(ValueError, match="round ID"):
        main.xray_blue_archive("../elsewhere", {}, b"zip", None)


def test_identity_validation_and_mismatch_detection(monkeypatch, tmp_path):
    configure_db(monkeypatch, tmp_path)
    assert main.validate_identity("3", " student1 ") == (3, "student1")
    with pytest.raises(ValueError, match="positive whole number"):
        main.validate_identity("0", "student1")
    with pytest.raises(ValueError, match="username"):
        main.validate_identity("3", " ")
    insert_workspace()
    with main.db_connection() as conn:
        assert main.identity_mapping(conn, 3, "student1")["workspace_id"] == "u3"
        with pytest.raises(ValueError, match="already mapped"):
            main.identity_mapping(conn, 3, "different")
        with pytest.raises(ValueError, match="already mapped"):
            main.identity_mapping(conn, 4, "student1")


def test_database_persists_mapping_and_workspace_url(monkeypatch, tmp_path):
    db = configure_db(monkeypatch, tmp_path)
    insert_workspace()
    first = main.get_row_by_uid(3)
    main.init_db()
    second = main.get_row_by_uid(3)
    assert first["workspace_id"] == second["workspace_id"] == "u3"
    assert main.workspace_url(second).endswith(
        ":9001/lab/tree/start_here.ipynb?token=test-token"
    )
    assert db.exists()


def test_identity_resolver_uses_launcher_token_mapping_not_client_identity(
    monkeypatch, tmp_path
):
    configure_db(monkeypatch, tmp_path)
    insert_workspace()
    monkeypatch.setenv("WORKSPACE_IDENTITY_RESOLVE_KEY", "resolver-key")
    arena_token = main.workspace_identity_token("u3")
    with main.db_connection() as conn:
        token_hash = conn.execute(
            "SELECT token_hash FROM workspace_identity_tokens WHERE workspace_id='u3'"
        ).fetchone()[0]
    assert token_hash == hashlib.sha256(arena_token.encode("utf-8")).hexdigest()
    assert token_hash != arena_token

    resolved = main.resolve_workspace_identity(
        main.WorkspaceIdentityResolve(token=arena_token, remote_addr=""),
        x_workspace_identity_key="resolver-key",
    )
    assert resolved == {
        "ctfd_user_id": 3,
        "ctfd_username": "student1",
        "workspace_id": "u3",
        "identity_source": "arena_token",
    }

    # Pre-existing workspaces can resolve through their already-issued token.
    legacy = main.resolve_workspace_identity(
        main.WorkspaceIdentityResolve(token="test-token", remote_addr=""),
        x_workspace_identity_key="resolver-key",
    )
    assert legacy["ctfd_username"] == "student1"
    assert legacy["identity_source"] == "jupyter_token"


def test_identity_resolver_rejects_untrusted_lookup_key(monkeypatch, tmp_path):
    configure_db(monkeypatch, tmp_path)
    monkeypatch.setenv("WORKSPACE_IDENTITY_RESOLVE_KEY", "resolver-key")
    with pytest.raises(main.HTTPException) as error:
        main.resolve_workspace_identity(
            main.WorkspaceIdentityResolve(token="not-a-token", remote_addr=""),
            x_workspace_identity_key="wrong-key",
        )
    assert error.value.status_code == 403


def test_patchguard_notebook_uses_nested_workspace_path():
    row = {"host_port": 9012, "jupyter_token": "test-token"}
    assert "patchguard/patchguard_starter.ipynb" in main.WORKSPACE_NOTEBOOKS
    assert main.workspace_url(row, "patchguard/patchguard_starter.ipynb").endswith(
        ":9012/lab/tree/patchguard/patchguard_starter.ipynb?token=test-token"
    )


@pytest.mark.parametrize("notebook", [
    "granny/granny_starter.ipynb", "granny2/granny2_starter.ipynb",
    "granny_infinity/granny_infinity_starter.ipynb", "granny_starter.ipynb",
])
def test_retired_notebooks_cannot_be_launched(notebook):
    assert notebook not in main.WORKSPACE_NOTEBOOKS
    with pytest.raises(ValueError, match="Unknown challenge notebook"):
        main.workspace_url({}, notebook)


def test_workspace_migration_stages_current_navigation_and_checks_completion(monkeypatch, tmp_path):
    script = tmp_path / "migrate.py"
    script.write_text("# migration")
    (tmp_path / "start_here.ipynb").write_text('{"cells": []}')
    (tmp_path / "README.md").write_text("Current exercises")
    monkeypatch.setattr(main, "WORKSPACE_MIGRATION_SCRIPT", script)
    monkeypatch.setattr(main, "WORKSPACE_TEMPLATE", tmp_path)
    staged = {}

    class Container:
        def put_archive(self, path, data):
            assert path == "/tmp"
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                staged.update({member.name: archive.extractfile(member).read()
                               for member in archive.getmembers()})
            return True

        def exec_run(self, args):
            assert args[0] == "python"
            assert args[2:4] == ["--workspace", "/workspace"]
            assert args[4] == "--template"
            assert args[1] == args[5] + "/migrate.py"
            return SimpleNamespace(exit_code=0)

    main.migrate_workspace_content(Container())
    assert {name.split("/")[-1] for name in staged} == {"migrate.py", "start_here.ipynb", "README.md"}

    class BrokenContainer(Container):
        def exec_run(self, args):
            return SimpleNamespace(exit_code=1)

    with pytest.raises(RuntimeError, match="student files were not reset"):
        main.migrate_workspace_content(BrokenContainer())


def test_arena_token_is_persisted_and_legacy_workspace_uses_jupyter_token(monkeypatch, tmp_path):
    configure_db(monkeypatch, tmp_path)
    insert_workspace()
    issued = main.workspace_identity_token("u3")
    with main.db_connection() as conn:
        saved = conn.execute(
            "SELECT token, token_hash FROM workspace_identity_tokens WHERE workspace_id='u3'"
        ).fetchone()
    assert saved["token"] == issued
    assert saved["token_hash"] == hashlib.sha256(issued.encode()).hexdigest()
    with main.db_connection() as conn:
        conn.execute("UPDATE workspace_identity_tokens SET token=NULL WHERE workspace_id='u3'")
    legacy = SimpleNamespace(attrs={"Config": {"Env": ["JUPYTER_TOKEN=test-token"]}})
    assert main._arena_token_for_workspace(main.get_row_by_uid(3), legacy) == "test-token"
    current = SimpleNamespace(attrs={"Config": {"Env": ["ARENA_TOKEN=container-token"]}})
    assert main._arena_token_for_workspace(main.get_row_by_uid(3), current) == "container-token"


def test_launcher_arena_registration_is_best_effort_and_uses_launcher_identity(monkeypatch, tmp_path):
    configure_db(monkeypatch, tmp_path)
    insert_workspace()
    monkeypatch.setattr(main, "ARENA_URL", "http://arena-service:5000")
    monkeypatch.setattr(main, "ARENA_LAUNCHER_KEY", "launcher-test-key")
    sent = {}

    class Response:
        def raise_for_status(self):
            pass

    def post(url, **kwargs):
        sent["url"] = url
        sent.update(kwargs)
        return Response()

    monkeypatch.setattr(main.requests, "post", post)
    container = SimpleNamespace(attrs={"Config": {"Env": ["ARENA_TOKEN=opaque-workspace-token"]}})
    main._register_arena_workspace(main.get_row_by_uid(3), container)
    assert sent["url"] == "http://arena-service:5000/internal/participants/register"
    assert sent["json"] == {
        "ctfd_user_id": 3,
        "username": "student1",
        "workspace_id": "u3",
        "token": "opaque-workspace-token",
    }
    assert sent["headers"] == {"X-Arena-Launcher-Key": "launcher-test-key"}


def test_red_and_blue_notebooks_are_allowlisted_for_both_workspace_entry_points():
    for name in ("xray_red/red_team_fgsm.ipynb", "xray_blue/blue_team.ipynb"):
        assert name in main.WORKSPACE_NOTEBOOKS
        row = {"host_port": 9012, "jupyter_token": "test-token"}
        assert main.workspace_url(row, name).endswith(f":9012/lab/tree/{name}?token=test-token")


def test_port_allocator_uses_lowest_port_and_docker_bindings(monkeypatch, tmp_path):
    configure_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "PORT_START", 9001)
    monkeypatch.setattr(main, "PORT_END", 9003)
    class Containers:
        def list(self, all=True):
            return [SimpleNamespace(attrs={"NetworkSettings": {"Ports": {"8888/tcp": [{"HostPort": "9001"}]}}})]
    client = SimpleNamespace(containers=Containers())
    with main.db_connection() as conn:
        assert main.allocate_port(client, conn) == 9002
        conn.execute("INSERT INTO workspaces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     ("u4", 4, "student2", "ai-range-u4", 9002, "tok", "Stopped", "c", "u"))
        assert main.allocate_port(client, conn) == 9003


def test_docker_labels_identify_managed_workspace():
    labels = {**main.LABELS, "ctfd.user_id": "3", "ctfd.username": "student1", "ai.range.workspace_id": "u3"}
    assert labels == {
        "ai.range.managed": "true",
        "ai.range.role": "participant",
        "ctfd.user_id": "3",
        "ctfd.username": "student1",
        "ai.range.workspace_id": "u3",
    }


def test_workspace_status_reconciles_with_actual_container():
    class Container:
        labels = {"ai.range.managed": "true"}
        status = "running"
        def reload(self):
            pass
    class Containers:
        def get(self, name):
            return Container()
    row = {"container_name": "ai-range-u3"}
    assert main.current_status(SimpleNamespace(containers=Containers()), row) == "Running"
    Container.status = "exited"
    assert main.current_status(SimpleNamespace(containers=Containers()), row) == "Stopped"


@pytest.mark.parametrize("gpu_enabled", [False, True])
def test_repeated_launch_reuses_same_workspace_container(monkeypatch, tmp_path, gpu_enabled):
    configure_db(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "PORT_START", 9001)
    monkeypatch.setattr(main, "PORT_END", 9003)
    monkeypatch.setattr(main, "GPU_ENABLED", gpu_enabled)
    monkeypatch.setenv("TORCH_DEVICE", "cuda" if gpu_enabled else "auto")
    migrations = []
    monkeypatch.setattr(main, "migrate_workspace_content", lambda container: migrations.append(container.name))

    class FakeContainer:
        def __init__(self, name, labels):
            self.name, self.labels, self.status = name, labels, "running"
        def reload(self):
            pass
        def start(self):
            self.status = "running"

    class Containers:
        def __init__(self):
            self.items = {}
            self.created = 0
            self.last_environment = None
            self.last_options = None
        def list(self, all=True):
            return []
        def get(self, name):
            if name not in self.items:
                raise NotFound("missing")
            return self.items[name]
        def create(self, image, name, labels, **kwargs):
            self.created += 1
            self.last_environment = kwargs["environment"]
            self.last_options = kwargs
            self.items[name] = FakeContainer(name, labels)
            return self.items[name]

    class FakeClient:
        def __init__(self):
            self.containers = Containers()
            self.images = SimpleNamespace(get=lambda image: object())
            self.networks = SimpleNamespace(get=lambda network: object())
        def ping(self):
            return True
        def close(self):
            pass

    client = FakeClient()
    monkeypatch.setattr(main, "docker_client", lambda: client)
    monkeypatch.setattr(main, "wait_ready", lambda name, token: True)
    first, ready1 = main.launch_workspace("3", "student1")
    second, ready2 = main.launch_workspace("3", "student1")
    assert ready1 and ready2
    assert (first["workspace_id"], first["host_port"], first["jupyter_token"]) == (
        second["workspace_id"], second["host_port"], second["jupyter_token"]
    )
    assert client.containers.created == 1
    assert migrations == ["ai-range-u3"]
    assert "TARGET_URL" not in client.containers.last_environment
    assert client.containers.last_environment["CTFD_PUBLIC_URL"] == main.CTFD_PUBLIC_URL
    assert client.containers.last_environment["ARENA_TOKEN"]
    assert client.containers.last_environment["TORCH_DEVICE"] == ("cuda" if gpu_enabled else "auto")
    if gpu_enabled:
        assert client.containers.last_options["device_requests"][0]["Capabilities"] == [["gpu"]]
    else:
        assert "device_requests" not in client.containers.last_options
    with main.db_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM workspaces").fetchone()[0] == 1


@pytest.mark.parametrize(("ready", "expected_refresh"), [(False, True), (True, False)])
def test_workspace_page_refreshes_only_while_starting(
    monkeypatch, tmp_path, ready, expected_refresh
):
    configure_db(monkeypatch, tmp_path)
    insert_workspace()

    class Container:
        labels = {"ai.range.managed": "true"}
        status = "running"

        def reload(self):
            pass

    class Containers:
        def get(self, name):
            return Container()

    class Client:
        containers = Containers()

        def ping(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(main, "docker_client", Client)
    monkeypatch.setattr(main, "jupyter_is_ready", lambda *args, **kwargs: ready)
    monkeypatch.setattr(main, "CTFD_PUBLIC_URL", "http://ctfd.example")

    response = main.workspace(3, make_request("/workspace/3"))
    html = response.body.decode("utf-8")

    assert response.status_code == 200
    assert ('http-equiv="refresh" content="4"' in html) is expected_refresh
    assert "Back to CTFd" in html
    assert 'href="http://ctfd.example"' in html
    if expected_refresh:
        assert "refreshes automatically" in html
    else:
        assert "refreshes automatically" not in html


def test_instructor_workspace_link_redirects_to_ctfd_admin_page():
    response = main.instructor_workspaces()
    assert response.status_code == 303
    assert response.headers["location"].endswith("/admin/workspaces")


def _arena_test_png(red_value=90):
    """A valid 256x256 RGB PNG without requiring imaging test dependencies."""
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    row = b"\0" + bytes((red_value, 40, 120)) * 256
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 256, 256, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * 256))
            + chunk(b"IEND", b""))


def _arena_existing_notebook():
    return {
        "cells": [
            {"cell_type": "markdown", "id": "overview", "metadata": {},
             "source": ["# Blue notebook\n", "My own introduction."]},
            {"cell_type": "code", "id": "my-experiment", "metadata": {"tags": ["mine"]},
             "execution_count": 7, "outputs": [{"output_type": "stream", "name": "stdout",
                                               "text": ["keep this output\n"]}],
             "source": ["sigma = 0.031  # participant edit\n"]},
        ],
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


class _ArenaWorkspaceContainer:
    """Model Docker archive reads and writes against one participant workspace."""

    def __init__(self, notebook, token="token-for-this-participant"):
        self.attrs = {"Config": {"Env": [f"ARENA_TOKEN={token}"]}}
        self.files = {
            "/workspace/xray_blue/blue_team.ipynb": json.dumps(notebook).encode("utf-8")
        }
        self.put_calls = 0

    def get_archive(self, path):
        content = self.files[path]
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            entry = tarfile.TarInfo(path.rsplit("/", 1)[-1])
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
        return iter([buffer.getvalue()]), {"name": path, "size": len(content)}

    def put_archive(self, destination, packed):
        self.put_calls += 1
        content = b"".join(packed) if not isinstance(packed, bytes) else packed
        with tarfile.open(fileobj=io.BytesIO(content)) as archive:
            for entry in archive.getmembers():
                if not entry.isfile():
                    continue
                path = destination.rstrip("/") + "/" + entry.name.lstrip("/")
                assert "/../" not in path
                self.files[path] = archive.extractfile(entry).read()
        return True


def _arena_blue_handoff_fixture(monkeypatch, *, pending=True, png=None, corrupt=None):
    notebook = _arena_existing_notebook()
    container = _ArenaWorkspaceContainer(notebook)
    requests_seen = []
    attack_id = "a" * 32
    png = _arena_test_png() if png is None else png
    sha = hashlib.sha256(png).hexdigest()

    class Response:
        def __init__(self, data=None, content=b"", headers=None):
            self.data = data
            self.content = content
            self.headers = headers or {}
            self.status_code = 200

        def json(self):
            return self.data

        def raise_for_status(self):
            return None

    def fake_get(url, **kwargs):
        requests_seen.append((url, kwargs))
        assert kwargs["headers"]["Authorization"] == "Bearer token-for-this-participant"
        if url == f"{main.ARENA_URL}/api/blue/pending":
            if not pending:
                return Response({"pending": False})
            return Response({"pending": True, "attack_id": attack_id,
                             "sequence_number": 1, "source_id": "0",
                             "sha256": "0" * 64 if corrupt == "metadata" else sha,
                             "artifact_url": f"/api/attacks/{attack_id}/artifact"})
        if url == f"{main.ARENA_URL}/api/attacks/{attack_id}/artifact":
            return Response(content=png, headers={
                "X-Artifact-SHA256": "0" * 64 if corrupt == "header" else sha})
        raise AssertionError(f"Unexpected Arena request: {url}")

    client = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(main, "docker_client", lambda: client)
    monkeypatch.setattr(main, "managed_container", lambda _client, _row: container)
    monkeypatch.setattr(main.requests, "get", fake_get)
    return notebook, container, requests_seen, fake_get


def test_arena_blue_launch_stages_verified_image_and_preserves_notebook(monkeypatch):
    original, container, requests_seen, _ = _arena_blue_handoff_fixture(monkeypatch)
    row = {"workspace_id": "u42", "ctfd_user_id": 42}

    main.prepare_arena_blue_workspace(row)

    staged = container.files[f"/workspace/xray_blue/inbox/arena_{'a' * 32}.png"]
    assert staged == _arena_test_png()
    updated = json.loads(container.files["/workspace/xray_blue/blue_team.ipynb"])
    assert updated["cells"][0] == original["cells"][0]
    assert updated["cells"][2] == original["cells"][1]
    assert updated["metadata"] == original["metadata"]
    assert len(updated["cells"]) == 3
    preview = "".join(updated["cells"][1]["source"])
    assert updated["cells"][1]["cell_type"] == "markdown"
    assert "<!-- ARENA_BLUE_PREVIEW -->" in preview
    assert f"inbox/arena_{'a' * 32}.png" in preview
    assert any(markup in preview for markup in ("![", "<img"))
    assert [url for url, _ in requests_seen] == [
        f"{main.ARENA_URL}/api/blue/pending",
        f"{main.ARENA_URL}/api/attacks/{'a' * 32}/artifact",
    ]

    # Reopening updates the same preview cell; it never duplicates it or resets edits.
    main.prepare_arena_blue_workspace(row)
    reopened = json.loads(container.files["/workspace/xray_blue/blue_team.ipynb"])
    assert len(reopened["cells"]) == 3
    assert reopened["cells"][2] == original["cells"][1]
    assert sum("<!-- ARENA_BLUE_PREVIEW -->" in "".join(cell["source"])
               for cell in reopened["cells"]) == 1


def test_arena_blue_launch_without_pending_attack_replaces_stale_preview(monkeypatch):
    original, container, _, _ = _arena_blue_handoff_fixture(monkeypatch)
    row = {"workspace_id": "u42", "ctfd_user_id": 42}
    main.prepare_arena_blue_workspace(row)

    def no_pending(url, **kwargs):
        assert url == f"{main.ARENA_URL}/api/blue/pending"
        assert kwargs["headers"]["Authorization"] == "Bearer token-for-this-participant"
        return SimpleNamespace(json=lambda: {"pending": False}, raise_for_status=lambda: None)

    monkeypatch.setattr(main.requests, "get", no_pending)
    main.prepare_arena_blue_workspace(row)
    updated = json.loads(container.files["/workspace/xray_blue/blue_team.ipynb"])
    preview = "".join(updated["cells"][1]["source"])
    assert "<!-- ARENA_BLUE_PREVIEW -->" in preview
    assert "pending" in preview.lower() or "waiting" in preview.lower()
    assert "inbox/arena_" not in preview
    assert updated["cells"][2] == original["cells"][1]
    assert len(updated["cells"]) == 3


@pytest.mark.parametrize("corrupt", ["metadata", "header"])
def test_arena_blue_launch_rejects_artifact_digest_mismatch_without_writes(
    monkeypatch, corrupt
):
    original, container, _, _ = _arena_blue_handoff_fixture(monkeypatch, corrupt=corrupt)
    with pytest.raises((RuntimeError, ValueError), match="(?i)digest|sha|integrity"):
        main.prepare_arena_blue_workspace({"workspace_id": "u42", "ctfd_user_id": 42})
    assert container.put_calls == 0
    assert container.files == {
        "/workspace/xray_blue/blue_team.ipynb": json.dumps(original).encode("utf-8")
    }
