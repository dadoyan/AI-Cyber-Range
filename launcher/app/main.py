from __future__ import annotations

import html
import hashlib
import io
import json
import logging
import os
import re
import secrets
import sqlite3
import tarfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote

import docker
import requests
from docker.errors import APIError, DockerException, ImageNotFound, NotFound
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-range.launcher")
app = FastAPI(title="AI Cyber Range Lab Launcher")

DB_PATH = Path(os.getenv("LAUNCHER_DB_PATH", "/data/workspaces.sqlite3"))
IMAGE = os.getenv("PARTICIPANT_IMAGE", "ai-cyber-participant:local")
NETWORK = os.getenv("RANGE_DOCKER_NETWORK", "ai-cyber-range_default")
PORT_START = int(os.getenv("WORKSPACE_PORT_START", "9001"))
PORT_END = int(os.getenv("WORKSPACE_PORT_END", "9099"))
BIND_HOST = os.getenv("WORKSPACE_BIND_HOST", "127.0.0.1")
PUBLIC_BASE = os.getenv("WORKSPACE_PUBLIC_BASE_URL", "http://localhost").rstrip("/")
XRAY_REDBLUE_URL = os.getenv("XRAY_REDBLUE_URL", "http://xray-redblue:5000")
XRAY_BLUE_NOTEBOOK = "xray_blue/blue_starter.ipynb"
XRAY_BLUE_TEMPLATE = Path("/app/templates/xray_blue/blue_starter.ipynb")
ARENA_BLUE_NOTEBOOK = "xray_blue/blue_team.ipynb"
ARENA_BLUE_TEMPLATE = Path("/app/templates/xray_blue/blue_team.ipynb")
ARENA_BLUE_PREVIEW_MARKER = "<!-- ARENA_BLUE_PREVIEW -->"
ARENA_BLUE_MAX_NOTEBOOK_BYTES = 32 * 1024 * 1024
ARENA_BLUE_MAX_IMAGE_BYTES = 2 * 1024 * 1024
ARENA_URL = os.getenv("ARENA_URL", "http://arena-service:5000").rstrip("/")
ARENA_LAUNCHER_KEY = os.getenv("ARENA_LAUNCHER_KEY", "")
READY_TIMEOUT = max(1, int(os.getenv("WORKSPACE_READY_TIMEOUT", "30")))
MEMORY_LIMIT = os.getenv("WORKSPACE_MEMORY_LIMIT", "").strip()
CPU_LIMIT = os.getenv("WORKSPACE_CPU_LIMIT", "").strip()
GPU_ENABLED = os.getenv("WORKSPACE_GPU_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
WORKSPACE_LAUNCH_SECRET = os.getenv("WORKSPACE_LAUNCH_SECRET", "")
CTFD_PUBLIC_URL = os.getenv("CTFD_PUBLIC_URL", "http://localhost:8001").rstrip("/")
START_NOTEBOOK = "start_here.ipynb"
WORKSPACE_MIGRATION_SCRIPT = Path("/app/migrations/retire_workspace_exercises.py")
WORKSPACE_TEMPLATE = Path("/app/templates")
WORKSPACE_NOTEBOOKS = {
    START_NOTEBOOK,
    "patchguard/patchguard_starter.ipynb",
    "xray_red/xray_red_blue_starter.ipynb",
    XRAY_BLUE_NOTEBOOK,
    "llm_safety/llm_safety_starter.ipynb",
    "xray_red/red_team.ipynb",
    "xray_red/red_team_fgsm.ipynb",
    ARENA_BLUE_NOTEBOOK,
}

LABELS = {
    "ai.range.managed": "true",
    "ai.range.role": "participant",
}


class CTFdWorkspaceLaunch(BaseModel):
    user_id: int = Field(gt=0)
    username: str = Field(min_length=1, max_length=64)
    notebook: str | None = Field(default=None, max_length=128)


class CTFdWorkspaceDelete(BaseModel):
    user_id: int = Field(gt=0)


class WorkspaceIdentityResolve(BaseModel):
    token: str = Field(default="", max_length=256)
    remote_addr: str = Field(default="", max_length=64)


@contextmanager
def db_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA busy_timeout=15000")
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with db_connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS workspaces (
            workspace_id TEXT PRIMARY KEY,
            ctfd_user_id INTEGER NOT NULL UNIQUE,
            ctfd_username TEXT NOT NULL COLLATE NOCASE UNIQUE,
            container_name TEXT NOT NULL UNIQUE,
            host_port INTEGER NOT NULL UNIQUE,
            jupyter_token TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS workspace_identity_tokens (
            workspace_id TEXT PRIMARY KEY,
            token_hash TEXT NOT NULL UNIQUE,
            token TEXT,
            created_at TEXT NOT NULL
        )""")
        token_columns = {row["name"] for row in conn.execute("PRAGMA table_info(workspace_identity_tokens)")}
        if "token" not in token_columns:
            conn.execute("ALTER TABLE workspace_identity_tokens ADD COLUMN token TEXT")


init_db()


def docker_client():
    return docker.from_env(timeout=15)


def require_workspace_admin_key(value: str) -> None:
    if not WORKSPACE_LAUNCH_SECRET or not secrets.compare_digest(
        value, WORKSPACE_LAUNCH_SECRET
    ):
        raise HTTPException(status_code=403, detail="Invalid workspace administration request.")


def require_identity_resolve_key(value: str) -> None:
    expected = os.getenv("WORKSPACE_IDENTITY_RESOLVE_KEY", WORKSPACE_LAUNCH_SECRET)
    if not expected or not secrets.compare_digest(value, expected):
        raise HTTPException(status_code=403, detail="Invalid identity lookup request.")


def workspace_identity_token(workspace_id: str) -> str:
    """Issue and retain an opaque token so Launcher can restore it after recreation."""
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with db_connection() as conn:
        conn.execute(
            """INSERT INTO workspace_identity_tokens (workspace_id, token_hash, token, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(workspace_id) DO UPDATE SET
                   token_hash=excluded.token_hash, token=excluded.token, created_at=excluded.created_at""",
            (workspace_id, token_hash, token, now()),
        )
    return token


def _container_environment_value(container, name: str) -> str:
    try:
        values = (container.attrs.get("Config") or {}).get("Env") or []
        prefix = f"{name}="
        return next((value[len(prefix):] for value in values if value.startswith(prefix)), "")
    except (AttributeError, TypeError):
        return ""


def _arena_token_for_workspace(row, container) -> str:
    # Existing workspaces predate ARENA_TOKEN. Their per-workspace Jupyter token
    # remains a random secret and is a compatible bearer token for this PoC.
    configured = _container_environment_value(container, "ARENA_TOKEN")
    if configured:
        return configured
    with db_connection() as conn:
        saved = conn.execute(
            "SELECT token FROM workspace_identity_tokens WHERE workspace_id=?",
            (row["workspace_id"],),
        ).fetchone()
    return str(saved["token"]) if saved and saved["token"] else str(row["jupyter_token"])


def _register_arena_workspace(row, container) -> None:
    if not ARENA_LAUNCHER_KEY:
        return
    token = _arena_token_for_workspace(row, container)
    try:
        response = requests.post(
            f"{ARENA_URL}/internal/participants/register",
            headers={"X-Arena-Launcher-Key": ARENA_LAUNCHER_KEY},
            json={
                "ctfd_user_id": int(row["ctfd_user_id"]),
                "username": str(row["ctfd_username"]),
                "workspace_id": str(row["workspace_id"]),
                "token": token,
            },
            timeout=3,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        # Arena is optional to the regular workspace lifecycle.
        logger.info("Arena participant registration is currently unavailable for workspace %s: %s", row["workspace_id"], exc)


def _unregister_arena_workspace(row) -> None:
    if not ARENA_LAUNCHER_KEY:
        return
    try:
        response = requests.post(
            f"{ARENA_URL}/internal/participants/unregister",
            headers={"X-Arena-Launcher-Key": ARENA_LAUNCHER_KEY},
            json={"ctfd_user_id": int(row["ctfd_user_id"]), "workspace_id": str(row["workspace_id"])},
            timeout=3,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.info("Arena workspace unregistration is currently unavailable for workspace %s: %s", row["workspace_id"], exc)


def _identity_payload(row, source: str) -> dict[str, Any]:
    return {
        "ctfd_user_id": int(row["ctfd_user_id"]),
        "ctfd_username": str(row["ctfd_username"]),
        "workspace_id": str(row["workspace_id"]),
        "identity_source": source,
    }


def _workspace_identity_by_peer_ip(remote_addr: str):
    """Map a legacy workspace request by its Docker peer address, never by body claims."""
    if not remote_addr or remote_addr in {"127.0.0.1", "::1", "localhost"}:
        return None
    client = None
    try:
        client = docker_client()
        for container in client.containers.list(all=True):
            labels = container.labels or {}
            if labels.get("ai.range.managed") != "true":
                continue
            networks = (container.attrs.get("NetworkSettings") or {}).get("Networks") or {}
            addresses = {
                str(details.get("IPAddress", ""))
                for details in networks.values()
                if isinstance(details, dict)
            }
            if remote_addr not in addresses:
                continue
            workspace_id = labels.get("ai.range.workspace_id", "")
            with db_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM workspaces WHERE workspace_id=?", (workspace_id,)
                ).fetchone()
            return row
    except DockerException:
        logger.info("Could not resolve a legacy workspace identity from its Docker peer address")
    finally:
        if client:
            client.close()
    return None


def validate_identity(user_id: str, username: str) -> tuple[int, str]:
    user_id = user_id.strip()
    username = username.strip()
    if not user_id.isdecimal() or int(user_id) <= 0:
        raise ValueError("CTFd User ID must be a positive whole number.")
    if not username or len(username) > 64 or any(ord(c) < 32 for c in username):
        raise ValueError("Enter a CTFd username of 1 to 64 printable characters.")
    return int(user_id), username


def identity_mapping(conn: sqlite3.Connection, user_id: int, username: str):
    by_id = conn.execute("SELECT * FROM workspaces WHERE ctfd_user_id=?", (user_id,)).fetchone()
    by_name = conn.execute("SELECT * FROM workspaces WHERE ctfd_username=?", (username,)).fetchone()
    if by_id and by_id["ctfd_username"].casefold() != username.casefold():
        raise ValueError(f"CTFd user ID {user_id} is already mapped to username {by_id['ctfd_username']!r}.")
    if by_name and int(by_name["ctfd_user_id"]) != user_id:
        raise ValueError(f"Username {username!r} is already mapped to CTFd user ID {by_name['ctfd_user_id']}.")
    return by_id


def workspace_url(row: Any, notebook: str | None = START_NOTEBOOK) -> str:
    path = "/lab"
    if notebook is not None:
        if notebook not in WORKSPACE_NOTEBOOKS:
            raise ValueError("Unknown challenge notebook")
        path = f"/lab/tree/{quote(notebook, safe='/')}"
    return f"{PUBLIC_BASE}:{row['host_port']}{path}?token={quote(row['jupyter_token'])}"


def managed_container(client, row):
    try:
        container = client.containers.get(row["container_name"])
    except NotFound:
        return None
    if container.labels.get("ai.range.managed") != "true":
        raise RuntimeError(f"Container name collision: {row['container_name']} is not managed by the Lab Launcher.")
    return container


def docker_reserved_ports(client) -> set[int]:
    used = set()
    for container in client.containers.list(all=True):
        try:
            bindings = container.attrs["NetworkSettings"]["Ports"] or {}
        except (KeyError, TypeError):
            continue
        for entries in bindings.values():
            for entry in entries or []:
                with_port = entry.get("HostPort")
                if with_port and with_port.isdecimal():
                    used.add(int(with_port))
    return used


def allocate_port(client, conn: sqlite3.Connection, exclude: int | None = None) -> int:
    reserved = {int(r[0]) for r in conn.execute("SELECT host_port FROM workspaces") if int(r[0]) != exclude}
    reserved.update(docker_reserved_ports(client))
    for port in range(PORT_START, PORT_END + 1):
        if port not in reserved:
            return port
    raise RuntimeError(f"No workspace ports are available in configured range {PORT_START}-{PORT_END}.")


def current_status(client, row) -> str:
    container = managed_container(client, row)
    if container is None:
        return "Missing"
    container.reload()
    if container.status == "running":
        return "Running"
    if container.status in {"created", "restarting"}:
        return "Starting"
    return "Stopped"


def refresh_row(row, client, *, probe_ready: bool = False):
    container_status = current_status(client, row)
    if container_status != "Running":
        status = container_status
    elif probe_ready:
        status = "Ready" if jupyter_is_ready(row["container_name"], row["jupyter_token"]) else "Starting"
    elif row["status"] in {"Ready", "Starting"}:
        status = row["status"]
    else:
        # Older workspace records used "Running" before readiness was tracked.
        status = "Starting"
    with db_connection() as conn:
        conn.execute("UPDATE workspaces SET status=?, updated_at=? WHERE workspace_id=?",
                     (status, now(), row["workspace_id"]))
    result = dict(row)
    result["status"] = status
    result["url"] = workspace_url(row)
    return result


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def jupyter_is_ready(container_name: str, token: str, timeout: float = 1.0) -> bool:
    url = f"http://{container_name}:8888/lab?token={quote(token)}"
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=False)
        return response.status_code in (200, 302)
    except requests.RequestException:
        return False


def wait_ready(container_name: str, token: str) -> bool:
    deadline = time.monotonic() + READY_TIMEOUT
    while time.monotonic() < deadline:
        if jupyter_is_ready(container_name, token, timeout=2):
            return True
        time.sleep(0.5)
    return False


def create_or_start(client, row, *, fresh: bool = False, wait_until_ready: bool = True):
    try:
        client.images.get(IMAGE)
    except ImageNotFound as exc:
        raise RuntimeError(f"Participant image {IMAGE!r} is missing. Run docker compose up --build -d first.") from exc
    try:
        client.networks.get(NETWORK)
    except NotFound as exc:
        raise RuntimeError(f"Configured Docker network {NETWORK!r} was not found. Start the cyber range with Docker Compose.") from exc

    container = managed_container(client, row)
    existing = container is not None
    if container is None:
        labels = {
            **LABELS,
            "ctfd.user_id": str(row["ctfd_user_id"]),
            "ctfd.username": row["ctfd_username"],
            "ai.range.workspace_id": row["workspace_id"],
        }
        arena_token = workspace_identity_token(row["workspace_id"])
        env = {
            "CTFD_USER_ID": str(row["ctfd_user_id"]),
            "CTFD_USERNAME": row["ctfd_username"],
            "ARENA_TOKEN": arena_token,
            "WORKSPACE_ID": row["workspace_id"],
            "JUPYTER_TOKEN": row["jupyter_token"],
            "PATCHGUARD_TARGET_URL": os.getenv("PATCHGUARD_TARGET_URL", "http://patchguard-target:8000"),
            "XRAY_REDBLUE_URL": XRAY_REDBLUE_URL,
            "ARENA_URL": ARENA_URL,
            "LLM_SAFETY_URL": os.getenv("LLM_SAFETY_URL", "http://llm-safety-target:8000"),
            "CTFD_PUBLIC_URL": CTFD_PUBLIC_URL,
            "TORCH_DEVICE": os.getenv("TORCH_DEVICE", "auto"),
        }
        options = {}
        if MEMORY_LIMIT:
            options["mem_limit"] = MEMORY_LIMIT
        if CPU_LIMIT:
            try:
                options["nano_cpus"] = int(float(CPU_LIMIT) * 1_000_000_000)
            except ValueError as exc:
                raise RuntimeError("WORKSPACE_CPU_LIMIT must be a number of CPU cores.") from exc
        if GPU_ENABLED:
            options["device_requests"] = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]
        container = client.containers.create(
            IMAGE,
            name=row["container_name"],
            detach=True,
            environment=env,
            labels=labels,
            ports={"8888/tcp": (BIND_HOST, int(row["host_port"]))},
            network=NETWORK,
            restart_policy={"Name": "unless-stopped"},
            **options,
        )
    container.reload()
    if container.status != "running":
        container.start()
    if existing:
        migrate_workspace_content(container)
    with db_connection() as conn:
        conn.execute("UPDATE workspaces SET status='Starting', updated_at=? WHERE workspace_id=?",
                     (now(), row["workspace_id"]))
    ready = (
        wait_ready(row["container_name"], row["jupyter_token"])
        if wait_until_ready
        else jupyter_is_ready(row["container_name"], row["jupyter_token"], timeout=0.25)
    )
    status = "Ready" if ready else "Starting"
    with db_connection() as conn:
        conn.execute("UPDATE workspaces SET status=?, updated_at=? WHERE workspace_id=?",
                     (status, now(), row["workspace_id"]))
    container.reload()
    _register_arena_workspace(row, container)
    return status, ready


def migrate_workspace_content(container) -> None:
    """Retire obsolete exercises without recreating a student's workspace."""
    staging_name = f"range-content-migration-{secrets.token_hex(12)}"
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        for source, name in (
            (WORKSPACE_MIGRATION_SCRIPT, "migrate.py"),
            (WORKSPACE_TEMPLATE / START_NOTEBOOK, START_NOTEBOOK),
            (WORKSPACE_TEMPLATE / "README.md", "README.md"),
        ):
            data = source.read_bytes()
            info = tarfile.TarInfo(f"{staging_name}/{name}")
            info.size = len(data)
            info.mode = 0o600
            archive.addfile(info, io.BytesIO(data))
    if not container.put_archive("/tmp", archive_bytes.getvalue()):
        raise RuntimeError("Could not stage the workspace content migration.")
    staging_path = f"/tmp/{staging_name}"
    result = container.exec_run([
        "python", f"{staging_path}/migrate.py", "--workspace", "/workspace",
        "--template", staging_path,
    ])
    if result.exit_code != 0:
        raise RuntimeError("Could not retire obsolete workspace content; student files were not reset.")


def launch_workspace(user_id: str, username: str, *, wait_until_ready: bool = True):
    uid, uname = validate_identity(user_id, username)
    client = docker_client()
    try:
        client.ping()
        with db_connection() as conn:
            existing = identity_mapping(conn, uid, uname)
            if existing:
                row = existing
            else:
                port = allocate_port(client, conn)
                workspace_id = f"u{uid}"
                row = {
                    "workspace_id": workspace_id,
                    "ctfd_user_id": uid,
                    "ctfd_username": uname,
                    "container_name": f"ai-range-u{uid}",
                    "host_port": port,
                    "jupyter_token": secrets.token_urlsafe(32),
                    "status": "Starting",
                    "created_at": now(),
                    "updated_at": now(),
                }
                conn.execute("INSERT INTO workspaces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(row.values()))
        try:
            status, ready = create_or_start(client, row, wait_until_ready=wait_until_ready)
        except Exception:
            # A newly allocated mapping should be retryable after failed creation.
            if not existing:
                try:
                    partial = client.containers.get(row["container_name"])
                    if partial.labels.get("ai.range.managed") == "true":
                        partial.remove(force=True)
                except NotFound:
                    pass
                except DockerException:
                    logger.exception("Could not clean up partially created workspace %s", row["workspace_id"])
                with db_connection() as conn:
                    conn.execute("DELETE FROM workspaces WHERE workspace_id=?", (row["workspace_id"],))
            raise
        with db_connection() as conn:
            row = conn.execute("SELECT * FROM workspaces WHERE workspace_id=?", (row["workspace_id"],)).fetchone()
        return refresh_row(row, client), ready
    except DockerException as exc:
        raise RuntimeError("Docker is unavailable. Check that Docker Desktop is running.") from exc
    finally:
        client.close()


def xray_blue_archive(round_id: str, manifest: dict[str, str], artifact: bytes,
                      notebook: bytes | None) -> bytes:
    """Build a small, path-bounded archive for one student's Blue folder."""
    if not re.fullmatch(r"[0-9a-f]{32}", round_id):
        raise ValueError("Invalid X-Ray round ID")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for directory in ("xray_blue", "xray_blue/rounds", f"xray_blue/rounds/{round_id}"):
            entry = tarfile.TarInfo(directory + "/")
            entry.type = tarfile.DIRTYPE
            entry.mode = 0o755
            archive.addfile(entry)

        def add_file(path: str, content: bytes) -> None:
            entry = tarfile.TarInfo(path)
            entry.size = len(content)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(content))

        add_file(f"xray_blue/rounds/{round_id}/successful_red_examples.npz", artifact)
        add_file("xray_blue/active_round.json", json.dumps(manifest, sort_keys=True).encode("utf-8"))
        if notebook is not None:
            add_file(XRAY_BLUE_NOTEBOOK, notebook)
    return buffer.getvalue()


def prepare_xray_blue_workspace(row: Any) -> None:
    """Copy the current accepted Red examples into an isolated student workspace."""
    base = XRAY_REDBLUE_URL.rstrip("/")
    try:
        round_response = requests.get(f"{base}/api/round", timeout=10)
        round_response.raise_for_status()
        round_state = round_response.json()
        latest_response = requests.get(f"{base}/api/round/latest", timeout=10)
        latest_response.raise_for_status()
        latest = latest_response.json()
        round_id = str(round_state.get("round_id", ""))
        red_run_id = str((latest.get("result") or {}).get("run_id", ""))
        if (not re.fullmatch(r"[0-9a-f]{32}", round_id)
                or not re.fullmatch(r"[0-9a-f]{32}", red_run_id)
                or latest.get("phase") not in {"blue", "complete"}
                or (latest.get("result") or {}).get("success") is not True
                or (round_state.get("red_run_id") != red_run_id)):
            raise RuntimeError("No active X-Ray Red artifact is ready for Blue")
        artifact_response = requests.get(
            f"{base}/api/round/latest/successes",
            params={"run_id": red_run_id}, timeout=60,
        )
        artifact_response.raise_for_status()
        artifact = artifact_response.content
        if not artifact.startswith(b"PK\x03\x04") or len(artifact) > 64 * 1024 * 1024:
            raise RuntimeError("The X-Ray Red example package has an invalid size")
        final_response = requests.get(f"{base}/api/round", timeout=10)
        final_response.raise_for_status()
        final_state = final_response.json()
        if (final_state.get("round_id") != round_id
                or final_state.get("red_run_id") != red_run_id
                or final_state.get("phase") not in {"blue", "complete"}):
            raise RuntimeError("The X-Ray round changed while preparing Blue")
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError("Could not fetch the active X-Ray Red examples") from exc

    manifest = {
        "round_id": round_id,
        "red_run_id": red_run_id,
        "artifact": f"rounds/{round_id}/successful_red_examples.npz",
        "sha256": hashlib.sha256(artifact).hexdigest(),
    }
    client = docker_client()
    try:
        container = managed_container(client, row)
        if container is None:
            raise RuntimeError("The participant workspace container is missing")
        notebook_exists = container.exec_run(["test", "-f", "/workspace/" + XRAY_BLUE_NOTEBOOK]).exit_code == 0
        notebook = None if notebook_exists else XRAY_BLUE_TEMPLATE.read_bytes()
        packed = xray_blue_archive(round_id, manifest, artifact, notebook)
        if not container.put_archive("/workspace", packed):
            raise RuntimeError("Could not place X-Ray Blue files in the participant workspace")
    except DockerException as exc:
        raise RuntimeError("Could not prepare the X-Ray Blue workspace") from exc
    finally:
        client.close()


def _arena_blue_preview_source(attack_id: str | None, sequence: int | None) -> list[str]:
    if attack_id is None:
        message = (
            "No Red attack is pending for this match. Open this notebook from the "
            "X-Ray Blue challenge page after Red submits a validated image."
        )
    else:
        message = (
            f"Validated Red attack {sequence} is ready. The image below was fetched "
            "for this workspace and checked against the Arena SHA-256 record.\n\n"
            f"![Accepted Red adversarial X-ray](inbox/arena_{attack_id}.png)\n\n"
            "Run section 1 to load the clean source beside it and continue the defense."
        )
    return [
        f"{ARENA_BLUE_PREVIEW_MARKER}\n",
        "## Red attack preview\n",
        "\n",
        message,
    ]


def _arena_blue_notebook_with_preview(notebook: bytes, source: list[str]) -> bytes | None:
    """Update the single managed Markdown cell without changing other notebook cells."""
    try:
        document = json.loads(notebook)
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError("The participant's Blue notebook is not valid JSON") from exc
    cells = document.get("cells") if isinstance(document, dict) else None
    if not isinstance(cells, list):
        raise RuntimeError("The participant's Blue notebook has no cells")
    previews = [
        cell for cell in cells
        if isinstance(cell, dict)
        and cell.get("cell_type") == "markdown"
        and ARENA_BLUE_PREVIEW_MARKER in "".join(cell.get("source", []))
    ]
    if len(previews) > 1:
        raise RuntimeError("The participant's Blue notebook has duplicate preview cells")
    if previews:
        if previews[0].get("source") == source:
            return None
        previews[0]["source"] = source
    else:
        cells.insert(min(1, len(cells)), {
            "cell_type": "markdown",
            "id": "blue-attack-preview",
            "metadata": {},
            "source": source,
        })
    return (json.dumps(document, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def _arena_blue_existing_notebook(container) -> bytes:
    path = f"/workspace/{ARENA_BLUE_NOTEBOOK}"
    try:
        stream, stat = container.get_archive(path)
    except NotFound:
        return ARENA_BLUE_TEMPLATE.read_bytes()
    size = int(stat.get("size", 0))
    if size < 0 or size > ARENA_BLUE_MAX_NOTEBOOK_BYTES:
        raise RuntimeError("The participant's Blue notebook exceeds the supported size")
    packed = b"".join(stream)
    if len(packed) > ARENA_BLUE_MAX_NOTEBOOK_BYTES + 1024 * 1024:
        raise RuntimeError("The participant's Blue notebook archive exceeds the supported size")
    try:
        with tarfile.open(fileobj=io.BytesIO(packed), mode="r:*") as archive:
            files = [member for member in archive.getmembers() if member.isfile()]
            if len(files) != 1 or files[0].size > ARENA_BLUE_MAX_NOTEBOOK_BYTES:
                raise RuntimeError("Could not read the participant's Blue notebook")
            extracted = archive.extractfile(files[0])
            if extracted is None:
                raise RuntimeError("Could not read the participant's Blue notebook")
            return extracted.read()
    except (tarfile.TarError, OSError) as exc:
        raise RuntimeError("Could not read the participant's Blue notebook") from exc


def _arena_blue_archive(notebook: bytes | None, image_path: str | None,
                        image: bytes | None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for directory in ("xray_blue", "xray_blue/inbox"):
            entry = tarfile.TarInfo(directory + "/")
            entry.type = tarfile.DIRTYPE
            entry.mode = 0o755
            entry.mtime = int(time.time())
            archive.addfile(entry)

        def add_file(path: str, data: bytes) -> None:
            entry = tarfile.TarInfo(path)
            entry.size = len(data)
            entry.mode = 0o644
            entry.mtime = int(time.time())
            archive.addfile(entry, io.BytesIO(data))

        if image_path is not None and image is not None:
            add_file(image_path, image)
        if notebook is not None:
            add_file(ARENA_BLUE_NOTEBOOK, notebook)
    return buffer.getvalue()


def prepare_arena_blue_workspace(row: Any) -> None:
    """Stage the authenticated pending Red image before opening the Arena Blue notebook."""
    client = docker_client()
    try:
        container = managed_container(client, row)
        if container is None:
            raise RuntimeError("The participant workspace container is missing")
        token = _arena_token_for_workspace(row, container)
        headers = {"Authorization": f"Bearer {token}"}
        try:
            pending_response = requests.get(
                f"{ARENA_URL}/api/blue/pending", headers=headers, timeout=10,
            )
            pending_response.raise_for_status()
            pending = pending_response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError("Could not load the participant's Arena Blue inbox") from exc
        if not isinstance(pending, dict) or type(pending.get("pending")) is not bool:
            raise RuntimeError("The Arena Blue inbox returned an invalid response")

        attack_id = None
        sequence = None
        image = None
        image_path = None
        if pending["pending"]:
            attack_id = pending.get("attack_id")
            sequence = pending.get("sequence_number")
            expected_sha = pending.get("sha256")
            if (not isinstance(attack_id, str)
                    or not re.fullmatch(r"[0-9a-f]{32}", attack_id)
                    or type(sequence) is not int or sequence < 1
                    or not isinstance(expected_sha, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
                    or pending.get("artifact_url") != f"/api/attacks/{attack_id}/artifact"):
                raise RuntimeError("The Arena Blue inbox returned invalid artifact metadata")
            try:
                artifact_response = requests.get(
                    f"{ARENA_URL}/api/attacks/{attack_id}/artifact",
                    headers=headers, timeout=15,
                )
                artifact_response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError("Could not fetch the participant's Red image") from exc
            image = artifact_response.content
            digest = hashlib.sha256(image).hexdigest()
            if (not image.startswith(b"\x89PNG\r\n\x1a\n")
                    or not 8 < len(image) <= ARENA_BLUE_MAX_IMAGE_BYTES
                    or digest != expected_sha
                    or artifact_response.headers.get("X-Artifact-SHA256") != digest):
                raise RuntimeError("The Arena Red image failed integrity checks")
            image_path = f"xray_blue/inbox/arena_{attack_id}.png"

        original = _arena_blue_existing_notebook(container)
        revised = _arena_blue_notebook_with_preview(
            original, _arena_blue_preview_source(attack_id, sequence),
        )
        if revised is not None or image is not None:
            packed = _arena_blue_archive(revised, image_path, image)
            if not container.put_archive("/workspace", packed):
                raise RuntimeError("Could not prepare the Arena Blue notebook and image")
    except DockerException as exc:
        raise RuntimeError("Could not prepare the Arena Blue workspace") from exc
    finally:
        client.close()


@app.post("/internal/workspace-launch", include_in_schema=False)
def launch_workspace_from_ctfd(
    payload: CTFdWorkspaceLaunch,
    x_workspace_launch_key: str = Header(default=""),
):
    require_workspace_admin_key(x_workspace_launch_key)
    if payload.notebook is not None and payload.notebook not in WORKSPACE_NOTEBOOKS:
        raise HTTPException(status_code=400, detail="Unknown challenge notebook.")
    try:
        row, ready = launch_workspace(
            str(payload.user_id), payload.username, wait_until_ready=False
        )
        if payload.notebook == XRAY_BLUE_NOTEBOOK:
            prepare_xray_blue_workspace(row)
        elif payload.notebook == ARENA_BLUE_NOTEBOOK:
            prepare_arena_blue_workspace(row)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("CTFd workspace launch failed for user ID %s", payload.user_id)
        raise HTTPException(status_code=503, detail="Workspace launch failed.") from exc
    return {"user_id": payload.user_id, "ready": ready, "status": row["status"]}


@app.post("/internal/identity/resolve", include_in_schema=False)
def resolve_workspace_identity(
    payload: WorkspaceIdentityResolve,
    x_workspace_identity_key: str = Header(default=""),
):
    """Resolve an opaque workspace token to its Launcher-owned CTFd mapping."""
    require_identity_resolve_key(x_workspace_identity_key)
    if payload.token:
        with db_connection() as conn:
            row = conn.execute(
                """SELECT w.* FROM workspace_identity_tokens AS t
                   JOIN workspaces AS w ON w.workspace_id=t.workspace_id
                   WHERE t.token_hash=?""",
                (hashlib.sha256(payload.token.encode("utf-8")).hexdigest(),),
            ).fetchone()
            if row:
                return _identity_payload(row, "arena_token")
            # Existing workspaces already have a launcher-generated random
            # Jupyter token stored alongside their CTFd identity mapping.
            row = conn.execute(
                "SELECT * FROM workspaces WHERE jupyter_token=?", (payload.token,)
            ).fetchone()
            if row:
                return _identity_payload(row, "jupyter_token")

    row = _workspace_identity_by_peer_ip(payload.remote_addr)
    if row:
        return _identity_payload(row, "docker_peer")
    raise HTTPException(status_code=404, detail="No managed workspace matches this identity.")


def workspace_inventory() -> list[dict[str, Any]]:
    with db_connection() as conn:
        rows = conn.execute("SELECT * FROM workspaces ORDER BY ctfd_user_id").fetchall()
    output = []
    client = None
    try:
        client = docker_client()
        client.ping()
        output = [refresh_row(row, client) for row in rows]
    except (DockerException, RuntimeError):
        logger.exception("Could not inspect managed participant workspaces")
        output = [{**dict(row), "status": "Unavailable"} for row in rows]
    finally:
        if client:
            client.close()
    return [
        {
            "ctfd_user_id": int(row["ctfd_user_id"]),
            "ctfd_username": row["ctfd_username"],
            "workspace_id": row["workspace_id"],
            "container_name": row["container_name"],
            "host_port": int(row["host_port"]),
            "status": row["status"],
            "created_at": row["created_at"],
        }
        for row in output
    ]


@app.get("/internal/workspaces", include_in_schema=False)
def list_workspaces_for_ctfd(x_workspace_launch_key: str = Header(default="")):
    require_workspace_admin_key(x_workspace_launch_key)
    return {"workspaces": workspace_inventory()}


@app.post("/internal/workspaces/delete", include_in_schema=False)
def delete_workspace_from_ctfd(
    payload: CTFdWorkspaceDelete,
    x_workspace_launch_key: str = Header(default=""),
):
    require_workspace_admin_key(x_workspace_launch_key)
    try:
        return delete_workspace(payload.user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("CTFd workspace deletion failed for user ID %s", payload.user_id)
        raise HTTPException(status_code=503, detail="Workspace deletion failed.") from exc


@app.post("/internal/workspaces/stop", include_in_schema=False)
def stop_workspace_from_ctfd(
    payload: CTFdWorkspaceDelete,
    x_workspace_launch_key: str = Header(default=""),
):
    require_workspace_admin_key(x_workspace_launch_key)
    try:
        row, _ = perform_action(payload.user_id, "stop")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("CTFd workspace stop failed for user ID %s", payload.user_id)
        raise HTTPException(status_code=503, detail="Workspace stop failed.") from exc
    return {
        "user_id": payload.user_id,
        "username": row["ctfd_username"],
        "status": row["status"],
        "stopped": True,
    }


def get_row_by_uid(user_id: int):
    with db_connection() as conn:
        return conn.execute("SELECT * FROM workspaces WHERE ctfd_user_id=?", (user_id,)).fetchone()


def delete_workspace(user_id: int) -> dict[str, Any]:
    row = get_row_by_uid(user_id)
    if not row:
        raise ValueError("No workspace is mapped to this CTFd user ID.")

    client = docker_client()
    try:
        client.ping()
        container = managed_container(client, row)
        if container:
            # Workspaces have no mounted user-data volumes; removing the container
            # removes its notebook files and writable layer as well.
            container.remove(force=True)
    except DockerException as exc:
        raise RuntimeError("Docker could not remove the workspace container.") from exc
    finally:
        client.close()

    _unregister_arena_workspace(row)
    with db_connection() as conn:
        conn.execute(
            "DELETE FROM workspace_identity_tokens WHERE workspace_id=?",
            (row["workspace_id"],),
        )
        conn.execute("DELETE FROM workspaces WHERE ctfd_user_id=?", (user_id,))
    return {
        "user_id": int(row["ctfd_user_id"]),
        "username": row["ctfd_username"],
        "workspace_id": row["workspace_id"],
        "container_name": row["container_name"],
        "deleted": True,
    }


def perform_action(user_id: int, action: str):
    row = get_row_by_uid(user_id)
    if not row:
        raise ValueError("No workspace is mapped to that CTFd user ID.")
    client = docker_client()
    try:
        client.ping()
        container = managed_container(client, row)
        if action == "reset":
            if container:
                container.remove(force=True)
            status, ready = create_or_start(client, row, fresh=True)
        elif action == "stop":
            if container:
                container.stop(timeout=10)
            status, ready = "Stopped", False
            with db_connection() as conn:
                conn.execute("UPDATE workspaces SET status=?, updated_at=? WHERE workspace_id=?",
                             (status, now(), row["workspace_id"]))
        elif action == "start":
            status, ready = create_or_start(client, row)
        else:
            raise ValueError("Unknown workspace action.")
        fresh = get_row_by_uid(user_id)
        return refresh_row(fresh, client), ready
    except DockerException as exc:
        raise RuntimeError("Docker is unavailable. Check that Docker Desktop is running.") from exc
    finally:
        client.close()


def page(
    title: str,
    body: str,
    error: str = "",
    *,
    refresh_seconds: int | None = None,
) -> HTMLResponse:
    error_html = f'<p class="error">{html.escape(error)}</p>' if error else ""
    refresh_meta = (
        f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">'
        if refresh_seconds and refresh_seconds > 0
        else ""
    )
    return HTMLResponse(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh_meta}<title>{html.escape(title)}</title><style>
body{{font:16px system-ui,sans-serif;max-width:900px;margin:3rem auto;padding:0 1rem;color:#162235;background:#f4f7fb}}main{{background:white;padding:2rem;border-radius:14px;box-shadow:0 6px 28px #182b4514}}h1{{margin-top:0}}label{{display:block;margin:1rem 0 .35rem}}input{{padding:.7rem;border:1px solid #aab7c7;border-radius:6px;width:min(100%,360px)}}button,.button{{background:#1456a0;color:white;border:0;border-radius:6px;padding:.7rem 1rem;text-decoration:none;display:inline-block;cursor:pointer;margin:.25rem}}button:disabled{{cursor:not-allowed;opacity:.85}}.secondary{{background:#52647a}}.inactive{{background:#7a8796}}.stop-running{{background:#f7d9d9;color:#842b2b;border:1px solid #e8bcbc}}.status{{display:inline-block;padding:.25rem .65rem;border-radius:999px;font-weight:650}}.status-ready{{background:#e1f5e8;color:#17633a}}.status-starting{{background:#fff1cf;color:#755200}}.status-stopped{{background:#e5e9ef;color:#465465}}.status-error{{background:#fee9e7;color:#9c2d23}}.error{{background:#fee9e7;color:#9c2d23;padding:.75rem;border-radius:6px}}.notice{{background:#edf5ff;padding:.75rem;border-radius:6px}}.warning{{background:#fff2d9;color:#704c00;border-left:4px solid #e1a52a;padding:.9rem;border-radius:6px}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:.6rem;border-bottom:1px solid #dde5ee}}.row-ready{{background:#f2faf4}}.row-starting{{background:#fffbef}}.row-stopped{{background:#f5f6f8}}.row-error{{background:#fff5f4}}code{{overflow-wrap:anywhere}}.actions{{margin-top:1.5rem;display:flex;align-items:center;flex-wrap:wrap;gap:.35rem}}.actions form{{display:inline}}.reset-area{{margin-top:1.1rem;padding-top:.8rem;border-top:1px solid #dde5ee}}
</style></head><body><main><h1>{html.escape(title)}</h1>{error_html}{body}<p><a href="/">Launcher home</a></p></main></body></html>""")


async def form_values(request: Request):
    data = parse_qs((await request.body()).decode("utf-8", errors="replace"))
    return {key: values[0] if values else "" for key, values in data.items()}


@app.get("/", response_class=HTMLResponse)
def home():
    return page("AI Cyber Range", """<p>Students: sign in to CTFd and choose <b>Launch Workspace</b> in its navigation. Your workspace will start or reopen automatically.</p>
<details><summary>Instructor or troubleshooting: manual launch</summary>
<form method="post" action="/launch"><label for="username">CTFd Username</label><input id="username" name="username" maxlength="64" required>
<label for="user_id">CTFd User ID</label><input id="user_id" name="user_id" type="number" min="1" required>
<p><button type="submit">Launch Workspace</button></p></form></details>""")


@app.post("/launch")
async def launch(request: Request):
    values = await form_values(request)
    try:
        row, ready = launch_workspace(values.get("user_id", ""), values.get("username", ""))
        suffix = "" if ready else "?message=Jupyter%20is%20still%20starting;%20refresh%20in%20a%20moment"
        return RedirectResponse(f"/workspace/{row['ctfd_user_id']}{suffix}", status_code=303)
    except (ValueError, RuntimeError, DockerException) as exc:
        return page("AI Cyber Range", '<p>Correct the details or retry after checking the range services.</p><form method="post" action="/launch"><label>CTFd Username</label><input name="username" required><label>CTFd User ID</label><input name="user_id" type="number" min="1" required><p><button>Launch Workspace</button></p></form>', str(exc))


@app.get("/workspace/{user_id}", response_class=HTMLResponse)
def workspace(user_id: int, request: Request):
    notebook = request.query_params.get("notebook")
    if notebook is not None and notebook not in WORKSPACE_NOTEBOOKS:
        raise HTTPException(status_code=400, detail="Unknown challenge notebook.")
    row = get_row_by_uid(user_id)
    if not row:
        response = page("Workspace not found", "<p>No workspace is mapped to this user ID.</p>")
        response.status_code = 404
        return response
    try:
        client = docker_client()
        try:
            client.ping()
            row = refresh_row(row, client, probe_ready=True)
        finally:
            client.close()
    except DockerException as exc:
        return page("Workspace status unavailable", "<p>Docker Desktop may be stopped. Retry after it is available.</p>", str(exc))
    if notebook and row["status"] == "Ready":
        return RedirectResponse(workspace_url(row, notebook), status_code=303)
    message = request.query_params.get("message", "")
    notice = f'<p class="notice">{html.escape(message)}</p>' if message else ""
    url = html.escape(row["url"], quote=True)
    browser_address = html.escape(f"{PUBLIC_BASE}:{int(row['host_port'])}/lab")
    status_value = row["status"]
    status = html.escape(status_value)
    status_class = {
        "Ready": "status-ready",
        "Starting": "status-starting",
        "Stopped": "status-stopped",
        "Missing": "status-stopped",
    }.get(status_value, "status-stopped")
    can_start = status_value in {"Stopped", "Missing"}
    can_stop = status_value in {"Ready", "Starting"}
    can_open = status_value == "Ready"
    start_class = "" if can_start else "inactive"
    stop_class = "stop-running" if can_stop else "inactive"
    start_disabled = "" if can_start else " disabled"
    stop_disabled = "" if can_stop else " disabled"
    open_url = (
        f'<a href="{url}">{browser_address}</a>'
        if can_open
        else f"<code>{browser_address}</code>"
    )
    open_action = (
        f'<a class="button" href="{url}" target="_blank" rel="noopener">Open Workspace</a>'
        if can_open
        else '<span class="button inactive" aria-disabled="true">Open Workspace</span>'
    )
    status_message = {
        "Starting": f'<p class="notice">Jupyter is still starting. This page refreshes automatically in a few seconds. <a href="/workspace/{user_id}">Refresh now</a>.</p>',
        "Stopped": '<p class="notice">The workspace is stopped. Start it to open Jupyter.</p>',
        "Missing": '<p class="notice">The workspace container is missing. Starting it will create a fresh workspace.</p>',
    }.get(status_value, "")
    body = f"""{notice}<p><b>Workspace status:</b> <span class="status {status_class}">{status}</span></p>{status_message}<p><b>CTFd account:</b> {html.escape(row['ctfd_username'])} (ID {row['ctfd_user_id']})<br><b>Workspace:</b> {html.escape(row['workspace_id'])}<br><b>Docker container:</b> <code>{html.escape(row['container_name'])}</code></p>
<p><b>Browser address:</b> {open_url}<br><small>Your computer uses port {int(row['host_port'])}; Jupyter listens on port 8888 inside the container.</small></p><p>{open_action}</p>
<div class="actions"><form method="post" action="/workspace/{user_id}/stop"><button class="{stop_class}"{stop_disabled}>Stop Workspace</button></form>
<form method="post" action="/workspace/{user_id}/start"><button class="{start_class}"{start_disabled}>Start Workspace</button></form></div>
<div class="reset-area"><a class="button secondary" href="/workspace/{user_id}/reset-confirm">Reset and Reboot Workspace</a></div>
<p><a class="button secondary" href="{html.escape(CTFD_PUBLIC_URL, quote=True)}">Back to CTFd</a></p>"""
    return page(
        "Participant Workspace",
        body,
        refresh_seconds=4 if status_value == "Starting" else None,
    )


@app.post("/workspace/{user_id}/{action}")
def workspace_action(user_id: int, action: str):
    if action not in {"start", "stop", "reset"}:
        return page("Unknown action", "<p>That workspace action is not available.</p>")
    try:
        row, ready = perform_action(user_id, action)
        message = "Workspace is ready." if ready else ("Workspace stopped." if action == "stop" else "Workspace is starting; refresh shortly.")
        return RedirectResponse(f"/workspace/{user_id}?message={quote(message)}", status_code=303)
    except (ValueError, RuntimeError, DockerException) as exc:
        return page("Workspace action failed", f'<p><a href="/workspace/{user_id}">Return to workspace</a></p>', str(exc))


@app.get("/workspace/{user_id}/reset-confirm", response_class=HTMLResponse)
def reset_confirm(user_id: int):
    row = get_row_by_uid(user_id)
    if not row:
        return page("Workspace not found", "<p>No workspace exists for this ID.</p>")
    return page("Confirm Reset and Reboot Workspace", f'<p class="warning"><strong>This permanently deletes the current workspace files and notebooks.</strong> Confirming will create a fresh workspace.</p><form method=post action=/workspace/{user_id}/reset><button>Confirm Reset and Reboot</button></form><p><a href=/workspace/{user_id}>Cancel</a></p>')


@app.get("/workspaces", response_class=HTMLResponse)
def instructor_workspaces():
    return RedirectResponse(f"{CTFD_PUBLIC_URL}/admin/workspaces", status_code=303)


@app.get("/api/workspaces")
def workspaces_api(x_workspace_launch_key: str = Header(default="")):
    require_workspace_admin_key(x_workspace_launch_key)
    return JSONResponse(workspace_inventory())


@app.get("/health")
def health():
    return {"status": "ok"}
