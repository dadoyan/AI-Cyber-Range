from __future__ import annotations

import logging
import json
import os
import secrets
from pathlib import Path
from urllib.parse import urlencode

import requests
from flask import (
    abort,
    current_app,
    flash,
    g,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_from_directory,
    session,
    url_for,
)

from CTFd.plugins import (
    register_admin_plugin_menu_bar,
    register_plugin_script,
    register_plugin_stylesheet,
    register_user_page_menu_bar,
)
from CTFd.utils.decorators import admins_only, authed_only
from CTFd.utils.user import get_current_user

from .xray_gate import BLUE_NAME, reconcile_blue_visibility, red_solved_by_user

logger = logging.getLogger("ctfd.workspace_launcher")
WORKSPACE_NOTEBOOKS = {
    "start_here.ipynb",
    "patchguard/patchguard_starter.ipynb",
    "xray_red/xray_red_blue_starter.ipynb",
    "xray_blue/blue_starter.ipynb",
    "llm_safety/llm_safety_starter.ipynb",
    "xray_red/red_team.ipynb",
    "xray_red/red_team_fgsm.ipynb",
    "xray_blue/blue_team.ipynb",
}
# The static Blue flag challenge is gated by a personal Red solve. The Arena's
# Blue notebook has independent instructor pairing (or solo-mode) authorization.
BLUE_NOTEBOOKS = {"xray_blue/blue_starter.ipynb"}


def load(app):
    assets_dir = Path(__file__).resolve().parent / "assets"

    @app.get("/workspace-launcher-assets/<path:asset>")
    def workspace_launcher_asset(asset):
        if asset not in {"xray_blue_locked.js", "xray_blue_locked.css", "challenge_status.js", "challenge_status.css"}:
            abort(404)
        return send_from_directory(assets_dir, asset)

    register_plugin_script("/workspace-launcher-assets/xray_blue_locked.js")
    register_plugin_stylesheet("/workspace-launcher-assets/xray_blue_locked.css")
    register_plugin_script("/workspace-launcher-assets/challenge_status.js")
    register_plugin_stylesheet("/workspace-launcher-assets/challenge_status.css")

    ctfd_url = os.environ.get(
        "WORKSPACE_CTFD_PUBLIC_URL", "http://localhost:8001"
    ).rstrip("/")
    mlflow_url = os.environ.get("MLFLOW_PUBLIC_URL", "http://localhost:5000").strip()
    launcher_url = os.environ.get(
        "WORKSPACE_LAUNCHER_PUBLIC_URL", "http://localhost:7000"
    ).rstrip("/")
    launcher_api = os.environ.get(
        "WORKSPACE_LAUNCHER_INTERNAL_URL",
        "http://launcher:7000/internal/workspace-launch",
    )
    launcher_internal_root = launcher_api.rsplit("/", 1)[0]

    @app.before_request
    def refresh_xray_blue_for_listing():
        if request.method == "GET" and request.path.rstrip("/") in {
            "/challenges", "/api/v1/challenges"
        }:
            try:
                g.xray_blue_ready = reconcile_blue_visibility()
            except Exception:
                logger.exception("Could not refresh X-Ray Blue visibility")
                g.xray_blue_ready = False
            try:
                # Either Arena or the static evaluator can award the Red flag.
                # A personal solve unlocks the Blue card; the static Blue
                # notebook still checks its shared artifact on launch.
                g.xray_blue_access = red_solved_by_user(get_current_user())
            except Exception:
                logger.exception("Could not check the current user's X-Ray Red solve")
                g.xray_blue_access = False
        if request.method == "POST" and request.path == "/api/v1/challenges/attempt":
            body = request.get_json(silent=True)
            if isinstance(body, dict) and body.get("challenge_id") is not None:
                from CTFd.models import Challenges

                blue = Challenges.query.filter_by(name=BLUE_NAME).first()
                if blue is not None and str(body["challenge_id"]) == str(blue.id):
                    try:
                        reconcile_blue_visibility()
                    except Exception:
                        logger.exception("Could not reconcile X-Ray Blue before submission")
                    try:
                        ready = red_solved_by_user(get_current_user())
                    except Exception:
                        logger.exception("Could not verify this user's X-Ray Red solve")
                        ready = False
                    if ready is not True:
                        return jsonify({"success": True, "data": {
                            "status": "locked",
                            "message": "Submit your own X-Ray Red flag to unlock Blue.",
                        }}), 403

    @app.after_request
    def refresh_xray_blue_after_solve(response):
        if (request.method == "GET" and request.path.rstrip("/") == "/api/v1/challenges"
                and request.args.get("view") != "admin" and response.status_code == 200
                and getattr(g, "xray_blue_access", False) is not True):
            payload = response.get_json(silent=True)
            items = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and item.get("name") == BLUE_NAME:
                        item["type"] = "hidden"
                        item["template"] = ""
                        item["script"] = ""
                        tags = list(item.get("tags") or [])
                        if not any(tag.get("value") == "xray-blue-locked" for tag in tags if isinstance(tag, dict)):
                            tags.append({"value": "xray-blue-locked"})
                        item["tags"] = tags
                        response.set_data(json.dumps(payload))
                        break
        if request.method == "POST" and request.path == "/api/v1/challenges/attempt":
            payload = response.get_json(silent=True)
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, dict) and data.get("status") == "correct":
                try:
                    reconcile_blue_visibility()
                except Exception:
                    logger.exception("Could not refresh X-Ray Blue after a solve")
        return response

    register_user_page_menu_bar(
        "Launch Workspace", f"{ctfd_url}/workspace-launch"
    )
    register_user_page_menu_bar(
        "My Workspace", f"{ctfd_url}/my-workspace"
    )

    def launch_for_current_user():
        user = get_current_user()
        launch_key = current_app.config.get("SECRET_KEY")
        if user is None or not launch_key:
            flash("Could not identify your CTFd account. Please sign in and retry.", "danger")
            return redirect(url_for("challenges.listing"))

        notebook = request.args.get("notebook", "start_here.ipynb")
        if notebook not in WORKSPACE_NOTEBOOKS:
            abort(400, description="Unknown challenge notebook.")
        if notebook in BLUE_NOTEBOOKS:
            try:
                solved_red = red_solved_by_user(user)
                blue_ready = reconcile_blue_visibility() if solved_red else False
            except Exception:
                logger.exception("Could not check X-Ray Blue readiness")
                solved_red = False
                blue_ready = None
            if blue_ready is not True:
                message = (
                    "The static Blue notebook needs a successful static Red run to publish its shared artifact. "
                    "You can use the Arena Blue notebook for your Arena attack."
                    if solved_red else "Submit your own X-Ray Red flag to unlock Blue."
                )
                flash(message, "warning")
                return redirect(url_for("challenges.listing"))

        try:
            response = requests.post(
                launcher_api,
                json={"user_id": user.id, "username": user.name, "notebook": notebook},
                headers={"X-Workspace-Launch-Key": str(launch_key)},
                timeout=90 if notebook in {"xray_blue/blue_starter.ipynb", "xray_blue/blue_team.ipynb"} else 20,
            )
            response.raise_for_status()
            result = response.json()
        except (requests.RequestException, ValueError):
            logger.exception("Could not start workspace for CTFd user ID %s", user.id)
            flash("The workspace could not be started. Please try again.", "danger")
            return redirect(url_for("challenges.listing"))

        message = (
            "Workspace is ready."
            if result.get("ready")
            else "Workspace is starting; refresh its status in a moment."
        )
        query = {"message": message}
        query["notebook"] = notebook
        destination = f"{launcher_url}/workspace/{user.id}?{urlencode(query)}"
        return redirect(destination)

    app.add_url_rule(
        "/workspace-launch",
        endpoint="ai_cyber_workspace_launch",
        view_func=authed_only(launch_for_current_user),
        methods=["GET"],
    )

    def my_workspace():
        user = get_current_user()
        launch_key = current_app.config.get("SECRET_KEY")
        if user is None or not launch_key:
            flash("Could not identify your CTFd account. Please sign in and retry.", "danger")
            return redirect(url_for("challenges.listing"))

        headers = {"X-Workspace-Launch-Key": str(launch_key)}
        inventory_endpoint = f"{launcher_internal_root}/workspaces"
        try:
            response = requests.get(inventory_endpoint, headers=headers, timeout=15)
            response.raise_for_status()
            workspaces = response.json().get("workspaces", [])
        except (requests.RequestException, ValueError):
            logger.exception("Could not load workspace status for CTFd user ID %s", user.id)
            flash("Workspace status is unavailable. Please try again in a moment.", "warning")
            return redirect(url_for("challenges.listing"))

        matching_workspace = next(
            (item for item in workspaces if str(item.get("ctfd_user_id")) == str(user.id)),
            None,
        )
        if matching_workspace is None:
            # A first visit creates the user's workspace and lands on its controls,
            # while existing stopped workspaces remain stopped for explicit Start.
            try:
                response = requests.post(
                    launcher_api,
                    json={"user_id": user.id, "username": user.name},
                    headers=headers,
                    timeout=20,
                )
                response.raise_for_status()
                result = response.json()
            except (requests.RequestException, ValueError):
                logger.exception("Could not create workspace for CTFd user ID %s", user.id)
                flash("Your workspace could not be created. Please try Launch Workspace again.", "danger")
                return redirect(url_for("challenges.listing"))

            message = (
                "Your workspace is ready. Open it, stop it, or reset it from this page."
                if result.get("ready")
                else "Your workspace is starting. This page will show its controls when it is ready."
            )
            return redirect(
                f"{launcher_url}/workspace/{user.id}?{urlencode({'message': message})}"
            )

        return redirect(f"{launcher_url}/workspace/{user.id}")

    app.add_url_rule(
        "/my-workspace",
        endpoint="ai_cyber_my_workspace",
        view_func=authed_only(my_workspace),
        methods=["GET"],
    )

    register_admin_plugin_menu_bar("Workspaces", "/admin/workspaces")

    def admin_workspaces():
        endpoint = f"{launcher_internal_root}/workspaces"
        headers = {
            "X-Workspace-Launch-Key": str(current_app.config.get("SECRET_KEY", ""))
        }

        if request.method == "POST":
            submitted_nonce = request.form.get("nonce", "")
            expected_nonce = str(session.get("nonce", ""))
            if not expected_nonce or not secrets.compare_digest(
                submitted_nonce, expected_nonce
            ):
                abort(403)

            try:
                user_id = int(request.form.get("user_id", ""))
                action = request.form.get("action", "")
                if user_id <= 0 or action not in {"stop", "delete"}:
                    raise ValueError("Invalid workspace action")
                action_endpoint = f"{endpoint}/{action}"
                response = requests.post(
                    action_endpoint,
                    json={"user_id": user_id},
                    headers=headers,
                    timeout=20,
                )
                response.raise_for_status()
                result = response.json()
                if action == "delete":
                    flash(
                        f"Workspace for {result['username']} was deleted. The CTFd account and score remain.",
                        "success",
                    )
                else:
                    flash(f"Workspace for {result['username']} was stopped. Its files are preserved.", "success")
            except (ValueError, KeyError):
                flash("Choose a valid workspace to delete.", "danger")
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    flash("That workspace no longer exists.", "warning")
                else:
                    logger.exception("Could not delete a participant workspace")
                    flash("The workspace could not be deleted. Please retry.", "danger")
            except requests.RequestException:
                logger.exception("Could not reach the launcher to delete a workspace")
                flash("The launcher is unavailable; the workspace was not deleted.", "danger")
            return redirect(url_for("ai_cyber_admin_workspaces"))

        workspaces = []
        load_error = ""
        try:
            response = requests.get(endpoint, headers=headers, timeout=15)
            response.raise_for_status()
            workspaces = response.json().get("workspaces", [])
        except (requests.RequestException, ValueError):
            logger.exception("Could not load participant workspaces for CTFd admin")
            load_error = "Workspace information is unavailable. Check that the launcher is running."

        template = """{% extends "admin/base.html" %}
{% block content %}
<div class="jumbotron"><div class="container"><h1>Participant Workspaces</h1>
<p>Stop a workspace while preserving its files, or permanently delete it.</p>
<p><a class="btn btn-primary" href="{{ mlflow_url }}" target="_blank" rel="noopener noreferrer">Open MLflow</a></p>
</div></div>
<div class="container-fluid">
  {% for category, message in get_flashed_messages(with_categories=true) %}
    <div class="alert alert-{{ category }}">{{ message }}</div>
  {% endfor %}
  {% if load_error %}<div class="alert alert-danger">{{ load_error }}</div>{% endif %}
  <div class="alert alert-warning"><strong>Deletion is permanent.</strong> It removes that workspace container and its files. The CTFd account and score are kept.</div>
  <div class="table-responsive"><table class="table table-striped">
    <thead><tr><th>CTFd user</th><th>User ID</th><th>Workspace</th><th>Container</th><th>Port</th><th>Status</th><th>Action</th></tr></thead>
    <tbody>
    {% for item in workspaces %}
      <tr>
        <td>{{ item.ctfd_username }}</td><td>{{ item.ctfd_user_id }}</td>
        <td>{{ item.workspace_id }}</td><td><code>{{ item.container_name }}</code></td>
        <td>{{ item.host_port }}</td><td>{{ item.status }}</td>
        <td class="text-nowrap">
          <form class="d-inline" method="post" action="{{ url_for('ai_cyber_admin_workspaces') }}">
            <input type="hidden" name="nonce" value="{{ nonce }}">
            <input type="hidden" name="user_id" value="{{ item.ctfd_user_id }}">
            <input type="hidden" name="action" value="stop">
            <button class="btn btn-warning" type="submit" {% if item.status not in ['Ready', 'Starting', 'Running'] %}disabled{% endif %}>Stop Workspace</button>
          </form>
          <form class="d-inline" method="post" action="{{ url_for('ai_cyber_admin_workspaces') }}" onsubmit="return confirm('Permanently delete this workspace and its files? The CTFd account and score will remain.');">
            <input type="hidden" name="nonce" value="{{ nonce }}">
            <input type="hidden" name="user_id" value="{{ item.ctfd_user_id }}">
            <input type="hidden" name="action" value="delete">
            <button class="btn btn-danger" type="submit">Delete Workspace</button>
          </form>
        </td>
      </tr>
    {% else %}<tr><td colspan="7">No participant workspaces are registered.</td></tr>{% endfor %}
    </tbody>
  </table></div>
</div>
{% endblock %}"""
        return render_template_string(
            template,
            workspaces=workspaces,
            load_error=load_error,
            nonce=session.get("nonce", ""),
            mlflow_url=mlflow_url,
        )

    app.add_url_rule(
        "/admin/workspaces",
        endpoint="ai_cyber_admin_workspaces",
        view_func=admins_only(admin_workspaces),
        methods=["GET", "POST"],
    )
