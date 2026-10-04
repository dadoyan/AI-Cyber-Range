# CTFd Workspace Launcher

Adds **Launch Workspace** and **My Workspace** items to the signed-in CTFd
navigation. Launch Workspace opens the selected starter notebook. My Workspace
opens that user's status and controls page, where they can open, start, stop, or
reset their workspace without entering a username or ID. On first use, it
creates the workspace and shows its controls. A stopped existing workspace
stays stopped until the user selects Start.

Challenge pages can request a starter notebook with
`/workspace-launch?notebook=patchguard%2Fpatchguard_starter.ipynb`,
`/workspace-launch?notebook=xray_red/red_team_fgsm.ipynb`,
`/workspace-launch?notebook=xray_blue/blue_team.ipynb`, or
`/workspace-launch?notebook=llm_safety/llm_safety_starter.ipynb`. The launcher
validates the notebook path and opens that file in the same personal Jupyter
workspace once it is ready. The ordinary navigation link opens
`start_here.ipynb`, which links to all available exercises.

CTFd's per-account `solved_by_me` status marks completed challenge cards. The
plugin highlights those cards in green with a visible **Solved** badge; new
accounts receive the same behavior automatically as they submit flags.

X-Ray Blue remains visible as a disabled prerequisite preview for each account
until that account submits its own X-Ray Red flag. A valid Arena Red submission
or a successful static Red run can award that flag. The static Blue notebook
launch additionally checks for a class-wide accepted static Red artifact;
the Arena Blue notebook uses its own match inbox and does not need that batch.
When the static artifact exists, its launch prepares the current successful Red
images in the participant's personal `xray_blue` folder.

The plugin is loaded from CTFd's `plugins` directory when the CTFd container
starts. Its service-to-service launch request uses the existing CTFd secret
configured for the range.

The `start_here.ipynb` readiness cell also displays a **Manage my workspace**
shortcut that returns to the authenticated CTFd route.

The admin-only **Workspaces** page includes an **Open MLflow** link. Its
destination comes from `MLFLOW_PUBLIC_URL` (default `http://localhost:5000`).

CTFd administrators also get a **Workspaces** link in the admin navigation.
That page lists mapped participant workspaces and provides stop and confirmed
delete actions. Stopping preserves workspace files. Deleting removes the
workspace container and its files; it does not delete the CTFd user or change
their score. The page and launcher administration API are restricted to CTFd
administrators and the internal shared key.
