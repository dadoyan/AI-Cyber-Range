# Interactive X-Ray Red/Blue Arena

The Arena adds a paired, iterative Red/Blue workflow alongside the original
class-wide X-Ray round. It uses the existing standard EfficientNet-B0 checkpoint,
test images, Launcher, Docker network, and MLflow service. The Arena is optional;
the existing static workflow continues to use `xray-redblue`'s `/api/red/*` and
`/api/blue/*` routes and its separate persisted round state.

## Architecture

```text
                        CTFd
                         |
                   Lab Launcher
                    /          \
             Red Jupyter     Blue Jupyter
                    \          /
                     Arena Service
                    /      |       \
                SQLite  Artifact    Protected X-Ray
                         Store       Evaluator
                                      |
                                    MLflow
```

The participant-to-participant interaction path is Red workspace → Arena → Blue
workspace, then Blue → Arena → protected evaluator → Arena → Red. MLflow mirrors
administrative telemetry; it is not used to deliver artifacts or results.
CTFd remains the entry/scoring site. Arena returns the configured X-Ray Red flag
after a valid attack and the X-Ray Blue flag after a successful defense; the
participant submits each flag in CTFd. Arena never writes CTFd solves itself.

## Start and pair a match

The interactive Arena uses the optional `xray-redblue` Compose profile:

```powershell
docker compose --profile xray-redblue up -d --build xray-redblue arena-service launcher
```

Create two ordinary CTFd accounts, then pair their CTFd numeric IDs and exact
usernames. The Launcher registers each workspace token when that user opens or
starts a workspace. In PowerShell, for example:

```powershell
python scripts/manage_red_blue_matches.py create --red-user-id 12 --red-username red-user --blue-user-id 13 --blue-username blue-user
```

The CLI reads `CTFD_SECRET_KEY` from the local `.env` file to authenticate its
admin call; it does not print the key. It can list, show, and reset matches:

```powershell
python scripts/manage_red_blue_matches.py list
python scripts/manage_red_blue_matches.py show <match-id>
python scripts/manage_red_blue_matches.py reset <match-id>
```

For one person to practice both roles with one CTFd account, an instructor can
create a solo match instead:

```powershell
python scripts/manage_red_blue_matches.py create-solo --user-id 3 --username learner
```

The same workspace then opens `xray_red/red_team_fgsm.ipynb` and
`xray_blue/blue_team.ipynb`. A successful Red submission appears in that
workspace's Blue inbox and returns the Red flag; the participant submits it in
CTFd, then switches notebooks to respond. A successful Blue response returns
the Blue flag. Solo
mode is an explicit instructor assignment, not automatic pairing. Existing
two-account matches remain unchanged. Both Arena and static workflows use the
same CTFd challenge flags, but keep separate artifacts and evaluator state.

After pairing, each participant signs in to CTFd and opens the role-specific
challenge link. Red opens `red_team_fgsm.ipynb`; Blue opens `blue_team.ipynb`. If a
workspace was already running at pairing time, opening it from CTFd again causes
the Launcher to register its current workspace credential with Arena.

The minimal instructor API is available at `http://localhost:8004/docs` and
requires `X-Arena-Admin-Key`. The key is the same configured range
`CTFD_SECRET_KEY`; keep it out of notebooks and participant requests.

## Identity and pairing

The Launcher injects a random `ARENA_TOKEN` when it creates a managed workspace
and registers the CTFd user ID, username, workspace ID, and token with Arena.
Arena stores only the token SHA-256 digest. Older workspaces that predate this
variable can use their existing per-workspace Jupyter token until recreated.
Participant calls send only `Authorization: Bearer ...`; match and role are
looked up from Arena's database. Client-provided user IDs, match IDs, and roles
are not trusted.

Instructor pairing assigns one Red and one Blue participant to an active match;
solo assignment grants both roles to one account.
Each CTFd user can belong to only one active match. The `/api/identity` endpoint
lets a notebook display the server-assigned identity and role.

## Attack and artifact rules

Arena accepts only a PNG tied to a `source_id` from the finite approved source
list. The protected evaluator selects its own clean image from the same fixed
test subset and requires that source to be correctly classified by the standard
EfficientNet-B0. It checks a 256×256 RGB PNG, measured normalized pixel-space
L-infinity distance no greater than `0.02`, and untargeted top-1
misclassification. Red-supplied metrics are descriptive metadata only. Only
valid attacks enter the Blue inbox. One unresolved attack is allowed per match.

The authoritative submitted bytes are stored under the persistent
`arena-data` volume at `artifacts/<match-id>/red/attack_NNN.png`. Arena computes
and stores SHA-256; Blue's helper verifies the same digest after download. Blue
submits only an attack ID and bounded configuration, never a replacement copy
of the Red image.

## Blue defense and grading

Blue may experiment locally and submits `sigma` in `[0, 0.1]`, 10–500 noisy
views, and an abstention threshold in `[0, 1]`. The protected evaluator fetches
the authoritative attack file from Arena storage and repeats the randomized
noisy-view vote with a deterministic seed derived from the match, attack,
parameters, and evaluator version. It reports clean utility, undefended and
defended predictions, vote confidence, defense success, and whether the attack
survived. A defense succeeds only when the clean source remains correctly
classified with adequate vote confidence and the attack is corrected or
abstained. This prevents an always-abstain configuration from passing.

After Blue responds, Red can read the current Blue parameters and adapt. Sequence
numbers represent implicit rounds; there is no scheduler. Participants refresh
the notebook cells manually.

The notebook defaults to 100 samples. On the CPU evaluator used for validation,
protected evaluation of both clean and adversarial images took 0.92 s at 10
samples, 4.42 s at 50, 9.10 s at 100, and 40.43 s at 500. Repeating the same
100-sample attack/configuration produced the same result (8.69 s). These are
single-host measurements, not a latency guarantee; 500 samples is intentionally
available as a slower optional experiment.

On source 0, the tested attacks remained adversarial at `sigma=0.005` with 100
samples while clean utility passed. Larger noise broke clean utility on that
source, and sweeps over `sigma=0.0–0.1` did not defend those specific attacks.
A separate live validation on approved source 3 demonstrated a Blue win: Red
submitted a sparse-FGSM candidate at `epsilon=1/255` with a 1% pixel mask; Blue
used `sigma=0.005`, 100 samples, and threshold `0.75`. The protected evaluator
preserved clean utility and returned `ABSTAIN` on the adversarial image at
0.64 vote confidence, below the threshold, with `attack_survived=false` and
`defense_success=true`. The Blue notebook's separate local estimate was
`NORMAL` at 0.51 confidence (51 of 100 votes); it uses its own seed, while the
protected evaluator uses a match-derived seed. This is a tested example, not a
guarantee for arbitrary attacks.

## API outline

Participant endpoints (all require the opaque workspace bearer token):

| Endpoint | Role | Purpose |
|---|---|---|
| `GET /api/identity` | Red or Blue | View server-assigned match and role |
| `GET /api/sources`, `GET /api/sources/{source_id}` | Red or Blue | List/download approved clean sources |
| `GET /api/model` | Red or Blue | Download the fixed checkpoint for local experiments |
| `POST /api/red/attacks` | Red | Validate an image, submit it, and return the Red flag |
| `GET /api/red/flag` | Red | Recover the flag after a valid attack |
| `GET /api/blue/pending` | Blue | Refresh the pending inbox |
| `GET /api/attacks/{attack_id}/artifact` | Blue | Download the authoritative Red PNG |
| `POST /api/blue/responses` | Blue | Evaluate a defense and return the Blue flag on success |
| `GET /api/blue/flag` | Blue | Recover the flag after a successful defense |
| `GET /api/red/latest-response` | Red | Read the latest defense and result |
| `GET /api/match/history`, `GET /api/match/status` | Red or Blue | Inspect sequence history and current match state |

Match status includes both assigned participants, the pending sequence, the
latest attack/response state, and the latest Blue configuration.

The Red notebook labels its attack-generator budget as **declared ε** and the
protected evaluator's measurement of the submitted PNG as **measured L∞**.
The evaluator continues to enforce the published overall maximum. Both role
notebooks show a compact sequence table with Red's class flip and budgets,
Blue's smoothing settings, the defense outcome, and Red/Blue event times in
UTC. A role-aware turn message says whether the participant should act or wait,
which sequence is active, and the next suggested step. The notebooks also use
clickable step progress, labeled result cards, side-by-side image comparisons,
team color accents, confidence bars, a measured-L∞ meter, an approved-source
thumbnail gallery, expandable technical details, and colored defense-result
markers in the history table.

Launcher registration and admin routes are internal/key-protected. Admin
operations create/list/show/reset matches. Reset removes Arena attack and
response rows and that match's Arena-managed files, resets sequence numbering,
and preserves CTFd accounts and workspaces. Resetting a Jupyter workspace does
not clear Arena history. MLflow audit runs remain in MLflow after Arena reset.

## Persistence and telemetry

SQLite state (`arena.db`) and authoritative artifacts persist in the named
`arena-data` Docker volume. Restarting Arena does not clear either. Red and Blue
attempts are mirrored to the `red_blue_arena` MLflow experiment with role,
match, sequence, CTFd identity, parameters, measured metrics, and the Red PNG or
Blue protected-result JSON. MLflow failures are logged and do not roll back
Arena state or block the interaction. Participants do not access MLflow to
exchange files.
Flag values are returned only to qualifying participants and are not written
to Arena's MLflow records.

## Limits of this PoC

- Pairing is instructor-created; there is no matchmaking, tournament, timer, or
  scheduler.
- Each match has one unresolved attack and uses manual notebook refresh.
- Arena identity uses opaque bearer tokens and Launcher registration, not full
  SSO.
- Red may see Blue's current defense parameters by design.
- The evaluator uses a finite approved source set and the existing fixed model.
- Evaluator CPU/GPU resources may be shared with other range workloads.
- Randomized smoothing is an instructional noisy-view defense, not a certified
  robustness guarantee.
- Paired roles use separate workspaces; solo mode uses one workspace. No shared
  volume, Docker socket, or direct MLflow artifact browsing is introduced.
