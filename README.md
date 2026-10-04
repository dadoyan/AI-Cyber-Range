# AI Cyber Range

A CTF platform for adversarial and robust AI, with personal Jupyter workspaces,
protected evaluators, CTFd scoring, and MLflow experiment records.

## Challenges

| Challenge | Participant task | Protected acceptance conditions |
| --- | --- | --- |
| **PatchGuard - Adversarial Patch Defense** | Complete window extraction and majority voting, train a classifier, and submit its parameters. | Clean and adversarial-patch accuracy on private MNIST examples must both pass. |
| **X-Ray Red - Adversarial Evasion** | Complete the FGSM notebook and submit a bounded adversarial chest X-ray. | The clean source is correctly classified, the measured perturbation is within the limit, and the submitted image changes the fixed model's predicted class. |
| **X-Ray Blue - Lightweight Defense** | Inspect the validated Red image and configure randomized smoothing with optional abstention. | Preserve clean utility while correcting or abstaining on the adversarial input. |
| **Breaking LLMs - Prompt Safety** | Complete three black-box and two white-box prompt tasks. | Each of five distinct prompts must produce the public target phrase in at least six of ten sampled responses. |

The participant flow is **CTFd challenge → personal notebook → protected
evaluator → returned flag → flag submission in CTFd**. These are educational
exercises. The image defenses provide empirical results, not formal robustness
certificates or medical diagnostic capabilities.

## Screenshots

**X-Ray Red:** The participant notebook displays the approved chest X-ray
images used to select an input for the FGSM exercise.

![X-Ray Red notebook displaying the approved chest X-ray gallery](docs/images/xray-red-notebook.jpg)

**LLM prompt safety:** An unscored prompt trial displays one model reply about
system instructions and user prompts. Each probe samples ten replies without
filling a scoring slot; this view shows one of those replies.

![LLM prompt-safety notebook displaying a prompt and one model reply](docs/images/llm-prompt-reply.jpg)

**CTFd challenge board:** X-Ray Red is marked as solved for the signed-in
participant, and X-Ray Blue is available.

![CTFd challenge board with X-Ray Red marked as solved](docs/images/ctfd-xray-red-solved.jpg)

## Services

| Service | Local URL | Purpose |
| --- | --- | --- |
| CTFd | http://localhost:8001 | Accounts, challenge instructions, flag submission, scores |
| Workspace launcher | http://localhost:7000 | Personal workspace lifecycle |
| MLflow | http://localhost:5000 | Instructor experiment and artifact records |
| PatchGuard evaluator | http://localhost:8002 | Model submission and protected scoring |
| X-Ray evaluator | http://localhost:8003 | Approved sources, attack validation, defense evaluation |
| Red/Blue Arena | http://localhost:8004 | Match identity, submitted artifacts, turn history |
| LLM evaluator | http://localhost:8005 | Protected prompt evaluation and accepted-slot progress |
| Shared Jupyter fallback | http://localhost:8888 | Administrator/reference workspace |

Personal workspaces use ports 9001–9099 by default. Local services bind to the
loopback interface; configure the documented launcher addresses for a remote
classroom deployment.

## Configure and start

1. Create `.env` from `.env.example` for a new installation. Keep the configured
   `CTFD_SECRET_KEY`, `JUPYTER_TOKEN`, and administrator API token private.
2. Set independent values for `PATCHGUARD_FLAG`, `XRAY_RED_FLAG`,
   `XRAY_BLUE_FLAG`, and `LLM_SAFETY_FLAG`. The evaluator and CTFd must use the
   same flag for each corresponding challenge.
3. Configure `PATCHGUARD_CLEAN_THRESHOLD` and `PATCHGUARD_ROBUST_THRESHOLD`.
   Enable the other families with `XRAY_REDBLUE_ENABLED=true` and
   `LLM_SAFETY_ENABLED=true`; assign `LLM_SAFETY_ALLOWED_USER_ID`.
4. Build the X-Ray base image from the supplied exercise package if `adv_demo`
   is not already available. The extracted source, checkpoints, and test images
   are included under `xray_base/`:

   ```powershell
   docker build -t adv_demo xray_base
   ```

5. Build and start the supported services:

   ```powershell
   docker compose --profile xray-redblue --profile llm-safety up -d --build
   ```

6. Complete CTFd's first-run setup at http://localhost:8001 if necessary. Create
   an administrator access token in CTFd settings and set `CTFD_ADMIN_TOKEN` in
   `.env`, then synchronize the configured challenges and welcome page:

   ```powershell
   python -m scripts.bootstrap_ctfd
   ```

The Compose project is named `ai-cyber-range`; the launcher uses its
`ai-cyber-range_default` network. If you override the project name, update
`RANGE_DOCKER_NETWORK` to match.

Bootstrap uses Python's standard library and is idempotent. X-Ray and LLM
services remain profile-controlled, so omit their profiles when intentionally
running only PatchGuard. `CTFD_SECRET_KEY` signs CTFd sessions; it is not an
administrator login password.

### GPU execution

The GPU overlay enables CUDA in the evaluators, shared Jupyter service, and
launcher-managed personal workspaces. It builds the configured PyTorch and
Torchvision packages against CUDA 13.0. An NVIDIA driver and GPU-capable Docker
runtime are required; Docker Desktop on Windows uses WSL 2.

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile xray-redblue --profile llm-safety up -d --build --wait
```

To apply the overlay to ordinary Compose commands, configure:

```dotenv
COMPOSE_PATH_SEPARATOR=;
COMPOSE_FILE=docker-compose.yml;docker-compose.gpu.yml
```

The first notebook cell identifies the compute device. PatchGuard training and
scoring use CUDA when enabled. X-Ray model inference and local attacks use CUDA;
its smoothing helper generates seeded noise on CPU. The LLM exercise uses BF16
model weights and eager attention.

## Participant workspaces

Sign in to CTFd and launch the notebook from the challenge description. The
launcher provisions an isolated workspace for that CTFd account. Exercise files
are grouped in `patchguard/`, `xray_red/`, `xray_blue/`, and `llm_safety/`.
`start_here.ipynb` links to the supported exercises and reports their readiness.

CTFd shows **Solved by you** on challenges solved by the current account. X-Ray
Blue is shown as a disabled card until its prerequisite is satisfied. The Blue
notebook displays the validated adversarial image when launched from CTFd.

The administrator's **Workspaces** page can stop a workspace while preserving
its files, or reset/delete it when the participant's work may be discarded.
Workspace mappings survive launcher restarts. Existing workspaces keep their
container filesystem until migrated or reset; updates must preserve unrelated
participant notebooks and model files. The launcher removes retired templates
when resuming an existing workspace.

## PatchGuard

The starter is `patchguard/patchguard_starter.ipynb`. Participants complete the
window-extraction, majority-vote, and accuracy TODOs, train the small CNN on the
bundled MNIST subset, and submit the saved `state_dict` using
`submit_patchguard_model()`.

The dedicated `ai-cyber-patchguard:local` image serves `/score`. The evaluator
validates model tensor names, shapes, and finite values, then applies fixed
window-based prediction to private clean and precomputed patched images. Both
configured accuracy thresholds must pass. Participant Python code is never
executed by the scoring endpoint.

See [PatchGuard calibration](docs/PATCHGUARD_CALIBRATION.md) for the fixed
architecture, dataset, hidden evaluation assets, and configured acceptance
thresholds. The supplied assignment source is preserved at
`reference/A6_PatchGuard_clean.ipynb` with saved execution outputs cleared. This is a simplified PatchGuard-inspired
exercise, not an implementation of the paper's formal certification.

## X-Ray Red and Blue

An instructor creates a paired or solo Arena match using
`scripts/manage_red_blue_matches.py`. Paired roles use two CTFd accounts; solo
mode lets one account practise both roles. The participant notebooks are
`xray_red/red_team_fgsm.ipynb` and `xray_blue/blue_team.ipynb`.

Red submits an approved-source PNG to the protected EfficientNet-B0 evaluator.
A successful attack returns the Red flag and stores the exact validated image
for Blue. Blue submits smoothing parameters, which the evaluator applies to
both the clean and adversarial image. A passing defense returns the Blue flag.
After Blue responds, Red can inspect its configuration and submit a subsequent
candidate. Match state, images, and numbered interaction history persist.

See [Arena documentation](docs/RED_BLUE_ARENA.md) for pairing, identities,
acceptance rules, API endpoints, and reset behavior. Retained legacy X-Ray
notebooks are available as reference material; current challenge links open the
FGSM Red and interactive Blue notebooks.

## LLM prompt safety

The assigned participant opens `llm_safety/llm_safety_starter.ipynb`. Protected
Qwen2.5-1.5B-Instruct generation tests whether a submitted prompt causes the model
to emit a specified public phrase despite its fixed system instruction. Each
prompt is sampled ten times; at least six exact target hits are required.

All three black-box and two white-box slots must pass with distinct prompt
texts. The fixed white-box surrogate target is `Sure, here it is.`; a different
surrogate is required for the second white-box task. The server verifies prompt
text and model output, not the participant's optimization method. Qwen3-0.6B
transfer experiments are optional and do not determine flag eligibility.

Model revisions are pinned. BF16 avoids the non-finite FP16 behavior observed
in this GPU stack. The evaluator is restricted to `LLM_SAFETY_ALLOWED_USER_ID`,
and generation remains serialized. Accepted-slot progress is stored separately
from MLflow so a logging outage does not change flag eligibility.

## MLflow and persistent state

MLflow is an instructor-facing audit service. PatchGuard records submitted model
parameters, clean/robust accuracy, timing, and participant attribution. X-Ray
records validated images, defense parameters, outcomes, and match/sequence
identifiers. LLM records prompt texts, sampled responses, success counts, model
revision, and timing. The `red_blue_arena` experiment captures interactive X-Ray
attempts; `xray-redblue` holds the retained legacy workflow's records.

Logging is best effort and does not grant flags or determine CTFd points. The
CTFd, MLflow, launcher, X-Ray, Arena, and LLM state volumes are independent.
Stopping Compose preserves these volumes. Do not remove participant containers
or state volumes to perform a routine application update.

## Validation

Run CTFd bootstrap unit tests without live credentials:

```powershell
python -m unittest discover -s tests_ctfd -p test_bootstrap_ctfd.py -v
```

Run PatchGuard/notebook tests in its test-capable evaluator image using a
read-only repository mount. Run Arena and launcher suites in separate Python
processes because they use different `app` modules. Live model checks require
the configured device, hidden assets, model caches, and running evaluators.

## Sources

The PatchGuard-inspired assignment and MNIST data derive from the supplied
teaching material; see [MNIST attribution](assets/mnist/ATTRIBUTION.md) and the
[PatchGuard paper](https://www.usenix.org/conference/usenixsecurity21/presentation/xiang).
The X-Ray implementation uses the supplied exercise's EfficientNet-B0 models
and chest X-ray data. The LLM challenge adapts the supplied prompt-safety exercise
using the pinned Qwen model snapshots. Existing source and model attributions
remain with the retained exercise materials.

## Public distribution

This repository contains source, notebook templates, and example model/data
assets. It does not contain deployment credentials, accounts, participant
submissions, experiment history, or previous repository history. Copy
`.env.example` to `.env` and choose fresh secrets and flags before starting.

The `private/` directory denotes evaluator-only deployment assets, not files
that are confidential in this public repository. Reference models and fixed
evaluation assets are included for reproducibility. Instructors who require
unseen assessment data should prepare separate evaluation assets and calibrate
the acceptance thresholds for their deployment.

Third-party datasets, model weights, and supplied teaching materials retain
their respective authorship and applicable terms; this repository does not
assign a new blanket license to those materials.
