# AI Cyber Range workspace

Open `start_here.ipynb` for exercise links and a readiness check. Each exercise
has its own folder:

- `patchguard/`: train a window classifier and evaluate patch robustness.
- `llm_safety/`: test and optimise prompts against the protected LLM evaluator.
- `xray_red/`: create a bounded adversarial X-ray using FGSM.
- `xray_blue/`: evaluate a defence against the accepted Red image.

Launch X-Ray Red and Blue from their CTFd challenge pages so the range selects
your role and stages the accepted image in the Blue notebook. Keep your edits
and generated models or images inside the corresponding exercise folder.

Run each notebook's setup cells before its TODO and submission cells. The
workspace provides your account identity to the protected evaluators. If an
evaluator returns a flag, submit it to the corresponding CTFd challenge.

Personal workspaces are opened from **Launch Workspace** in CTFd. The welcome
notebook links to workspace controls and checks evaluator availability without
submitting a candidate or changing challenge points.
