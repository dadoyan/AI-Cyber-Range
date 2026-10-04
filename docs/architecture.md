# AI Cyber Range roles

- **CTFd** presents PatchGuard, X-Ray Red, X-Ray Blue, and LLM challenges and
  validates returned flags for points. It performs no model inference.
- **Workspace launcher** maps CTFd accounts to isolated Jupyter containers and
  manages launch, resume, stop, reset, and retired-template cleanup.
- **Participant workspaces** provide the supplied notebooks and a writable
  environment for model training, adversarial inputs, and defense experiments.
- **PatchGuard evaluator** validates uploaded model parameters and independently
  measures clean and patched accuracy against private assets.
- **X-Ray evaluator and Arena** validate approved-source attacks, store artifacts,
  evaluate defense configurations, and maintain Red/Blue match history.
- **LLM evaluator** samples protected model responses and stores per-account
  accepted-slot progress.
- **MLflow** stores evaluation telemetry and artifacts independently of scoring.

The services communicate over the Compose network. Participant notebooks obtain
flags from the appropriate protected service; CTFd awards points only after
participants submit those flags. Evaluator and workspace state persist across
routine service restarts.
