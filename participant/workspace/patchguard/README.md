# PatchGuard starter workspace

Open `patchguard_starter.ipynb` from the PatchGuard CTFd challenge. It trains
the fixed `BasicCNN` on the locally bundled first 3,000 MNIST training examples
and first 500 public test examples; it does not need internet access.

Complete `extract_windows`, `patchguard_predict`, and `patchguard_accuracy`,
run the notebook in order, save `/workspace/patchguard_model.pt`, then call
`submit_patchguard_model()`. The launcher supplies your CTFd user ID and name
for the administrative MLflow record. The target returns the flag only after
both hidden clean and robust accuracy thresholds pass.

The evaluator receives only your model state dictionary. Its hidden MNIST
examples, patch locations, reference weights, and flag are kept server-side.
