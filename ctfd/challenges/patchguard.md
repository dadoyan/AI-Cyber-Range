# PatchGuard - Adversarial Patch Defense

Train the window-based MNIST classifier from the PatchGuard starter and submit
its PyTorch `state_dict`. The protected evaluator checks clean accuracy on a
held-out dataset and robust accuracy against precomputed localized patch
attacks. Both accuracy requirements must pass:

- Clean accuracy: at least **{{CLEAN_THRESHOLD}}**
- Robust accuracy: at least **{{ROBUST_THRESHOLD}}**

This is a simplified PatchGuard-inspired exercise based on cropped sliding
windows and majority voting. It is an educational baseline, not a certified
defense or a robustness guarantee.

[Open the PatchGuard workspace](/workspace-launch?notebook=patchguard%2Fpatchguard_starter.ipynb).
Run the notebook cells in order, complete the three marked TODO functions, and
submit the resulting model with `submit_patchguard_model()`. If the evaluator
returns a flag, submit it to this challenge for points.
