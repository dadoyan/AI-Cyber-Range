# PatchGuard source and evaluation configuration

## Source notebook

The reference assignment is `reference/A6_PatchGuard_clean.ipynb` (canonical LF
SHA-256: `ed2db0bbc59c7559b3ffdfe0f7d935cd17e99d483cb122cd29b593c3e04c0166`). Saved outputs and execution metadata have been cleared;
code, instructional prose, source metadata, and kernel information are preserved.
It uses MNIST, the first 3,000 training and first 500 public test examples, and
the supplied LeNet-style `BasicCNN`. Its baseline trains for 30 epochs; its
window model trains for 100 epochs. The patch is 12×12, windows are 16×16, and
window stride is 2. The source sets `patch_x=patch_y=12`, which evaluates to
top-left `(2,2)` on a 28×28 image. It optimizes an independent patch for each
image using untargeted cross-entropy ascent for 20 steps at 8/255, clamps patch
deltas to `[-1,1]`, and clips output pixels to `[0,1]`.

The source prose describes zeroing pixels outside each window; its code instead
passes cropped 16×16 windows directly to the CNN. The range uses that cropped
window behavior. The participant notebook leaves `extract_windows`,
`patchguard_predict`, and `patchguard_accuracy` as TODOs.

## Range implementation

The launcher deploys `patchguard/patchguard_starter.ipynb` to each personal
workspace. It uses bundled offline MNIST files. The protected evaluator selects
its configured CPU or GPU device, loads the fixed `BasicCNN(16)` architecture,
and accepts a validated state dictionary without executing participant code.
It checks private clean examples and precomputed adversarially patched examples
using majority voting over image windows.

The hidden evaluation set contains 100 clean MNIST images and 500 adversarial
variants across five fixed locations. The asset builder uses a fixed baseline
and 20 untargeted cross-entropy-ascent steps at `8/255`. Hidden images, labels,
locations, and adversarial tensors are stored under
`private/patchguard/eval_assets/`; this directory is mounted read-only into the
evaluator and is not copied to participant workspaces. Its `metadata.json`
defines asset-generation parameters and provenance, rather than participant
experiment history.

## Functional assets and acceptance settings

`private/patchguard/reference_train.py` generates the baseline, reference,
random-control, and one-epoch weak-control model weights.
`private/patchguard/build_eval_assets.py` builds the private evaluation tensors.
Canonical model weights, the hidden MNIST seed data, evaluation tensors, and
asset metadata are retained so the challenge can be evaluated and its assets
can be rebuilt. Source metrics and historical benchmark reports are not required
for evaluation.

`PATCHGUARD_CLEAN_THRESHOLD` and `PATCHGUARD_ROBUST_THRESHOLD` configure the two
acceptance cutoffs. Both requirements must pass before the evaluator returns
a flag. The configured deployment uses 0.80 clean accuracy and 0.60 aggregate
robust accuracy; the environment remains the source of truth for deployment
configuration. These are empirical acceptance cutoffs for fixed finite assets,
not certified-robustness guarantees.

The evaluator checks tensor keys, shapes, values, and upload validity before
inference. Successful and rejected evaluations can be recorded in MLflow with
participant attribution, model artifacts, accuracies, and elapsed time. Logging
is independent of flag eligibility. New validation can be performed with
`private/patchguard/validate_uploads.py`; stored results from earlier validations
are not needed to run the challenge.
