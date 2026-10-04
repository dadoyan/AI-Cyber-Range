# MNIST subset

This directory contains processed tensors from the MNIST handwritten-digits
dataset distributed by `torchvision.datasets.MNIST`. It contains only the
first 3,000 official training examples and first 500 public test examples for
the participant environment. The separate held-out evaluation examples are
stored under `private/patchguard/` and are not copied into participant images.

The participant notebook loads these files with `download=False`, so the
exercise works after the Docker images have been built without internet access.
