#!/usr/bin/env python3
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.utils import save_image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from tqdm import tqdm
import contextlib
import io


# Paths and constants
test_data = "labeled-chest-xray-images/chest_xray/test"
out_dir = "adversarial-examples"
adv_img_dir = os.path.join(out_dir, "adv-images")

IMG_SIZE = 256
BATCH_SIZE = 16
NUM_WORKERS = 1

os.environ.setdefault("GLOG_minloglevel", "2") # hide INFO and WARNING from glog
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR") # for torch C++ logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Preprocessing normalization but applied inside model so ART sees inputs in [0,1]
class NormalizeLayer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean = torch.tensor(mean).view(3, 1, 1)
        std = torch.tensor(std).view(3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return (x - self.mean) / self.std


# Build EfficientNet_B0 model architecture (if model changes, this must change)
def build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# Preprocessing convert DataLoader -> (X_np, y_np) for ART (inputs in [0,1])
def numpy_from_loader(loader):
    Xs, Ys = [], []
    for xb, yb in loader:
        Xs.append(xb.numpy())
        Ys.append(yb.numpy())
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(Ys, axis=0)
    return X.astype(np.float32), y.astype(np.int64)


# Randomized smoothing helper
# If robust is disabled, just call classifier.predict once.
# If robust is enabled, it performs randomized smoothing-style inference:
#   - sample n_samples Gaussian noises
#   - predict on X + noise
#   - average probabilities
def smoothed_predict(
    classifier,
    X,
    batch_size: int,
    robust: bool = False,
    sigma: float = 0.05,
    n_samples: int = 16,
):
    if not robust:
        # Original behavior, unchanged
        return classifier.predict(X, batch_size=batch_size)

    N = X.shape[0]
    num_classes = classifier.nb_classes
    proba_sum = np.zeros((N, num_classes), dtype=np.float32)

    for _ in range(n_samples):
        noise = np.random.normal(loc=0.0, scale=sigma, size=X.shape).astype(
            np.float32
        )
        X_noisy = np.clip(X + noise, 0.0, 1.0)
        proba_sum += classifier.predict(X_noisy, batch_size=batch_size)

    return proba_sum / n_samples


# Preprocessing build test DataLoader
def build_test_loader():
    img_size = IMG_SIZE
    test_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    print("Loading test set from:", test_data)
    test_ds = datasets.ImageFolder(root=test_data, transform=test_transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    class_names = test_ds.classes
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Num test samples:", len(test_ds))
    return test_loader, class_names, num_classes


# Preprocessing build EfficientNet and ART wrapper
def build_classifier(model_name: str, num_classes: int):
    img_size = IMG_SIZE

    model_path = os.path.join("models", model_name)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print("Building EfficientNet_B0 and loading weights from:", model_path)

    base_model = build_efficientnet_b0(num_classes=num_classes)
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
    base_model.load_state_dict(state)
    base_model.to(device)
    base_model.eval()

    norm_layer = NormalizeLayer(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ).to(device)

    full_model = nn.Sequential(norm_layer, base_model)
    full_model.eval()
    print("Model and normalization ready on", device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(full_model.parameters(), lr=1e-3)

    classifier = PyTorchClassifier(
        model=full_model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, img_size, img_size),
        nb_classes=num_classes,
        clip_values=(0.0, 1.0),
    )
    return classifier


# Clean evaluation of the models using the eval command
def clean_eval(
    classifier: PyTorchClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names,
    robust: bool,
    smooth_sigma: float,
    smooth_samples: int,
):
    print("\n[Clean evaluation] Predicting on clean test set...")
    attack_batch_size = BATCH_SIZE

    preds_clean_proba = smoothed_predict(
        classifier,
        X_test,
        batch_size=attack_batch_size,
        robust=robust,
        sigma=smooth_sigma,
        n_samples=smooth_samples,
    )
    preds_clean = np.argmax(preds_clean_proba, axis=1)

    acc = (preds_clean == y_test).mean()
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds_clean, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_test, preds_clean)

    print("\n=== CLEAN TEST METRICS (EfficientNet_B0) ===")
    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall:    {rec*100:.2f}%")
    print(f"F1:        {f1*100:.2f}%")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print("\nCounts: TN FP\n        FN TP")
    print(f"TN={tn}  FP={fp}\nFN={fn}  TP={tp}\n")

    print("Classification report:")
    print(classification_report(y_test, preds_clean, target_names=class_names))

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "preds_clean.npy"), preds_clean)
    if len(class_names) == 2:
        probs_clean = preds_clean_proba[:, 1]
        np.save(os.path.join(out_dir, "probs_clean.npy"), probs_clean)
    final_dir = os.path.abspath(out_dir)
    print(f"Saved preds_clean.npy and probs_clean.npy to: {final_dir}")

    correct_mask = (preds_clean == y_test)
    n_correct = int(correct_mask.sum())
    print(f"Originally-correct samples: {n_correct} / {len(y_test)}")

    return preds_clean, correct_mask, n_correct

# FGSM / PGD attacks
def run_fgsm_attack(
    classifier,
    X_test,
    y_test,
    epsilons,
    attack_batch_size,
    num_classes,
    robust,
    smooth_sigma,
    smooth_samples,
):
    attack_name = "fgsm"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(adv_img_dir, exist_ok=True)

    for eps in epsilons:
        print(f"\n--- FGSM attack (eps = {eps:g}) ---")

        # Step 1 generate adversarial examples
        print("[1/3] Generating adversarial examples...")
        attack = FastGradientMethod(estimator=classifier, eps=eps)

        t0 = time.time()
        X_adv_batches = []
        for i in tqdm(
            range(0, len(X_test), attack_batch_size),
            desc="Progress",
        ):
            batch = X_test[i : i + attack_batch_size]
            X_adv_batch = attack.generate(batch)
            X_adv_batches.append(X_adv_batch)
        X_adv = np.concatenate(X_adv_batches, axis=0)
        t1 = time.time()
        tqdm.write(f"Generated adversarial examples in {t1 - t0:.1f}s")

        out_x = os.path.join(out_dir, f"x_adv_fgsm_eps{eps:.3f}.npy")
        np.save(out_x, X_adv)

        # Step 2 inference on adversarial examples
        print("[2/3] Running model inference on adversarial examples...")
        adv_proba_batches = []
        t2 = time.time()
        for i in tqdm(
            range(0, len(X_adv), attack_batch_size),
            desc="Progress",
        ):
            batch_adv = X_adv[i : i + attack_batch_size]
            proba_batch = smoothed_predict(
                classifier,
                batch_adv,
                batch_size=attack_batch_size,
                robust=robust,
                sigma=smooth_sigma,
                n_samples=smooth_samples,
            )
            adv_proba_batches.append(proba_batch)
        preds_adv_proba = np.concatenate(adv_proba_batches, axis=0)
        preds_adv = np.argmax(preds_adv_proba, axis=1)
        t3 = time.time()
        print(f"Finished model inference in {t3 - t2:.1f}s")

        # Step 3 compute metrics such as accuracy and ASR
        print("[3/3] Calculating adversarial accuracy and ASR...")
        clean_proba_batches = []
        for i in tqdm(
            range(0, len(X_test), attack_batch_size),
            desc="Progress",
        ):
            batch_clean = X_test[i : i + attack_batch_size]
            proba_batch = smoothed_predict(
                classifier,
                batch_clean,
                batch_size=attack_batch_size,
                robust=robust,
                sigma=smooth_sigma,
                n_samples=smooth_samples,
            )
            clean_proba_batches.append(proba_batch)
        preds_clean_proba = np.concatenate(clean_proba_batches, axis=0)
        preds_clean = np.argmax(preds_clean_proba, axis=1)

        correct_mask = (preds_clean == y_test)
        n_correct = int(correct_mask.sum())

        adv_correct = int((preds_adv == y_test).sum())
        adv_acc = adv_correct / len(y_test)

        if n_correct > 0:
            flipped_mask = np.logical_and(correct_mask, preds_adv != preds_clean)
            n_flipped = int(flipped_mask.sum())
            asr = n_flipped / n_correct
        else:
            flipped_mask = np.zeros_like(y_test, dtype=bool)
            n_flipped = 0
            asr = float("nan")

        print(
            f"\nAdv accuracy: {adv_acc*100:.2f}%  "
            f"(model was correct on {adv_correct}/{len(y_test)} adversarial predictions)"
        )
        if n_correct > 0:
            print(
                f"ASR: {asr*100:.2f}%  "
                f"({n_flipped}/{n_correct} originally correct predictions were flipped by the attack)"
            )
        else:
            print(
                "ASR: undefined (no samples were correctly classified on the clean data, "
                "so there is no originally-correct subset to flip)."
            )

        np.save(
            os.path.join(out_dir, f"preds_adv_fgsm_eps{eps:.3f}.npy"),
            preds_adv,
        )
        if num_classes == 2:
            probs_adv = preds_adv_proba[:, 1]
            np.save(
                os.path.join(out_dir, f"probs_adv_fgsm_eps{eps:.3f}.npy"),
                probs_adv,
            )

        eps_tag = f"{eps:.4f}"
        flipped_indices = np.where(flipped_mask)[0]
        if len(flipped_indices) > 0:
            k = min(10, len(flipped_indices))
            chosen = np.random.choice(flipped_indices, size=k, replace=False)

            for idx in chosen:
                clean_tensor = torch.from_numpy(X_test[idx])
                adv_tensor = torch.from_numpy(X_adv[idx])

                pair = torch.cat([clean_tensor, adv_tensor], dim=2)

                img_filename = f"{attack_name}_eps{eps_tag}_idx{idx:04d}.png"
                save_path = os.path.join(adv_img_dir, img_filename)
                save_image(pair, save_path)

            print()
            print(f"Saved {k} adversarial images for eps={eps_tag}")
        else:
            print()
            print(f"No flipped samples, so no adversarial images saved for eps={eps_tag}")


def run_pgd_attack(
    classifier,
    X_test,
    y_test,
    epsilons,
    attack_batch_size,
    num_classes,
    step,
    iterations,
    robust,
    smooth_sigma,
    smooth_samples,
):
    attack_name = "pgd"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(adv_img_dir, exist_ok=True)

    for eps in epsilons:
        print(
            f"\n--- PGD attack (eps = {eps:g}, step = {step:g}, iters = {iterations}) ---"
        )

        # Step 1 generate adversarial examples
        print("[1/3] Generating adversarial examples...")
        attack = ProjectedGradientDescent(
            estimator=classifier,
            norm=np.inf,
            eps=eps,
            eps_step=step,
            max_iter=iterations,
            targeted=False,
            num_random_init=1,
            batch_size=attack_batch_size,
        )

        t0 = time.time()
        X_adv_batches = []
        for i in tqdm(
            range(0, len(X_test), attack_batch_size),
            desc="Progress",
        ):
            batch = X_test[i : i + attack_batch_size]
            X_adv_batch = attack.generate(batch)
            X_adv_batches.append(X_adv_batch)
        X_adv = np.concatenate(X_adv_batches, axis=0)
        t1 = time.time()
        tqdm.write(f"Generated adversarial examples in {t1 - t0:.1f}s")

        out_x = os.path.join(out_dir, f"x_adv_pgd_eps{eps:.3f}.npy")
        np.save(out_x, X_adv)

        # Step 2 inference on adversarial examples
        print("[2/3] Running model inference on adversarial examples...")
        adv_proba_batches = []
        t2 = time.time()
        for i in tqdm(
            range(0, len(X_adv), attack_batch_size),
            desc="Progress",
        ):
            batch_adv = X_adv[i : i + attack_batch_size]
            proba_batch = smoothed_predict(
                classifier,
                batch_adv,
                batch_size=attack_batch_size,
                robust=robust,
                sigma=smooth_sigma,
                n_samples=smooth_samples,
            )
            adv_proba_batches.append(proba_batch)
        preds_adv_proba = np.concatenate(adv_proba_batches, axis=0)
        preds_adv = np.argmax(preds_adv_proba, axis=1)
        t3 = time.time()
        print(f"Finished model inference in {t3 - t2:.1f}s")

        # Step 3 compute metrics such as accuracy and ASR
        print("[3/3] Calculating adversarial accuracy and ASR...")
        clean_proba_batches = []
        for i in tqdm(
            range(0, len(X_test), attack_batch_size),
            desc="Progress",
        ):
            batch_clean = X_test[i : i + attack_batch_size]
            proba_batch = smoothed_predict(
                classifier,
                batch_clean,
                batch_size=attack_batch_size,
                robust=robust,
                sigma=smooth_sigma,
                n_samples=smooth_samples,
            )
            clean_proba_batches.append(proba_batch)
        preds_clean_proba = np.concatenate(clean_proba_batches, axis=0)
        preds_clean = np.argmax(preds_clean_proba, axis=1)

        correct_mask = (preds_clean == y_test)
        n_correct = int(correct_mask.sum())

        adv_correct = int((preds_adv == y_test).sum())
        adv_acc = adv_correct / len(y_test)

        if n_correct > 0:
            flipped_mask = np.logical_and(correct_mask, preds_adv != preds_clean)
            n_flipped = int(flipped_mask.sum())
            asr = n_flipped / n_correct
        else:
            flipped_mask = np.zeros_like(y_test, dtype=bool)
            n_flipped = 0
            asr = float("nan")

        print(
            f"\nAdv accuracy: {adv_acc*100:.2f}%  "
            f"(model was correct on {adv_correct}/{len(y_test)} predictions made on the adversarial examples)"
        )
        if n_correct > 0:
            print(
                f"ASR: {asr*100:.2f}%  "
                f"({n_flipped}/{n_correct} originally correct model predictions were flipped by the attack)"
            )
        else:
            print(
                "ASR: undefined (no samples were correctly classified on the clean data, "
                "so there is no originally-correct subset to flip)."
            )

        np.save(
            os.path.join(out_dir, f"preds_adv_pgd_eps{eps:.3f}.npy"),
            preds_adv,
        )
        if num_classes == 2:
            probs_adv = preds_adv_proba[:, 1]
            np.save(
                os.path.join(out_dir, f"probs_adv_pgd_eps{eps:.3f}.npy"),
                probs_adv,
            )

        eps_tag = f"{eps:.4f}"
        flipped_indices = np.where(flipped_mask)[0]
        if len(flipped_indices) > 0:
            k = min(10, len(flipped_indices))
            chosen = np.random.choice(flipped_indices, size=k, replace=False)

            for idx in chosen:
                clean_tensor = torch.from_numpy(X_test[idx])
                adv_tensor = torch.from_numpy(X_adv[idx])

                pair = torch.cat([clean_tensor, adv_tensor], dim=2)
                img_filename = f"{attack_name}_eps{eps_tag}_idx{idx:04d}.png"
                save_path = os.path.join(adv_img_dir, img_filename)
                save_image(pair, save_path)

            print()
            print(f"Saved {k} adversarial images for eps={eps_tag}")
        else:
            print()
            print(f"No flipped samples, so no adversarial images saved for eps={eps_tag}")

# Public API for adv_cli.py
def run_eval(
    model_name: str,
    robust: bool,
    smooth_sigma: float,
    smooth_samples: int,
):
    print("Using device:", device)
    test_loader, class_names, num_classes = build_test_loader()
    X_test, y_test = numpy_from_loader(test_loader)
    print(
        f"X_test shape: {X_test.shape}, dtype={X_test.dtype}, "
        f"min/max: {X_test.min():.4f}/{X_test.max():.4f}"
    )

    classifier = build_classifier(model_name, num_classes)

    if robust:
        print(
            f"Robust inference enabled: sigma={smooth_sigma}, "
            f"samples={smooth_samples}"
        )

    clean_eval(
        classifier,
        X_test,
        y_test,
        class_names,
        robust=robust,
        smooth_sigma=smooth_sigma,
        smooth_samples=smooth_samples,
    )


def run_attack(
    model_name: str,
    attack_type: str,
    eps_str: str,
    step: float,
    iterations: int,
    robust: bool,
    smooth_sigma: float,
    smooth_samples: int,
):
    print("Using device:", device)

    with contextlib.redirect_stdout(io.StringIO()):
        test_loader, class_names, num_classes = build_test_loader()
        X_test, y_test = numpy_from_loader(test_loader)
        classifier = build_classifier(model_name, num_classes)

    print("ART PyTorchClassifier ready")

    if robust:
        print(
            f"Robust inference enabled: sigma={smooth_sigma}, "
            f"samples={smooth_samples}"
        )

    epsilons = [float(x) for x in eps_str.split(",") if x.strip() != ""]
    attack_batch_size = BATCH_SIZE

    if attack_type.upper() == "FGSM":
        run_fgsm_attack(
            classifier=classifier,
            X_test=X_test,
            y_test=y_test,
            epsilons=epsilons,
            attack_batch_size=attack_batch_size,
            num_classes=num_classes,
            robust=robust,
            smooth_sigma=smooth_sigma,
            smooth_samples=smooth_samples,
        )
    else:
        run_pgd_attack(
            classifier=classifier,
            X_test=X_test,
            y_test=y_test,
            epsilons=epsilons,
            attack_batch_size=attack_batch_size,
            num_classes=num_classes,
            step=step,
            iterations=iterations,
            robust=robust,
            smooth_sigma=smooth_sigma,
            smooth_samples=smooth_samples,
        )

    final_dir = os.path.abspath(out_dir)
    tqdm.write(f"All artifacts saved to: {final_dir}")
