#!/usr/bin/env python3
import os
import sys
import torch
from art import __version__ as art_version
from attack_core import test_data, out_dir


def cmd_health(args):
    print("=== Image Health Check ===")

    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"ART version: {art_version}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device:     {torch.cuda.get_device_name(0)}")

    if os.path.isdir(test_data):
        print(f"[OK] Test data directory exists: {test_data}")
    else:
        print(f"[ERROR] Test data directory not found: {test_data}")

    models_dir = "models"
    expected_models = ["efficientnet_b0.pth", "efficientnet_b0_robust.pth"]
    if not os.path.isdir(models_dir):
        print(f"[ERROR] Models directory not found: {models_dir}")
    else:
        for m in expected_models:
            path = os.path.join(models_dir, m)
            if os.path.isfile(path):
                print(f"[OK] Model file found: {path}")
            else:
                print(f"[WARN] Model file missing: {path}")

    try:
        os.makedirs(out_dir, exist_ok=True)
        test_file = os.path.join(out_dir, ".health_check.tmp")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        print(f"[OK] Writeable output directory: {out_dir}")
    except Exception as e:
        print(f"[ERROR] Cannot write to output directory {out_dir}: {e}")

    print("=== Image Health Check Done ===")


def cmd_info(args):

    print("=== Adversarial Evasion Attack Demo ===\n")

    print("1. Dataset")
    print("     - Type: Chest X-ray images, these are frontal chest radiographs, where each row of the dataset is one patient X-ray image paired with a label indicating their diagnosis")
    print("     - Classes: NORMAL vs PNEUMONIA (This is a binary classification task predicting whether an X-ray belongs to a healthy patient or one with pneumonia)")
    print("     - Dataset Size: 5,216 images for training and 624 for testing")
    print("     - This Docker image only includes the test dataset: labeled-chest-xray-images/chest_xray/test")
    print("     - Training has already been performed beforehand and train data is not inside the container only evaluation on the test data and the attack happen on container")
    print()

    print("2. Models")
    print("     - Docker provides two pre-trained models based on EfficientNet_B0, a lightweight convolutional neural network (CNN)")
    print("     - EfficientNet_B0 is designed to achieve strong accuracy while remaining computationally efficient, making it suitable for medical imaging tasks and for our case adversarial attack demonstrations")
    print("     - efficientnet_b0.pth")
    print("         * Standard model trained on the chest X-ray training data")
    print("         * Good clean accuracy and a strong baseline for evaluating vulnerability against adversarial attacks")
    print("     - efficientnet_b0_robust.pth")
    print("         * Robust version of the standard model trained with added input noise on the dataset (noise-augmented training to support the randomized smoothing inference defense)")
    print("         * Slightly lower clean accuracy ~10%, but more resistant against adversarial attacks")
    print()

    print("3. Attacks")
    print("     - This demo focuses on evasion attacks, where small input changes are crafted to fool a trained model during inference so that it makes wrong predictions")
    print("     - We use white-box attacks, meaning the attacker has full access to the model’s architecture, parameters, and gradients (weights). This allows the attacker to craft adversarial images (also called perturbations or adversarial examples) that directly exploit the model’s weaknesses")
    print("     - Supported attacks on the model:")
    print("         * FGSM (Fast Gradient Sign Method): A single-step white-box attack that uses the gradient of the loss function with respect to the input image. The attacker nudges the image in the direction that increases the model’s loss the most: adv_img = img + eps * sign(∂L/∂img) (think of the sign part as the gradient that tells us how to change the image to increase the model’s loss)")
    print("         * Parameters: The eps parameter of the attack controls the perturbation strength, larger eps means stronger, more visible distortions on the images and typically higher attack success rates")
    print("         * In Layman's terms: FGSM adds carefully chosen noise to the image that pushes the model toward a wrong prediction")
    print()
    print("         * PGD (Projected Gradient Descent): PGD is a multi-step and stronger variant of FGSM. Instead of one update, it applies several small FGSM-like updates to gradually increase the model's loss. The update rule can be written as: adv_img_{t+1} = Project( adv_img_t + step_size * sign(∂L/∂img) ) where the Project(...) function ensures the adversarial image stays within the allowed perturbation limit |adv_img − img|∞ ≤ eps")
    print("         * Parameters: eps (overall maximum allowed distortion), step_size (size of each update) and iterations (how many update steps). More iterations generally create a stronger attack")
    print("         * In Layman's terms: PGD adds small, carefully chosen noise over multiple steps, slowly pushing the image until the model confidently makes a wrong prediction")
    print()
    print("     - To evaluate how succesful our attacks are, we calculate and report the following metrics:")
    print("         * Adversarial accuracy: Fraction of test samples still correctly classified when the attack takes place Accuracy = # of correctly classified samples / total # of test samples")
    print("         * ASR (Attack Success Rate): Among the samples that were originally (when model wasn't under attack) classified correctly, how many are flipped (their prediction changes) by the attack ASR = # of originally correct classified samples that become misclassified after the attack / # of originally correct classified samples ")
    print()

    print("4. Model Robustness and Randomized Smoothing")
    print("     - There are 2 defense mechanisms demonstrated:")
    print("         * Robust training: During training the model sees noisy versions of the inputs, making it more stable against small perturbations used in evasion attacks")
    print("         * Robust inference via randomized smoothing: At inference time, instead of predicting once on an image, the classifier predicts on many noisy copies (img + N(0, sigma^2)) and averages the results")
    print("         * Parameters: The sigma parameter controls how much noise is added, and smooth_samples sets how many noisy copies are averaged. Higher values increase robustness up to a point but make inference slower; too large sigma may harm accuracy a lot")
    print("         * In Layman's terms: if the model stays consistent even when noise is added, it becomes much harder for an attacker to create a small perturbation that flips the final prediction")
    print()

    print("5. What you should observe: ")
    print("     - Clean eval:")
    print("         * efficientnet_b0.pth: Typically has higher clean accuracy")
    print("         * efficientnet_b0_robust.pth: Slightly lower clean accuracy due to noise being added on training")
    print()
    print("     - Under attack (FGSM / PGD):")
    print("         * For the standard model, ASR is very high and accuracy drops sharply")
    print("         * The robust model with robust inference enabled generally achieves low ASR and maintains high accuracy, making the attacks far less effective")
    print()
    print("Use the 'help' command for the suggested sequence of commands for the exercise")
    print()


def cmd_help(args):

    print("=== Help: How to Run the Exercise ===\n")

    print("You generally interact with this image via Docker. Examples below use IMAGE_NAME as a placeholder for the Docker image tag.")
    print("For example if you run 'sudo docker build -t adv-attack-demo' you replace IMAGE_NAME with 'adv-attack-demo'.")
    print("The ipc=host and --shm-size=2g parameters are added on every command run and they mainly boost performance on docker by a little bit.\n")

    print("1. Health check (verify environment, data, and models): 'sudo docker run --rm --ipc=host --shm-size=2g IMAGE_NAME health'\n")
    print("2. Optionally you can use the following command to get more info on the demonstrated models, attacks, metrics but keep in mind its a long read: 'sudo docker run --rm --ipc=host --shm-size=2g IMAGE_NAME info'\n")
    print("3. Evaluate model on clean test data (no attack) to see that the accuracy and other metrics are originally high: 'sudo docker run --rm --ipc=host --shm-size=2g IMAGE_NAME eval --model efficientnet_b0.pth'\n")
    print("4. Run FGSM attack, eps controls the strength of the attack even 'eps=0.001' can have very high ASR and plummet model accuracy: 'sudo docker run --rm --ipc=host --shm-size=2g IMAGE_NAME attack --attack FGSM --eps 0.001 --model efficientnet_b0.pth'\n")
    print("4.1 Optional step: You can change eps to 0.01 or you can pass multiple epsilons, e.g. '--eps 0.001,0.01' to see the difference\n")
    print("5. Run PGD attack, here we have 2 more parameters step size and iterations which also control the strength of the attack: 'sudo docker run --rm --shm-size=2g IMAGE_NAME attack --attack PGD --eps 0.001 --step 0.0005 --iterations --model efficientnet_b0.pth'\n")
    print("6. Run the same attacks on the robust model and observe the differences: 'sudo docker run --rm --ipc=host --shm-size=2g IMAGE_NAME attack --attack FGSM --eps 0.001 --robust --model efficientnet_b0_robust.pth'\n")
    print("7. Questions to think about:")
    print("     - How does accuracy and ASR change between the attacks?")
    print("     - How do ASR and accuracy change as eps increases?")
    print("     - Does the robust model and robust inference actually reduce ASR and maintain higher accuracy compared to the standard model during the attack?")
    print()
