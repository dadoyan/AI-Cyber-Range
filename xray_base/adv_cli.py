#!/usr/bin/env python3
import argparse
from attack_core import run_eval, run_attack
from helper import cmd_health, cmd_info, cmd_help


def cmd_eval(args):
    run_eval(
        model_name=args.model,
        robust=args.robust,
        smooth_sigma=args.smooth_sigma,
        smooth_samples=args.smooth_samples,
    )


def cmd_attack(args):
    run_attack(
        model_name=args.model,
        attack_type=args.attack,
        eps_str=args.eps,
        step=args.step,
        iterations=args.iterations,
        robust=args.robust,
        smooth_sigma=args.smooth_sigma,
        smooth_samples=args.smooth_samples,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Adversarial Evasion Attack CLI (EfficientNet_B0 model)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Health of image check if models and files are there
    p_health = subparsers.add_parser("health", help="Check environment and files")
    p_health.set_defaults(func=cmd_health)

    # Conceptual explanation about dataset, models, attacks, robustness
    p_info = subparsers.add_parser(
        "info",
        help="Show conceptual info about dataset, models, attacks, and robustness",
    )
    p_info.set_defaults(func=cmd_info)

    # Exercise instructions and suggested command sequence
    p_help = subparsers.add_parser(
        "help",
        help="Show exercise-oriented help and example commands",
    )
    p_help.set_defaults(func=cmd_help)

    # robust options
    def add_robust_args(p):
        p.add_argument(
            "--robust",
            action="store_true",
            help="Enable randomized-smoothing-style inference (averaging noisy copies).",
        )
        p.add_argument(
            "--smooth-sigma",
            type=float,
            default=0.05,
            help="Stddev of Gaussian noise for randomized smoothing in [0,1] space.",
        )
        p.add_argument(
            "--smooth-samples",
            type=int,
            default=16,
            help="Number of noisy samples per input for randomized smoothing.",
        )

    # Evaluation of models
    p_eval = subparsers.add_parser("eval", help="Evaluate a model on clean test set")
    p_eval.add_argument(
        "--model",
        required=True,
        help=(
            "Model file name inside 'models', e.g. "
            "efficientnet_b0.pth or efficientnet_b0_robust.pth"
        ),
    )
    add_robust_args(p_eval)
    p_eval.set_defaults(func=cmd_eval)

    # Attack
    p_attack = subparsers.add_parser(
        "attack",
        help="Run FGSM or PGD attack on a selected model",
    )
    p_attack.add_argument(
        "--model",
        required=True,
        help=(
            "Model file name inside 'models', e.g. "
            "efficientnet_b0.pth or efficientnet_b0_robust.pth"
        ),
    )
    p_attack.add_argument(
        "--attack",
        choices=["FGSM", "PGD"],
        default="FGSM",
        help="Attack type (FGSM or PGD)",
    )
    p_attack.add_argument(
        "--eps",
        default="0.001",
        help="Attack epsilon(s), e.g. '0.01' or '0.001,0.03'",
    )
    p_attack.add_argument(
        "--step",
        type=float,
        default=0.0005,
        help="PGD step size (eps_step). Used only when --attack PGD",
    )
    p_attack.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of PGD iterations (max_iter). Used only when --attack PGD",
    )
    add_robust_args(p_attack)
    p_attack.set_defaults(func=cmd_attack)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
