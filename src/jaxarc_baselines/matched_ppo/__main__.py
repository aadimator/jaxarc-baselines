"""Command line interface for matched PPO runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .run import evaluate_checkpoint, run_training
from .summary import summarize_results


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("jaxarc", "arcle", "arcle-async"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--rollout-length", type=int)
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument(
        "--policy-architecture",
        choices=("factorized", "spatial"),
    )
    parser.add_argument("--num-minibatches", type=int)
    parser.add_argument("--output-dir", type=Path)


def _resolved_config(args: argparse.Namespace):
    return load_config(args.config).with_overrides(
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        rollout_length=args.rollout_length,
        eval_episodes=args.eval_episodes,
        policy_architecture=args.policy_architecture,
        num_minibatches=args.num_minibatches,
        evaluate_after_training=(
            False if getattr(args, "skip_evaluation", False) else None
        ),
        output_dir=args.output_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Matched end-to-end PPO benchmark: JaxARC versus ARCLE."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    _add_config_arguments(train)
    train.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="omit post-training evaluation from timing-only runs",
    )

    evaluate = subparsers.add_parser("evaluate")
    _add_config_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "summarize":
        summarize_results(args.output_dir.resolve())
        return

    config = _resolved_config(args)
    if args.command == "train":
        run_training(config, args.backend, args.seed)
        return
    evaluate_checkpoint(
        config,
        args.backend,
        args.seed,
        args.checkpoint.resolve(),
        output_path=(args.output.resolve() if args.output else None),
    )


if __name__ == "__main__":
    main()
