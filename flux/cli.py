"""`flux-eval` command-line entry point."""

import argparse
import json
from pathlib import Path

from .evaluation import EvalConfig, run_evaluation


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="flux-eval",
        description="Train and evaluate FLUX on a MovieLens dataset "
                    "(ranking quality vs a Euclidean baseline + fairness "
                    "exposure report).")
    p.add_argument("--dataset", default="ml-100k",
                   choices=["ml-100k", "ml-1m"])
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fairness-weight", type=float, default=5.0)
    p.add_argument("--output", type=Path, default=None,
                   help="write full results as JSON to this path")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    config = EvalConfig(
        dataset=args.dataset, dim=args.dim, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, k=args.k, seed=args.seed,
        fairness_weight=args.fairness_weight,
    )
    results = run_evaluation(config, verbose=not args.quiet)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
