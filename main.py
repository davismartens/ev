import argparse
import asyncio

from ev.evaluator import EvalConfig, PromptEvaluator
from ev.versioning import EVALS_ROOT, load_active_version
from ev.improvement import optimize_prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt eval harness with self-improvement loop."
    )
    parser.add_argument(
        "test_name",
        help="Name of the test folder under EVALS (for example 'test1').",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of self-improvement iterations to run.",
    )
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()

    test_dir = EVALS_ROOT / args.test_name
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    # load active version from log.json
    version_id = load_active_version(test_dir)

    config = EvalConfig(
        test_name=args.test_name,
        version_id=version_id,
    )

    # init the class here
    evaluator = PromptEvaluator(config)

    # pass it into the modular optimizer
    await optimize_prompts(evaluator, iterations=args.iterations)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
