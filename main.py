import argparse
import asyncio

from ev.evaluator import EvalConfig, PromptEvaluator
from ev.versioning import EVALS_ROOT, load_active_version
from ev.improvement import optimize_prompts
from ev.agent.runner import AvailableModels
from ev.utils.model_util import resolve_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt eval harness with self-improvement loop."
    )
    parser.add_argument(
        "test_name",
        help="Name of the test folder under EVALS (for example 'test1').",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=5,
        help="Number of self-improvement iterations to run.",
    )
    parser.add_argument(
        "-c",
        "--cycles",
        type=int,
        default=1,
        help="Number of evaluation cycles per case for stress testing.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help=(
            "Model for both generation and eval, in the form provider[identifier]. "
            "Examples: 'openai[gpt-5]', 'openai[gpt5_nano]', "
            "'groq[kimi_k2_instruct]', 'groq[moonshotai/kimi-k2-instruct]', "
            "'groq[openai/gpt-oss-120b]'. "
            "If omitted, defaults to AvailableModels.groq.kimi_k2_instruct."
        ),
    )
    parser.add_argument(
        "--gen-model",
        type=str,
        default=None,
        help=(
            "Override generation model only, same format as --model. "
            "If both --model and --gen-model are set, --gen-model wins for generation."
        ),
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=None,
        help=(
            "Override eval model only, same format as --model. "
            "If both --model and --eval-model are set, --eval-model wins for eval."
        ),
    )
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()

    test_dir = EVALS_ROOT / args.test_name
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    version_id = load_active_version(test_dir)

    # default: exactly old behaviour
    generation_model = AvailableModels.groq.kimi_k2_instruct
    eval_model = AvailableModels.groq.kimi_k2_instruct

    # shared override
    if args.model is not None:
        shared_cfg = resolve_model_config(args.model)
        generation_model = shared_cfg
        eval_model = shared_cfg

    # specific overrides
    if args.gen_model is not None:
        generation_model = resolve_model_config(args.gen_model)

    if args.eval_model is not None:
        eval_model = resolve_model_config(args.eval_model)

    config = EvalConfig(
        test_name=args.test_name,
        version_id=version_id,
        generation_model=generation_model,
        eval_model=eval_model,
    )

    evaluator = PromptEvaluator(config)

    await optimize_prompts(
        evaluator=evaluator,
        iterations=args.iterations,
        cycles=max(1, args.cycles),
    )


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
