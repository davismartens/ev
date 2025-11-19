import asyncio
import shutil
from pathlib import Path

import typer

from ev.evaluator import PromptEvaluator, EvalConfig
from ev.versioning import load_active_version, EVALS_ROOT
from ev.improvement import optimize_prompts
from ev.core.config import configure_key_source
from ev.agent.runner import AvailableModels
from ev.utils.model_util import resolve_model_config
from ev.utils.pretty import console, step, substep, success, fail


app = typer.Typer(
    help="Prompt eval to stress test agents and with generate robust prompts.",
    add_completion=False,
)


@app.command(help="Name of the new test folder under EVALS (e.g. 'test1')")
def create(
    test: str = typer.Argument(
        ...,
    ),
):
    test_dir = EVALS_ROOT / test
    if test_dir.exists():
        fail(f"Test '{test}' already exists at {test_dir}")
        raise typer.Exit(code=1)

    step(f"Creating new test '{test}'")
    substep(f"root: {EVALS_ROOT}")
    substep(f"path: {test_dir}")

    cases_dir = test_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    (cases_dir / "example.json").write_text(
        '{ "example": true }\n',
        encoding="utf-8",
    )
    substep("added cases/example.json")

    (test_dir / "eval.md").write_text(
        "# Example criterion\nDescribe pass/fail logic here.\n",
        encoding="utf-8",
    )
    substep("added eval.md")

    (test_dir / "schema.py").write_text(
        "from pydantic import BaseModel\n\n\n"
        "class Response(BaseModel):\n"
        "    # TODO: define expected fields\n"
        "    result: str | None = None\n",
        encoding="utf-8",
    )
    substep("added schema.py")

    (test_dir / "system_prompt.j2").write_text(
        "You are an assistant that solves the task described in the user prompt.\n",
        encoding="utf-8",
    )
    (test_dir / "user_prompt.j2").write_text(
        "Task description:\n{{ data.<field name> }}\n",
        encoding="utf-8",
    )
    substep("added system_prompt.j2 and user_prompt.j2")

    success(f"Created new test scaffold at {test_dir}")


@app.command(help="Name of the test folder under 'evals' to delete")
def delete(
    test: str = typer.Argument(
        ...,
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Delete without confirmation.",
    ),
):
    test_dir = EVALS_ROOT / test

    if not test_dir.exists():
        fail(f"Test '{test}' does not exist at {test_dir}")
        raise typer.Exit(code=1)

    step(f"Delete request for test '{test}'")
    substep(f"path: {test_dir}")

    if not yes:
        confirm = typer.confirm(
            f"Delete test '{test}' and ALL contents at {test_dir}?"
        )
        if not confirm:
            fail("Aborted")
            raise typer.Exit(code=0)

    substep("removing directory…")
    shutil.rmtree(test_dir)

    success(f"Deleted test '{test}' at {test_dir}")


@app.command(help="Existing test name to copy from")
def copy(
    source: str = typer.Argument(
        ...,
    ),
):
    src_dir = EVALS_ROOT / source
    if not src_dir.exists():
        fail(f"Source test '{source}' does not exist at {src_dir}")
        raise typer.Exit(code=1)

    dest = f"{source}_copy"
    dst_dir = EVALS_ROOT / dest

    if dst_dir.exists():
        fail(f"Destination test '{dest}' already exists at {dst_dir}")
        raise typer.Exit(code=1)

    step(f"Copying test '{source}'")
    substep(f"from: {src_dir}")
    substep(f"to:   {dst_dir}")

    shutil.copytree(src_dir, dst_dir)

    success(f"Copied '{source}' → '{dest}'")


def _resolve_models(
    model: str | None,
    gen_model: str | None,
    eval_model: str | None,
):
    generation_model = AvailableModels.groq.kimi_k2_instruct
    eval_model_cfg = AvailableModels.groq.kimi_k2_instruct

    if model is not None:
        shared_cfg = resolve_model_config(model)
        generation_model = shared_cfg
        eval_model_cfg = shared_cfg

    if gen_model is not None:
        generation_model = resolve_model_config(gen_model)

    if eval_model is not None:
        eval_model_cfg = resolve_model_config(eval_model)

    return generation_model, eval_model_cfg


@app.command(help="Name of the test folder under 'evals' (e.g. 'myAgent')")
def run(
    test: str = typer.Argument(
        ...,
    ),
    iterations: int = typer.Option(
        1,
        "--iterations",
        "-i",
        help="Number of self-improvement iterations to run.",
    ),
    cycles: int = typer.Option(
        1,
        "--cycles",
        "-c",
        help="Number of evaluation cycles per case for stress testing.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help=(
            "Model for both generation and eval, in the form provider[identifier]. "
            "Examples: 'openai[gpt-5]', 'openai[gpt5_nano]', "
            "'groq[kimi_k2_instruct]', 'groq[moonshotai/kimi-k2-instruct]', "
            "'groq[openai/gpt-oss-120b]'."
        ),
    ),
    gen_model: str | None = typer.Option(
        None,
        "--gen-model",
        help=(
            "Override generation model only. "
            "If both --model and --gen-model are set, --gen-model wins."
        ),
    ),
    eval_model: str | None = typer.Option(
        None,
        "--eval-model",
        help=(
            "Override eval model only. "
            "If both --model and --eval-model are set, --eval-model wins."
        ),
    ),
    key: str = typer.Option(
        "file",
        "--key",
        "-k",
        help="Where to load API keys from: 'file' (.env, default) or 'env'.",
    ),
):
    # key source selection
    try:
        configure_key_source(key)
    except ValueError as e:
        fail(str(e))
        raise typer.Exit(code=1)

    # ensure test directory exists
    test_dir = EVALS_ROOT / test
    if not test_dir.exists():
        fail(f"Test '{test}' not found at {test_dir}")
        raise typer.Exit(code=1)

    step(f"Running eval on '{test}'")
    substep(f"test path: {test_dir}")
    substep(f"iterations: {iterations}")
    substep(f"cycles: {cycles}")
    substep(f"key source: {key}")

    # model resolution
    generation_model, eval_model_cfg = _resolve_models(model, gen_model, eval_model)
    substep(f"generation model: {generation_model.name}")
    substep(f"eval model:       {eval_model_cfg.name}")

    # load version
    version_id = load_active_version(test_dir)
    substep(f"active version: {version_id}")

    # evaluator config
    config = EvalConfig(
        test_name=test,
        version_id=version_id,
        generation_model=generation_model,
        eval_model=eval_model_cfg,
    )
    evaluator = PromptEvaluator(config)

    substep("starting optimization loop…")

    # run optimizer
    asyncio.run(
        optimize_prompts(
            evaluator=evaluator,
            iterations=iterations,
            cycles=max(1, cycles),
        )
    )

    success("Run completed")



@app.command()
def eval(
    test: str = typer.Argument(
        ...,
        help="Name of the test folder under EVALS (e.g. 'test1').",
    ),
    cycles: int = typer.Option(
        1,
        "--cycles",
        "-c",
        help="Number of evaluation cycles per case for stress testing.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help=(
            "Model for both generation and eval, in the form provider[identifier]. "
            "If omitted, defaults to AvailableModels.groq.kimi_k2_instruct."
        ),
    ),
    gen_model: str | None = typer.Option(
        None,
        "--gen-model",
        help=(
            "Override generation model only. "
            "If both --model and --gen-model are set, --gen-model wins."
        ),
    ),
    eval_model: str | None = typer.Option(
        None,
        "--eval-model",
        help=(
            "Override eval model only. "
            "If both --model and --eval-model are set, --eval-model wins."
        ),
    ),
    key: str = typer.Option(
        "file",
        "--key",
        "-k",
        help="Where to load API keys from: 'file' (.env, default) or 'env'.",
    ),
):
    try:
        configure_key_source(key)
    except ValueError as e:
        fail(str(e))
        raise typer.Exit(code=1)

    test_dir = EVALS_ROOT / test
    if not test_dir.exists():
        fail(f"Test '{test}' not found at {test_dir}")
        raise typer.Exit(code=1)

    step(f"Running eval for '{test}'")
    substep(f"test path: {test_dir}")
    substep(f"cycles: {cycles}")
    substep(f"key source: {key}")

    generation_model, eval_model_cfg = _resolve_models(model, gen_model, eval_model)
    substep(f"generation model: {generation_model.name}")
    substep(f"eval model:       {eval_model_cfg.name}")

    version_id = load_active_version(test_dir)
    substep(f"active version: {version_id}")

    config = EvalConfig(
        test_name=test,
        version_id=version_id,
        generation_model=generation_model,
        eval_model=eval_model_cfg,
    )
    evaluator = PromptEvaluator(config)

    substep("starting evaluation…")

    asyncio.run(
        evaluator.run_all_cases(
            write_summary=True,
            cycles=max(1, cycles),
        )
    )

    success("Eval completed")



@app.command(help="Name of the test folder under 'evals' (e.g. 'myAgent')")
def version(
    test: str = typer.Argument(
        ...,
    ),
):
    test_dir = EVALS_ROOT / test
    if not test_dir.exists():
        fail(f"Test '{test}' not found at {test_dir}")
        raise typer.Exit(code=1)

    step(f"Fetching active version for '{test}'")
    substep(f"path: {test_dir}")

    version_id = load_active_version(test_dir)

    success(f"Active version: {version_id}")


@app.command(help="List tests inside 'evals'.")
def list():
    step("Available tests")
    for p in EVALS_ROOT.iterdir():
        if p.is_dir():
            substep(p.name)


def main():
    app()


if __name__ == "__main__":
    main()
