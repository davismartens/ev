import asyncio
import shutil
from pathlib import Path

import typer

from ev.evaluator import PromptEvaluator, EvalConfig
from ev.versioning import load_active_version
from ev.improvement import optimize_prompts
from ev.core.config import settings

app = typer.Typer()

ROOT_DIR = Path(__file__).resolve().parents[1]
EVALS_ROOT = ROOT_DIR / settings.EVALS_ROOT


@app.command()
def create(
    test: str = typer.Argument(..., help="Name of the new test folder under EVALS (e.g. 'test1')."),
):
    test_dir = EVALS_ROOT / test
    if test_dir.exists():
        typer.echo(f"Test '{test}' already exists at {test_dir}")
        raise typer.Exit(code=1)

    cases_dir = test_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    # minimal example case
    (cases_dir / "example.json").write_text(
        '{ "example": true }\n',
        encoding="utf-8",
    )

    # eval criteria
    (test_dir / "eval.md").write_text(
        "# Example criterion\nDescribe pass/fail logic here.\n",
        encoding="utf-8",
    )

    # schema
    (test_dir / "schema.py").write_text(
        "from pydantic import BaseModel\n\n\n"
        "class Response(BaseModel):\n"
        "    # TODO: define expected fields\n"
        "    result: str | None = None\n",
        encoding="utf-8",
    )

    # prompts
    (test_dir / "system_prompt.j2").write_text(
        "You are an assistant that solves the task described in the user prompt.\n",
        encoding="utf-8",
    )
    (test_dir / "user_prompt.j2").write_text(
        "Task description:\n{{ data.<field name> }}\n",
        encoding="utf-8",
    )

    typer.echo(f"Created new test scaffold at {test_dir}")


@app.command()
def delete(
    test: str = typer.Argument(..., help="Name of the test folder under EVALS to delete."),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Delete without confirmation.",
    ),
):
    test_dir = EVALS_ROOT / test
    if not test_dir.exists():
        typer.echo(f"Test '{test}' does not exist at {test_dir}")
        raise typer.Exit(code=1)

    if not yes:
        confirm = typer.confirm(f"Delete test '{test}' and all its contents at {test_dir}?")
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit(code=0)

    shutil.rmtree(test_dir)
    typer.echo(f"Deleted test '{test}' at {test_dir}")


@app.command()
def copy(
    source: str = typer.Argument(..., help="Existing test name to copy from."),
):
    """
    Copy an existing test to <source>_copy.
    """
    src_dir = EVALS_ROOT / source
    if not src_dir.exists():
        typer.echo(f"Source test '{source}' does not exist at {src_dir}")
        raise typer.Exit(code=1)

    dest = f"{source}_copy"
    dst_dir = EVALS_ROOT / dest

    if dst_dir.exists():
        typer.echo(f"Destination test '{dest}' already exists at {dst_dir}")
        raise typer.Exit(code=1)

    shutil.copytree(src_dir, dst_dir)
    typer.echo(f"Copied test '{source}' -> '{dest}' ({src_dir} -> {dst_dir})")



@app.command()
def run(
    test: str = typer.Argument(..., help="Name of the test folder under EVALS (e.g. 'test1')."),
    iterations: int = typer.Option(
        1,
        "--iterations",
        "-i",
        help="Number of self-improvement iterations to run.",
    ),
):
    test_dir = EVALS_ROOT / test
    version_id = load_active_version(test_dir)

    config = EvalConfig(test_name=test, version_id=version_id)
    evaluator = PromptEvaluator(config)

    asyncio.run(optimize_prompts(evaluator, iterations))


@app.command()
def eval(
    test: str = typer.Argument(..., help="Name of the test folder under EVALS (e.g. 'test1')."),
):
    test_dir = EVALS_ROOT / test
    version_id = load_active_version(test_dir)

    config = EvalConfig(test_name=test, version_id=version_id)
    evaluator = PromptEvaluator(config)

    asyncio.run(evaluator.run_all_cases(write_summary=True))


@app.command()
def version(
    test: str = typer.Argument(..., help="Name of the test folder under EVALS (e.g. 'test1')."),
):
    test_dir = EVALS_ROOT / test
    version_id = load_active_version(test_dir)
    print(version_id)


def main():
    app()


if __name__ == "__main__":
    main()
