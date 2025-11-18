import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from ev.utils.pretty import console, step, substep, success, fail, spinner
from ev.utils.logger import logger
from ev.evaluator import PromptEvaluator, EvalConfig
from ev.versioning import create_version_from_prompts
import difflib
from rich.panel import Panel
from rich.syntax import Syntax
from pathlib import Path


class PromptImprovement(BaseModel):
    notes: str
    new_system_prompt: Optional[str] = None
    new_user_prompt: Optional[str] = None


def _get_case_schema_skeleton(cases_dir: Path) -> Dict[str, Any]:
    case_files = sorted(cases_dir.glob("*.json"))
    if not case_files:
        return {}

    first = json.loads(case_files[0].read_text(encoding="utf-8"))

    skeleton = {}
    for k, v in first.items():
        skeleton[k] = type(v).__name__

    return skeleton


async def _propose_prompt_improvement(
    evaluator: PromptEvaluator,
    current_system_src: str,
    current_user_src: str,
    last_summary: Dict[str, Any],
    change_notes: List[Dict[str, Any]],
) -> PromptImprovement:
    system_prompt = (
        "You improve prompt templates for an automated evaluation harness.\n"
        "You receive:\n"
        "- The current system prompt template\n"
        "- The current user prompt template\n"
        "- A JSON summary of test results (pass_rate, per case results)\n"
        "- A JSON list of previous iterations and prompt changes\n\n"
        "Goal:\n"
        "- Suggest concrete changes to the templates to improve the pass_rate.\n"
        "- Never remove or rename dynamic placeholders like {{variable}} or Jinja blocks.\n"
        "- If you cannot reasonably improve the prompts because data is missing\n"
        "  or the current goal cannot be met, do not change the prompts and explain why.\n"
    )

    schema_skeleton = _get_case_schema_skeleton(evaluator.cases_dir)
    notes_json = json.dumps(change_notes, indent=2)

    user_prompt = (
        "Current system prompt template:\n"
        "<<<SYSTEM_PROMPT_TEMPLATE>>>\n"
        f"{current_system_src}\n"
        "<<<END_SYSTEM_PROMPT_TEMPLATE>>>\n\n"
        "Current user prompt template:\n"
        "<<<USER_PROMPT_TEMPLATE>>>\n"
        f"{current_user_src}\n"
        "<<<END_USER_PROMPT_TEMPLATE>>>\n\n"
        "Last test summary (JSON):\n"
        f"{json.dumps(last_summary, indent=2)}\n\n"
        "Case data schema (first case):\n"
        f"{json.dumps(schema_skeleton, indent=2)}\n\n"
        "Previous iterations and prompt changes (JSON list):\n"
        f"{notes_json}\n\n"
        "Instructions:\n"
        "- You may reference any case field using the exact syntax {{ data.<field> }} or {{ data.<field>.<nested fields> }} for nested fields\n"
        "- Do not invent fields; use only what appears in the schema.\n"
        "- Do not remove or rename placeholders.\n"
    )

    result = await evaluator.runner.generate(
        system_prompts=[system_prompt],
        user_prompts=[user_prompt],
        response_model=PromptImprovement,
        model=evaluator.config.eval_model,
    )
    return result


def diff_strings(old: str, new: str) -> str:
    old_lines = old.splitlines()
    new_lines = new.splitlines()

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile="old",
        tofile="new",
        lineterm="",
    )
    return "\n".join(diff)


async def optimize_prompts(
    evaluator: PromptEvaluator,
    iterations: int,
    cycles: int = 1,
) -> None:
    system_path = evaluator.version_dir / "system_prompt.j2"
    user_path = evaluator.version_dir / "user_prompt.j2"

    base_system_src = system_path.read_text(encoding="utf-8")
    base_user_src = user_path.read_text(encoding="utf-8")

    change_notes: List[Dict[str, Any]] = []

    log_path = evaluator.versions_dir / "log.json"
    if log_path.exists():
        entries = json.loads(log_path.read_text(encoding="utf-8"))
        current_entry = next(
            (e for e in entries if e.get("version") == evaluator.config.version_id),
            None,
        )
        original_pass_rate = (
            current_entry["pass_rate"] if current_entry is not None else 0.0
        )
    else:
        original_pass_rate = 0.0

    step("Initial evaluation")
    substep("restoring current prompts")
    system_path.write_text(base_system_src, encoding="utf-8")
    user_path.write_text(base_user_src, encoding="utf-8")

    baseline_summary = await evaluator.run_all_cases(
        write_summary=False,
        cycles=cycles,
    )

    baseline_pass_rate = baseline_summary["pass_rate"]
    success(f"initial pass_rate {baseline_pass_rate:.3f}")

    best_system_src = base_system_src
    best_user_src = base_user_src
    best_summary: Optional[Dict[str, Any]] = baseline_summary
    best_pass_rate = baseline_pass_rate

    current_system_src = base_system_src
    current_user_src = base_user_src

    for i in range(iterations):
        console.print("")
        step(f"Iteration {i+1}")

        with spinner() as prog:
            prog.add_task("analyzing version and proposing improvements…", total=None)
            improvement = await _propose_prompt_improvement(
                evaluator=evaluator,
                current_system_src=current_system_src,
                current_user_src=current_user_src,
                last_summary=best_summary,
                change_notes=change_notes,
            )

        if not improvement.new_system_prompt and not improvement.new_user_prompt:
            substep("no improvements suggested")
            break

        candidate_system_src = improvement.new_system_prompt or current_system_src
        candidate_user_src = improvement.new_user_prompt or current_user_src

        if (
            candidate_system_src == current_system_src
            and candidate_user_src == current_user_src
        ):
            substep("proposed prompts identical to current; stopping")
            break

        system_path.write_text(candidate_system_src, encoding="utf-8")
        user_path.write_text(candidate_user_src, encoding="utf-8")

        with spinner() as prog:
            prog.add_task("evaluating candidate version…", total=None)
            candidate_summary = await evaluator.run_all_cases(
                write_summary=False,
                cycles=cycles,
            )

        candidate_pass_rate = candidate_summary["pass_rate"]

        if candidate_pass_rate > best_pass_rate:
            success(f"improved {best_pass_rate:.3f} → {candidate_pass_rate:.3f}")
        else:
            fail(f"no gain {candidate_pass_rate:.3f} (best {best_pass_rate:.3f})")

        change_notes.append(
            {
                "iteration": i + 1,
                "prev_pass_rate": best_pass_rate,
                "candidate_pass_rate": candidate_pass_rate,
                "used_as_best": candidate_pass_rate > best_pass_rate,
                "system_changed": candidate_system_src != current_system_src,
                "user_changed": candidate_user_src != current_user_src,
                "system_diff": diff_strings(current_system_src, candidate_system_src)
                if candidate_system_src != current_system_src
                else "",
                "user_diff": diff_strings(current_user_src, candidate_user_src)
                if candidate_user_src != current_user_src
                else "",
            }
        )

        if candidate_pass_rate > best_pass_rate:
            best_pass_rate = candidate_pass_rate
            best_system_src = candidate_system_src
            best_user_src = candidate_user_src
            best_summary = candidate_summary

            if best_pass_rate >= 1.0:
                success("full pass rate achieved (100 percent) — stopping early")
                current_system_src = best_system_src
                current_user_src = best_user_src
                break

        current_system_src = best_system_src
        current_user_src = best_user_src

    if best_pass_rate >= 1.0:
        success("final result: 100 percent pass rate (no further tuning required)")

    system_path.write_text(base_system_src, encoding="utf-8")
    user_path.write_text(base_user_src, encoding="utf-8")

    console.print("")

    if best_system_src == base_system_src and best_user_src == base_user_src:
        substep("no better prompt variant found; stopping")
        return

    if best_pass_rate <= original_pass_rate:
        fail(f"best {best_pass_rate:.3f} did not beat active {original_pass_rate:.3f}")
        return

    step("Creating improved version")

    new_version_id = create_version_from_prompts(
        test_dir=evaluator.test_dir,
        system_src=best_system_src,
        user_src=best_user_src,
        pass_rate=best_pass_rate,
        cycles=cycles,
    )
    success(f"new version {new_version_id} created (pass_rate {best_pass_rate:.3f})")

    if best_system_src != base_system_src:
        diff_text = diff_strings(base_system_src, best_system_src)
        if diff_text:
            syntax = Syntax(
                diff_text,
                "diff",
                theme="ansi_dark",
                line_numbers=False,
                word_wrap=True,
            )
            console.print(Panel(syntax, title="System Prompt Diff", border_style="cyan"))

    if best_user_src != base_user_src:
        diff_text = diff_strings(base_user_src, best_user_src)
        if diff_text:
            syntax = Syntax(
                diff_text,
                "diff",
                theme="ansi_dark",
                line_numbers=False,
                word_wrap=True,
            )
            console.print(Panel(syntax, title="User Prompt Diff", border_style="magenta"))

    final_config = EvalConfig(
        test_name=evaluator.config.test_name,
        version_id=new_version_id,
        generation_model=evaluator.config.generation_model,
        eval_model=evaluator.config.eval_model,
    )
    final_evaluator = PromptEvaluator(final_config)

    console.print("")
    step("Validating new version")
    await final_evaluator.run_all_cases(write_summary=True, cycles=cycles)
