import json
import re
import sys
import importlib.util
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from ev.utils.pretty import console, step, substep, success, fail, spinner

from ev.agent.runner import Runner, ModelConfig, AvailableModels
from ev.agent.composer import Composer
from ev.utils.logger import logger
from ev.versioning import EVALS_ROOT


class CriteriaResult(BaseModel):
    criteria_name: str
    criteria_passed: bool


class EvalOut(BaseModel):
    name: str
    objectives: List[CriteriaResult]
    max_iterations: Optional[int] = None


@dataclass
class EvalConfig:
    test_name: str
    version_id: str
    generation_model: ModelConfig = AvailableModels.groq.kimi_k2_instruct
    eval_model: ModelConfig = AvailableModels.groq.kimi_k2_instruct


class PromptEvaluator:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.test_dir = EVALS_ROOT / config.test_name
        self.cases_dir = self.test_dir / "cases"
        self.versions_dir = self.test_dir / "versions"
        self.version_dir = self.versions_dir / config.version_id

        if not self.version_dir.exists():
            raise FileNotFoundError(f"Version dir not found: {self.version_dir}")

        if not self.cases_dir.exists():
            raise FileNotFoundError(f"Missing cases/ directory in {self.test_dir}")

        schema_path = self.test_dir / "schema.py"
        if not schema_path.exists():
            raise FileNotFoundError(f"Missing schema.py in {self.test_dir}")

        spec = importlib.util.spec_from_file_location(
            f"{self.config.test_name}_schema",
            schema_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load schema module from {schema_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        response_model = None
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel:
                response_model = obj
                break

        if response_model is None:
            raise RuntimeError("No valid Pydantic BaseModel found in schema.py")

        self.response_model = response_model
        self.runner = Runner()

    def _render_prompts(self, case_data: Dict[str, Any]) -> Dict[str, str]:
        data: Dict[str, Any] = {"data": case_data}

        system_prompt = Composer._load_template(
            "system_prompt",
            sub_dir=str(self.version_dir),
            **data,
        )

        user_prompt = Composer._load_template(
            "user_prompt",
            sub_dir=str(self.version_dir),
            **data,
        )

        return {
            "system": system_prompt,
            "user": user_prompt,
        }

    async def _call_generation(self, system_prompt: str, user_prompt: str) -> Any:
        console.log(
            f"[dim][gen][/dim] using model [bold]{self.config.generation_model.name}[/bold] "
            f"via [cyan]{self.config.generation_model.provider}[/cyan]"
        )

        result = await self.runner.generate(
            user_prompts=[user_prompt],
            system_prompts=[system_prompt],
            response_model=self.response_model,
            model=self.config.generation_model,
        )
        return result

    async def run_all_cases(self, write_summary: bool = True, cycles: int = 1) -> Dict[str, Any]:
        case_files = sorted(self.cases_dir.glob("*.json"))
        if not case_files:
            warn_msg = f"No case JSON files found in {self.cases_dir}"
            console.print(f"[yellow]{warn_msg}[/yellow]")
            return {
                "version": self.config.version_id,
                "total_cases": 0,
                "passed_cases": 0,
                "pass_rate": 0.0,
                "cases": [],
            }

        total_cycles = max(1, cycles)

        step(f"Running {len(case_files)} cases")
        substep(f"test: {self.config.test_name}")
        substep(f"version: {self.config.version_id}")
        if total_cycles > 1:
            substep(f"cycles per case: {total_cycles}")
        console.print("")

        summary: Dict[str, Any] = {
            "version": self.config.version_id,
            "total_cases": len(case_files),
            "passed_cases": 0,
            "pass_rate": 0.0,
            "cases": [],
        }

        for case_file in case_files:
            case_name = case_file.stem
            step(f"Case {case_name}")

            case_data = json.loads(case_file.read_text(encoding="utf-8"))

            substep("rendering prompts")
            prompts = self._render_prompts(case_data)

            objective_names: Optional[List[str]] = None
            objective_pass_counts: Optional[List[int]] = None
            case_full_pass_cycles = 0

            for cycle_idx in range(total_cycles):
                if total_cycles > 1:
                    substep(f"cycle {cycle_idx + 1}/{total_cycles}")

                with spinner() as prog:
                    prog.add_task("generating model output…", total=None)
                    output_data = await self._call_generation(
                        prompts["system"],
                        prompts["user"],
                    )

                with spinner() as prog:
                    prog.add_task("running evaluation…", total=None)
                    eval_out = await self._call_eval(
                        case_name=case_name,
                        case_data=case_data,
                        output_data=output_data,
                        original_task=prompts["user"],
                    )

                if not eval_out.objectives:
                    continue

                if objective_names is None:
                    objective_names = [obj.criteria_name for obj in eval_out.objectives]
                    objective_pass_counts = [0] * len(objective_names)

                all_passed_this_cycle = True

                for idx, obj in enumerate(eval_out.objectives):
                    if obj.criteria_passed:
                        objective_pass_counts[idx] += 1
                    else:
                        all_passed_this_cycle = False

                if all_passed_this_cycle:
                    case_full_pass_cycles += 1

            if not objective_names or objective_pass_counts is None:
                objectives_list: List[Dict[str, float]] = []
                case_pass_rate = 0.0
                passed_criteria_count = 0
                total_criteria_count = 0
            else:
                objective_rates: List[float] = [
                    count / float(total_cycles) for count in objective_pass_counts
                ]
                objectives_list = []
                for name, rate in zip(objective_names, objective_rates):
                    short_name = name[:20]
                    objectives_list.append({short_name: rate})

                case_pass_rate = sum(objective_rates) / len(objective_rates)
                passed_criteria_count = sum(1 for r in objective_rates if r >= 1.0)
                total_criteria_count = len(objective_rates)

            case_block = {
                "case_name": case_name,
                "objectives": objectives_list,
                "pass_rate": case_pass_rate,
            }

            if total_cycles == 1:
                if total_criteria_count > 0 and passed_criteria_count == total_criteria_count:
                    summary["passed_cases"] += 1
                    success(
                        f"case {case_name} passed {passed_criteria_count}/{total_criteria_count}"
                    )
                else:
                    fail(
                        f"case {case_name} passed {passed_criteria_count}/{total_criteria_count}"
                    )
            else:
                if objective_names and case_full_pass_cycles == total_cycles:
                    summary["passed_cases"] += 1
                    success(
                        f"case {case_name} stable across {total_cycles} cycles "
                        f"({total_criteria_count} criteria)"
                    )
                else:
                    fail(
                        f"case {case_name} unstable across {total_cycles} cycles "
                        f"(full pass cycles {case_full_pass_cycles}/{total_cycles})"
                    )

            summary["cases"].append(case_block)
            console.print("")

        if summary["total_cases"] > 0:
            summary["pass_rate"] = summary["passed_cases"] / summary["total_cases"]
        else:
            summary["pass_rate"] = 0.0

        if write_summary:
            summary_path = self.version_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            substep(f"summary saved -> {summary_path}")

        PromptEvaluator.print_summary_table(summary)
        return summary

    async def _call_eval(
        self,
        case_name: str,
        case_data: Dict[str, Any],
        output_data: Any,
        original_task: str,
    ) -> EvalOut:
        case_json = json.dumps(case_data, indent=2)
        output_json = json.dumps(output_data.model_dump(), indent=2)

        eval_criteria = Composer._load_template(
            "eval",
            sub_dir=str(self.test_dir),
            case_name=case_name,
        )

        criteria_names: List[str] = []
        for line in eval_criteria.splitlines():
            m = re.match(r"^#\s*(.+)$", line.strip())
            if m:
                criteria_names.append(m.group(1).strip())

        system_prompt = Composer._load_template(
            "system_prompt",
            sub_dir="agent/config/eval",
            case_name=case_name,
        )

        user_prompts = [
            eval_criteria,
            f"Original task the model was asked to solve:\n{original_task}",
            f"Agent output to assess:\n{output_json}",
            f"Case data (for context):\n{case_json}",
        ]

        logger.info("[EVAL] Evaluating case '%s'", case_name)

        eval_result = await self.runner.generate(
            system_prompts=[system_prompt],
            user_prompts=user_prompts,
            response_model=EvalOut,
            model=self.config.eval_model,
        )

        by_name: Dict[str, CriteriaResult] = {
            obj.criteria_name: obj for obj in eval_result.objectives
        }

        aligned_objectives: List[CriteriaResult] = []
        for name in criteria_names:
            aligned_objectives.append(
                by_name.get(
                    name,
                    CriteriaResult(criteria_name=name, criteria_passed=False),
                )
            )

        eval_result.objectives = aligned_objectives
        return eval_result

    def print_summary_table(summary: dict):
        print("")
        print("=== SUMMARY TABLE ===")
        print(f"Version: {summary['version']}")
        print(f"Pass rate: {summary['pass_rate']:.2f}")

        cycles = summary.get("cycles")
        if cycles is not None:
            print(f"Cycles: {cycles}")
        print("")

        headers = ["Case", "Criteria", "Score"]
        print(f"{headers[0]:<20} | {headers[1]:<20} | {headers[2]:<6}")
        print(f"{'-'*20} | {'-'*20} | {'-'*6}")

        for case in summary["cases"]:
            case_name = case["case_name"]

            objectives = case.get("objectives", [])
            if not objectives:
                print(f"{case_name:<20} | {'(no criteria)':<20} | {'0.00':<6}")
                print(f"{'-'*20} | {'-'*20} | {'-'*6}")
                continue

            for idx, obj in enumerate(objectives):
                crit_name = list(obj.keys())[0]
                value = obj[crit_name]
                if isinstance(value, bool):
                    score_str = "1.00" if value else "0.00"
                else:
                    score_str = f"{float(value):.2f}"

                if idx == 0:
                    print(f"{case_name:<20} | {crit_name:<20} | {score_str:<6}")
                else:
                    print(f"{'':<20} | {crit_name:<20} | {score_str:<6}")

            print(f"{'-'*20} | {'-'*20} | {'-'*6}")
