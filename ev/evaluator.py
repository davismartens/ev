import json
import re
import sys
import importlib.util
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from ev.utils.pretty import console, step, substep, success, fail, spinner

from ev.agents.runner import Runner, ModelConfig, AvailableModels
from ev.agents.composer import Composer
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


    async def run_all_cases(self, write_summary: bool = True) -> Dict[str, Any]:
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

        step(f"Running {len(case_files)} cases")
        substep(f"test: {self.config.test_name}")
        substep(f"version: {self.config.version_id}")
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

            # render prompts (fast: no spinner)
            substep("rendering prompts")
            prompts = self._render_prompts(case_data)

            # call generation (slow: keep spinner)
            with spinner() as prog:
                prog.add_task("generating model output…", total=None)
                output_data = await self._call_generation(
                    prompts["system"],
                    prompts["user"],
                )

            # eval run (slow: keep spinner)
            with spinner() as prog:
                prog.add_task("running evaluation…", total=None)
                eval_out = await self._call_eval(
                    case_name=case_name,
                    case_data=case_data,
                    output_data=output_data,
                    original_task=prompts["user"],
                )

            objectives_list: List[Dict[str, bool]] = []
            passed_count = 0

            if eval_out.objectives:
                for obj in eval_out.objectives:
                    name = obj.criteria_name[:20]
                    objectives_list.append({name: obj.criteria_passed})
                    if obj.criteria_passed:
                        passed_count += 1

                pass_fraction = passed_count / len(eval_out.objectives)
            else:
                pass_fraction = 0.0

            case_block = {
                "case_name": case_name,
                "objectives": objectives_list,
                "pass_rate": pass_fraction,
            }

            if passed_count == len(eval_out.objectives) and len(eval_out.objectives) > 0:
                summary["passed_cases"] += 1
                success(f"case {case_name} passed {passed_count}/{len(eval_out.objectives)}")
            else:
                fail(f"case {case_name} passed {passed_count}/{len(eval_out.objectives)}")

            summary["cases"].append(case_block)
            console.print("")  # spacing

        if summary["total_cases"] > 0:
            summary["pass_rate"] = summary["passed_cases"] / summary["total_cases"]
        else:
            summary["pass_rate"] = 0.0

        if write_summary:
            summary_path = self.version_dir / "summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            substep(f"summary saved -> {summary_path}")

        # pretty table
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
            sub_dir="agents/config/eval",
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
        print("")

        # Build header
        headers = ["Case", "Criteria", "Passed"]
        print(f"{headers[0]:<20} | {headers[1]:<20} | {headers[2]:<6}")
        print(f"{'-'*20} | {'-'*20} | {'-'*6}")

        for case in summary["cases"]:
            case_name = case["case_name"]

            for idx, obj in enumerate(case["objectives"]):
                # key = only key in the dict, ex: {"decision": true}
                crit_name = list(obj.keys())[0]
                passed = "✅" if obj[crit_name] else "❌"

                if idx == 0:
                    # print case name on first row
                    print(f"{case_name:<20} | {crit_name:<20} | {passed:<6}")
                else:
                    # subsequent rows indent case column
                    print(f"{'':<20} | {crit_name:<20} | {passed:<6}")

            print(f"{'-'*20} | {'-'*20} | {'-'*6}")
