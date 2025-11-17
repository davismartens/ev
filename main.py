import argparse
import asyncio
import re
import json
import logging
import uuid
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, Field

# Your existing runner and model config
from agents.runner import Runner, ModelConfig, AvailableModels
from agents.composer import Composer
from datetime import datetime

# ---------------------------------------------------------------------
# Pydantic models for eval output
# ---------------------------------------------------------------------


class CriteriaResult(BaseModel):
    criteria_name: str
    criteria_passed: bool

class EvalOut(BaseModel):
    name: str
    objectives: List[CriteriaResult]
    max_iterations: Optional[int] = None


# ---------------------------------------------------------------------
# Config and logging
# ---------------------------------------------------------------------

logger = logging.getLogger("eval_runner")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Root directory that contains the EVALS folder
# This script assumes it lives in C:\USERS\DAVIS\DEV-PROJECTS\EVAL\eval_runner.py
ROOT_DIR = Path(__file__).resolve().parent
EVALS_ROOT = ROOT_DIR / "EVALS"

@dataclass
class EvalConfig:
    test_name: str
    version_id: str
    generation_model: ModelConfig = AvailableModels.groq.kimi_k2_instruct
    eval_model: ModelConfig = AvailableModels.groq.kimi_k2_instruct



# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def resolve_dotted_class(path: str) -> Any:
    module_name, class_name = path.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, class_name)


def find_latest_version_id(versions_dir: Path) -> Optional[str]:
    """
    Return the latest version id by modification time, or None if none exist.
    """
    if not versions_dir.exists():
        return None

    candidates = [p for p in versions_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.name


def snapshot_prompts(test_dir: Path) -> str:
    versions_dir = test_dir / "versions"
    versions_dir.mkdir(exist_ok=True)

    system_src = test_dir / "system_prompt.j2"
    user_src = test_dir / "user_prompt.j2"

    missing = [str(p.name) for p in [system_src, user_src] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot snapshot prompts, missing: {', '.join(missing)} "
            f"in {test_dir}"
        )


    u = str(uuid.uuid4())
    h = u[:8]
    timestamp = datetime.now().strftime("%d %b %Y %H-%M-%S")
    timestamp_safe = timestamp.replace(":", "-")
    version_id = f"{h} - {timestamp_safe}"
    version_dir = versions_dir / version_id

    version_dir.mkdir(parents=True, exist_ok=False)

    system_dst = version_dir / "system_prompt.j2"
    user_dst = version_dir / "user_prompt.j2"

    system_dst.write_text(system_src.read_text(encoding="utf-8"), encoding="utf-8")
    user_dst.write_text(user_src.read_text(encoding="utf-8"), encoding="utf-8")

    logger.info(f"Created version {version_id} in {version_dir}")
    return version_id


# ---------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------


class PromptEvaluator:
    def __init__(self, config: EvalConfig) -> None:
        self.config = config
        self.test_dir = EVALS_ROOT / config.test_name
        self.cases_dir = self.test_dir / "cases"
        self.versions_dir = self.test_dir / "versions"
        self.version_dir = self.versions_dir / config.version_id
        self.results_dir = self.test_dir / "results" / config.version_id
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if not self.version_dir.exists():
            raise FileNotFoundError(f"Version dir not found: {self.version_dir}")

        self.eval_md_path = self.test_dir / "eval.md"
        if not self.eval_md_path.exists():
            raise FileNotFoundError(f"Missing eval.md in {self.test_dir}")

        if not self.cases_dir.exists():
            raise FileNotFoundError(f"Missing cases/ directory in {self.test_dir}")

        schema_path = self.test_dir / "schema.py"
        if not schema_path.exists():
            raise FileNotFoundError(f"Missing schema.py in {self.test_dir}")

        module_name = f"evals.{config.test_name}.schema"
        module = import_module(module_name)

        response_model = None
        for attr in dir(module):
            obj = getattr(module, attr)
            if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel:
                response_model = obj
                break

        if response_model is None:
            raise RuntimeError("No valid Pydantic BaseModel found in schema.py")

        self.response_model = response_model

        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.version_dir)),
            undefined=StrictUndefined,
            autoescape=False,
        )

        self.runner = Runner()


    def _render_prompts(self, case_data: Dict[str, Any]) -> Dict[str, str]:
        case_json = json.dumps(case_data, indent=2)

        # context used by templates
        context = {"case": case_data, "case_json": case_json}
        context.update(case_data)

        system_prompt = Composer._load_template(
            "system_prompt",
            base_dir=str(self.version_dir),
            **context,
        )

        user_prompt = Composer._load_template(
            "user_prompt",
            base_dir=str(self.version_dir),
            **context,
        )

        return {
            "system": system_prompt,
            "user": user_prompt,
        }


    async def _call_generation(
        self, system_prompt: str, user_prompt: str
    ) -> Any:
        logger.info("[GEN] Calling LLM for case with %s", self.config.generation_model)

        result = await self.runner.generate(
            user_prompts=[user_prompt],
            system_prompts=[system_prompt],
            response_model=self.response_model,
            model=self.config.generation_model,
        )
        return result


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
            base_dir=str(self.test_dir),
            case_name=case_name,
        )

        criteria_names: List[str] = []
        for line in eval_criteria.splitlines():
            m = re.match(r"^#\s*(.+)$", line.strip())
            if m:
                name = m.group(1).strip()
                criteria_names.append(name)

        system_prompt = Composer._load_template(
            "system_prompt",
            base_dir="agents/config/eval",
            case_name=case_name,
        )

        user_prompts = [
            eval_criteria,
            f"Original task the model was asked to solve:\n{original_task}",
            f"Case data (for context):\n{case_json}",
            f"Output data to assess:\n{output_json}",
        ]

        logger.info("[EVAL] Evaluating case '%s'", case_name)

        eval_result = await self.runner.generate(
            system_prompts=[system_prompt],
            user_prompts=user_prompts,
            response_model=EvalOut,
            model=self.config.eval_model,
        )

        by_name: Dict[str, CriteriaResult] = {}
        for obj in eval_result.objectives:
            by_name[obj.criteria_name] = obj

        aligned_objectives: List[CriteriaResult] = []
        for name in criteria_names:
            obj = by_name.get(name)
            if obj is None:
                obj = CriteriaResult(criteria_name=name, criteria_passed=False)
            aligned_objectives.append(obj)

        eval_result.objectives = aligned_objectives

        return eval_result



    async def run_all_cases(self) -> None:
        case_files = sorted(self.cases_dir.glob("*.json"))
        if not case_files:
            logger.warning("No case JSON files found in %s", self.cases_dir)
            return

        logger.info(
            "Running %d cases for test '%s', version '%s'",
            len(case_files),
            self.config.test_name,
            self.config.version_id,
        )

        summary = {
            "version": self.config.version_id,
            "total_cases": len(case_files),
            "passed_cases": 0,
            "pass_rate": 0.0,
            "cases": [],
        }

        for case_file in case_files:
            case_name = case_file.stem
            logger.info("[CASE] %s", case_name)

            case_data = json.loads(case_file.read_text(encoding="utf-8"))
            prompts = self._render_prompts(case_data)

            output_data = await self._call_generation(
                prompts["system"],
                prompts["user"],
            )

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

            summary["cases"].append(case_block)

        if summary["total_cases"] > 0:
            summary["pass_rate"] = summary["passed_cases"] / summary["total_cases"]
        else:
            summary["pass_rate"] = 0.0

        summary_path = self.results_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("[SAVE] Wrote summary to %s", summary_path)

        PromptEvaluator.update_log_pass_rate(
            self.versions_dir,
            self.config.version_id,
            summary["pass_rate"],
        )


    @staticmethod
    def load_active_version(test_dir: Path) -> str:
        versions_dir = test_dir / "versions"
        versions_dir.mkdir(exist_ok=True)

        log_path = versions_dir / "log.json"

        # If log.json does not exist, initialize it
        if not log_path.exists():
            # If there are existing version dirs, pick the latest by mtime
            candidates = [p for p in versions_dir.iterdir() if p.is_dir()]
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                version_id = latest.name
            else:
                # No versions yet; snapshot from root system/user prompts
                version_id = snapshot_prompts(test_dir)

            entries = [
                {
                    "version": version_id,
                    "pass_rate": 0.0,
                    "is_active": True,
                }
            ]
            log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
            return version_id

        # log.json exists; load it
        entries = json.loads(log_path.read_text(encoding="utf-8"))

        # Try to find an active version
        for entry in entries:
            if entry.get("is_active"):
                return entry["version"]

        # No active version marked; choose or create one
        if entries:
            entries[0]["is_active"] = True
            version_id = entries[0]["version"]
        else:
            candidates = [p for p in versions_dir.iterdir() if p.is_dir()]
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                version_id = latest.name
            else:
                version_id = snapshot_prompts(test_dir)

            entries = [
                {
                    "version": version_id,
                    "pass_rate": 0.0,
                    "is_active": True,
                }
            ]

        log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return version_id



    def update_log_pass_rate(versions_dir: Path, version_id: str, pass_rate: float) -> None:
        log_path = versions_dir / "log.json"
        if not log_path.exists():
            return

        entries = json.loads(log_path.read_text(encoding="utf-8"))

        for entry in entries:
            if entry.get("version") == version_id:
                entry["pass_rate"] = pass_rate
                break

        log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")



# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prompt eval harness for LLM agents."
    )
    parser.add_argument(
        "test_name",
        help="Name of the test folder under EVALS (for example 'test1').",
    )
    parser.add_argument(
        "--version",
        help=(
            "Version id under versions/ to use. "
            "If omitted, uses the latest existing version."
        ),
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help=(
            "If set, snapshot current system_prompt.j2 and user_prompt.j2 "
            "from the test root into a new versions/<short_id>/ folder and "
            "run evals for that new version."
        ),
    )
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()

    test_dir = EVALS_ROOT / args.test_name
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    # Always resolve the active version via log.json, creating it if needed
    version_id = PromptEvaluator.load_active_version(test_dir)

    config = EvalConfig(
        test_name=args.test_name,
        version_id=version_id,
    )

    evaluator = PromptEvaluator(config)
    await evaluator.run_all_cases()



def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
