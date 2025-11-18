import re
import json
from typing import Optional, Type, Any
from pydantic import BaseModel, ValidationError
from openai import AsyncOpenAI
from groq import AsyncGroq

from ev.core.config import settings
from ev.agents.composer import Composer
from dataclasses import dataclass
from typing import Dict, Any



@dataclass(frozen=True)
class ModelConfig:
    provider: str                 # "openai" or "groq"
    name: str                     # model string to send
    extra_params: Dict[str, Any]  # provider/model-specific kwargs


class AvailableModels:
    class openai:
        gpt5 = ModelConfig(
            provider="openai",
            name="gpt-5",
            extra_params={"response_format": {"type": "json_object"}},
        )
        gpt5_mini = ModelConfig(
            provider="openai",
            name="gpt-5-mini",
            extra_params={"response_format": {"type": "json_object"}},
        )
        gpt5_nano = ModelConfig(
            provider="openai",
            name="gpt-5-nano",
            extra_params={"response_format": {"type": "json_object"}},
        )

    class groq:
        gpt_oss_120b = ModelConfig(
            provider="groq",
            name="openai/gpt-oss-120b",
            extra_params={"temperature": 0, "top_p": 1, "max_tokens": 4096},
        )
        qwen3_32b = ModelConfig(
            provider="groq",
            name="qwen/qwen3-32b",
            extra_params={
                "temperature": 0.3,
                "top_p": 0.95,
                "max_completion_tokens": 4096,
                "reasoning_effort": "none",
                "response_format": {"type": "json_object"},
            },
        )
        kimi_k2_instruct = ModelConfig(
            provider="groq",
            name="moonshotai/kimi-k2-instruct", #moonshotai/kimi-k2-instruct-0905
            extra_params={"temperature": 0, "top_p": 1, "max_tokens": 4096},
        )


class Runner:
    def __init__(
        self,
        ctx: Optional[Any] = None,
    ):
        self.ctx = ctx

        # Init clients once
        self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)


    @staticmethod
    def load_json_response(response_text: str) -> dict:

        if not response_text or not response_text.strip():
            raise ValueError("Empty LLM response")

        response_text = response_text.strip()

        # get fenced code block ```json ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
        if match:
            raw_json = match.group(1).strip()
            return json.loads(raw_json)

        # clear <think>...</think> or other XML-ish wrappers - reaosning models
        response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

        # find the first {...} JSON block in the text
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            raw_json = match.group(0).strip()
            return json.loads(raw_json)

        raise ValueError(f"No JSON object found in LLM response: {response_text[:200]}")



    async def generate(
        self,
        user_prompts: list[str],
        system_prompts: list[str],
        response_model: Type[BaseModel],
        max_retries: int = 2,
        model: ModelConfig = AvailableModels.groq.qwen3_32b,
    ) -> BaseModel:
        schema_json = response_model.model_json_schema()
        schema_str = json.dumps(schema_json, indent=2)

        schema_instruction = Composer._load_template(
            "schema_enforcer",
            sub_dir="agents/schema",
            schema_str=schema_str
        )

        messages = [{"role": "system", "content": schema_instruction.strip()}]
        messages += [{"role": "system", "content": prompt.strip()} for prompt in system_prompts]
        messages += [{"role": "user", "content": prompt.strip()} for prompt in user_prompts]


        validation_error: str | None = None

        for attempt in range(max_retries):
            try:
                if validation_error:
                    messages.append({
                        "role": "system",
                        "content": f"[Attempt {attempt+1}] Validation failed: {validation_error}"
                                "\nPlease fix output to strictly conform to schema."
                    })

                # === OpenAI branch ===
                if model.provider == "openai":
                    response = await self.openai_client.chat.completions.create(
                        model=model.name,
                        messages=messages,
                        **model.extra_params,
                    )
                    raw = response.choices[0].message.content

                # === Groq branch ===
                elif model.provider == "groq":
                    response = await self.groq_client.chat.completions.create(
                        model=model.name,
                        messages=messages,
                        stream=False,
                        stop=None,
                        **model.extra_params,
                    )
                    raw = response.choices[0].message.content

                else:
                    raise ValueError(f"Unknown provider for model {model}")

                if model.extra_params.get("response_format", {}).get("type") == "json_object":
                    try:
                        parsed = json.loads(raw)  # strict mode
                    except json.JSONDecodeError:
                        parsed = Runner.load_json_response(raw)  # fallback
                else:
                    parsed = Runner.load_json_response(raw)

                return response_model.model_validate(parsed)


            except (json.JSONDecodeError, ValidationError) as parse_error:
                print(f"[generate parse error #{attempt+1}] {parse_error}")

                # Show raw response if available
                if "raw" in locals():
                    print(f"[generate raw output #{attempt+1}]:\n{raw}\n")

                validation_error = str(parse_error)

                if attempt == max_retries - 1:
                    raise ValueError(
                        f"Failed to parse or validate LLM output after {max_retries} attempts. "
                        f"Last error: {validation_error}\nRaw output: {raw if 'raw' in locals() else 'N/A'}"
                    )

