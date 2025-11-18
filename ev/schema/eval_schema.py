from typing import List, Optional
from pydantic import BaseModel


class CriteriaResult(BaseModel):
    criteria_name: str
    criteria_passed: bool

class EvalOut(BaseModel):
    name: str
    objectives: List[CriteriaResult]
    max_iterations: Optional[int] = None


class PromptImprovement(BaseModel):
    notes: str
    new_system_prompt: Optional[str] = None
    new_user_prompt: Optional[str] = None
